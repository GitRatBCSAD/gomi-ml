#!/usr/bin/env python3
"""
Extract Lizard structural complexity metrics for every ApacheJIT commit.

Processes both apachejit_train_with_messages.csv and apachejit_test_small.csv.
For each commit:
  1. Clones the project repo (full bare, no filter)
  2. Batch-indexes all commit hashes via git cat-file --batch-check
  3. Lists changed source files (git diff-tree --diff-filter=AM)
  4. Reads file content (git cat-file --batch per commit)
  5. Runs Lizard → AvgCCN, AvgNLOC, FuncCount, AvgParams
  6. Averages across changed files

Output:
  dataset/apachejit_train_lizard.csv
  dataset/apachejit_test_lizard.csv

Columns: commit_id, project, avg_ccn, avg_nloc, func_count, avg_params, file_count, buggy

Usage:
    python scripts/extract_apache_lizard.py
    python scripts/extract_apache_lizard.py --verify
    python scripts/extract_apache_lizard.py --workers 64
    python scripts/extract_apache_lizard.py --split train   # only train
    python scripts/extract_apache_lizard.py --split test    # only test
"""

import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        total = kw.get("total", "?")
        desc  = kw.get("desc", "")
        for i, x in enumerate(it):
            if i % 200 == 0:
                print(f"  {desc}: {i}/{total}")
            yield x

try:
    import lizard
except ImportError:
    print("ERROR: lizard not installed. Run: pip install lizard")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent.parent / "dataset"
CLONE_DIR   = DATASET_DIR / ".clones" / "apache"

HF_DATASET_REPO = "GitRatBCSAD/gomi-datasets"

INPUT_FILES = {
    "train": "jit/apachejit/apachejit_train_with_messages.csv",
    "test":  "jit/apachejit/apachejit_test_small.csv",
}
OUTPUT_FILES = {
    "train": DATASET_DIR / "apachejit_train_lizard.csv",
    "test":  DATASET_DIR / "apachejit_test_lizard.csv",
}

# apache/hadoop-hdfs and apache/hadoop-mapreduce were merged into apache/hadoop
REPO_OVERRIDES = {
    "apache/hadoop-hdfs":     "apache/hadoop",
    "apache/hadoop-mapreduce": "apache/hadoop",
}

CODE_EXTENSIONS = {
    ".py", ".java", ".go", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp",
    ".cs", ".js", ".ts", ".rb", ".rs", ".scala", ".kt", ".swift", ".php", ".m",
}

BATCH_TIMEOUT = 120


# ── HF token ──────────────────────────────────────────────────────────────────

def get_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    try:
        from huggingface_hub import HfFolder
        cached = HfFolder.get_token()
        if cached:
            return cached
    except Exception:
        pass
    return None


# ── Git helpers ────────────────────────────────────────────────────────────────

def clone_repo(project: str) -> Path | None:
    canonical = REPO_OVERRIDES.get(project, project)
    url  = f"https://github.com/{canonical}.git"
    name = canonical.replace("/", "_")
    dest = CLONE_DIR / name
    if dest.exists():
        return dest
    print(f"  cloning {url}")
    r = subprocess.run(
        ["git", "clone", "--bare", url, str(dest)],
        capture_output=True,
    )
    if r.returncode != 0:
        print(f"  FAILED {url}: {r.stderr.decode()[:200]}")
        return None
    return dest


def hashes_in_repo(repo: Path, hashes: list) -> tuple:
    inp = "\n".join(hashes).encode()
    try:
        r = subprocess.run(
            ["git", "--git-dir", str(repo),
             "cat-file", "--batch-check=%(objectname) %(objecttype)"],
            input=inp, capture_output=True, timeout=BATCH_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT indexing {repo.name}")
        return repo, set()
    found = set()
    for line in r.stdout.decode(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] != "missing":
            found.add(parts[0])
    return repo, found


def build_hash_index(repos: dict, hashes: list) -> dict:
    """repos: {project: repo_path}. Returns {hash: repo_path}."""
    repo_paths = list(set(repos.values()))
    print(f"  batch-indexing {len(hashes)} hashes across {len(repo_paths)} repos...")

    index = {}
    done  = 0
    with ThreadPoolExecutor(max_workers=len(repo_paths)) as ex:
        futures = {ex.submit(hashes_in_repo, rp, hashes): rp for rp in repo_paths}
        for fut in as_completed(futures):
            rp, found = fut.result()
            done += 1
            for h in found:
                if h not in index:
                    index[h] = rp
            print(f"  indexed {done}/{len(repo_paths)} repos  (found: {len(index)})",
                  end="\r", flush=True)
    print()
    missing = len(hashes) - len(index)
    print(f"  found {len(index)}/{len(hashes)}  |  {missing} not in any repo")
    return index


def get_changed_code_files(repo: Path, commit_hash: str) -> list:
    r = subprocess.run(
        ["git", "--git-dir", str(repo), "diff-tree",
         "--no-commit-id", "-r", "--name-only", "--diff-filter=AM", commit_hash],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return []
    files = [f.strip() for f in r.stdout.strip().splitlines() if f.strip()]
    return [f for f in files if Path(f).suffix.lower() in CODE_EXTENSIONS]


def get_files_content_batch(repo: Path, commit_hash: str, filepaths: list) -> dict:
    if not filepaths:
        return {}
    requests = "".join(f"{commit_hash}:{fp}\n" for fp in filepaths)
    r = subprocess.run(
        ["git", "--git-dir", str(repo), "cat-file", "--batch"],
        input=requests.encode(), capture_output=True,
    )
    if r.returncode != 0:
        return {}
    results = {}
    data = r.stdout
    pos  = 0
    for fp in filepaths:
        if pos >= len(data):
            break
        nl = data.find(b'\n', pos)
        if nl == -1:
            break
        header = data[pos:nl].decode(errors="replace")
        pos = nl + 1
        parts = header.split()
        if len(parts) < 3 or parts[1] == "missing":
            continue
        size = int(parts[2])
        results[fp] = data[pos:pos + size].decode("utf-8", errors="replace")
        pos += size + 1
    return results


def run_lizard(filepath: str, source: str):
    info  = lizard.analyze_file.analyze_source_code(filepath, source)
    funcs = info.function_list
    if not funcs:
        return None
    return {
        "avg_ccn":    sum(fn.cyclomatic_complexity for fn in funcs) / len(funcs),
        "avg_nloc":   sum(fn.nloc                  for fn in funcs) / len(funcs),
        "avg_params": sum(fn.parameter_count       for fn in funcs) / len(funcs),
        "func_count": len(funcs),
    }


# ── Per-commit worker ──────────────────────────────────────────────────────────

def process_one_commit(args):
    idx, commit_id, project, buggy, hash_index = args
    base  = {"commit_id": commit_id, "project": project, "buggy": buggy}
    empty = {**base, "avg_ccn": "", "avg_nloc": "", "func_count": 0,
             "avg_params": "", "file_count": 0}

    repo = hash_index.get(commit_id)
    if repo is None:
        return idx, empty, "no_repo"

    code_files = get_changed_code_files(repo, commit_id)
    if not code_files:
        return idx, empty, "no_code"

    contents = get_files_content_batch(repo, commit_id, code_files)
    per_file  = [run_lizard(fp, c) for fp, c in contents.items()]
    per_file  = [m for m in per_file if m]

    if not per_file:
        return idx, {**empty, "file_count": len(code_files)}, "no_funcs"

    n = len(per_file)
    return idx, {
        **base,
        "avg_ccn":    round(sum(m["avg_ccn"]    for m in per_file) / n, 4),
        "avg_nloc":   round(sum(m["avg_nloc"]   for m in per_file) / n, 4),
        "func_count": sum(m["func_count"] for m in per_file),
        "avg_params": round(sum(m["avg_params"] for m in per_file) / n, 4),
        "file_count": n,
    }, "ok"


# ── Process one split ──────────────────────────────────────────────────────────

def process_split(split: str, rows: list, hash_index: dict, workers: int) -> Path:
    out_path = OUTPUT_FILES[split]
    fields   = ["commit_id", "project", "avg_ccn", "avg_nloc",
                "func_count", "avg_params", "file_count", "buggy"]

    work = [(i, r["commit_id"], r["project"], r["buggy"], hash_index)
            for i, r in enumerate(rows)]

    results = [None] * len(rows)
    counts  = {"ok": 0, "no_repo": 0, "no_code": 0, "no_funcs": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one_commit, item): item[0] for item in work}
        for fut in tqdm(as_completed(futures), total=len(rows), desc=split):
            try:
                idx, row, status = fut.result()
            except Exception:
                idx = futures[fut]
                r   = rows[idx]
                row = {"commit_id": r["commit_id"], "project": r["project"],
                       "buggy": r["buggy"], "avg_ccn": "", "avg_nloc": "",
                       "func_count": 0, "avg_params": "", "file_count": 0}
                status = "error"
            results[idx] = row
            counts[status] = counts.get(status, 0) + 1

    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    total = len(rows)
    print(f"  [{split}] {counts['ok']}/{total} analyzed  |  "
          f"no_repo={counts['no_repo']}  no_code={counts['no_code']}  "
          f"no_funcs={counts['no_funcs']}  errors={counts['error']}")
    print(f"  → {out_path}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    verify_only = "--verify" in sys.argv
    workers     = 64
    split_only  = None
    if "--workers" in sys.argv:
        idx = sys.argv.index("--workers")
        try:
            workers = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            pass
    if "--split" in sys.argv:
        idx = sys.argv.index("--split")
        split_only = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
    return verify_only, workers, split_only


def main():
    verify_only, workers, split_only = parse_args()

    token = get_hf_token()
    if not token:
        print("ERROR: No HF token. Fix: export HF_TOKEN=... OR huggingface-cli login")
        sys.exit(1)

    from huggingface_hub import hf_hub_download

    CLONE_DIR.mkdir(parents=True, exist_ok=True)

    splits = [split_only] if split_only else list(INPUT_FILES.keys())

    # ── Load all rows across splits to discover all projects ──────────────────
    all_rows   = {}
    all_projs  = set()
    for split in splits:
        print(f"\nLoading {split} CSV from HF...")
        path = hf_hub_download(HF_DATASET_REPO, INPUT_FILES[split],
                               repo_type="dataset", token=token)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            all_projs.add(r.get("project", "").strip())
        all_rows[split] = rows
        print(f"  {len(rows)} rows, {len(all_projs)} unique projects so far")

    # ── Clone all repos ────────────────────────────────────────────────────────
    print(f"\nCloning {len(all_projs)} repos...")
    repo_map = {}  # project → Path
    for proj in sorted(all_projs):
        canonical = REPO_OVERRIDES.get(proj, proj)
        repo = clone_repo(proj)
        if repo:
            repo_map[proj] = repo
            # Also map canonical name in case multiple projects share a repo
            repo_map[canonical] = repo
    print(f"  {len(set(repo_map.values()))} distinct repos ready")

    # ── Build hash index across all splits ────────────────────────────────────
    all_hashes = list({r["commit_id"] for rows in all_rows.values() for r in rows})
    print(f"\nBuilding hash index for {len(all_hashes)} unique commits...")
    hash_index = build_hash_index(repo_map, all_hashes)

    if verify_only:
        sample    = all_hashes[:200]
        found     = sum(1 for h in sample if h in hash_index)
        print(f"\nVerify: {found}/{len(sample)} sample hashes found ({found/len(sample)*100:.0f}%)")
        print("Re-run without --verify to process.")
        return

    # ── Process each split ────────────────────────────────────────────────────
    out_files = []
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}  ({len(all_rows[split])} commits)")
        out = process_split(split, all_rows[split], hash_index, workers)
        out_files.append(out)

    print(f"\n{'='*60}")
    print("Done. Output files:")
    for f in out_files:
        print(f"  {f}")
    print()
    print("Next steps:")
    print("  1. Upload apachejit_*_lizard.csv to HuggingFace dataset repo.")
    print("  2. Update notebook cells 14 + 15 to load and join Lizard data.")


if __name__ == "__main__":
    main()
