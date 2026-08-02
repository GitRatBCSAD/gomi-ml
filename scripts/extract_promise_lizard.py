#!/usr/bin/env python3
"""
Extract Lizard structural complexity metrics for every PROMISE-recovered commit.

Input:  dataset/promise_{project}_recovered.csv
        (from scripts/recover_promise_messages.py — commit_hash, project,
         message, buggy, author_date_unix_timestamp, + raw Kamei fields)

35 projects (4 of the original 39 — activemq/kafka/zeppelin/zookeeper —
excluded, already in ApacheJIT). One repo each, all verified apache/<name>
GitHub URLs. Full clones, same shared dataset/.clones/ dir as the other
extract_*_lizard.py scripts.

For each commit:
  1. List changed source files (git diff-tree --cc --diff-filter=AM — the
     --cc handles merge commits correctly if any exist; byte-identical to
     plain diff-tree for non-merges, so unconditionally safe). Note: PROMISE's
     own CSV already lists the changed `file` per row, but this script re-derives
     the file list itself via diff-tree for consistency with the other three
     extraction scripts (DRY — same function, not a special case).
  2. Read file content (git cat-file --batch, fully local)
  3. Run Lizard -> AvgCCN, AvgNLOC, FuncCount, AvgParams
  4. Average across changed files

Output: dataset/promise_{project}_lizard.csv
Columns: commit_id, project, avg_ccn, avg_nloc, func_count, avg_params, file_count, buggy

Usage:
    python scripts/extract_promise_lizard.py                  # all 35 projects
    python scripts/extract_promise_lizard.py --project pig     # single project
    python scripts/extract_promise_lizard.py --verify          # sample only, no CSV written
    python scripts/extract_promise_lizard.py --workers 16
"""

import csv
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

DATASET_DIR = Path(__file__).parent.parent / "dataset"
CLONE_DIR   = DATASET_DIR / ".clones"  # shared with the other extract_*_lizard.py scripts

# Excludes activemq/kafka/zeppelin/zookeeper — already in ApacheJIT.
# Also excludes pig and pdfbox — their commit hashes don't exist in the
# current GitHub mirror at all (history rewrite ~April 2014, see
# recover_promise_messages.py for the full explanation). manifoldcf kept,
# recovered CSV already trimmed to only its post-migration commits.
# All URLs verified via `git ls-remote` before writing this script.
PROJECTS = [
    "archiva", "calcite", "cayenne", "commons-jexl", "deltaspike", "derby",
    "directory-fortress-core", "eagle", "falcon", "flume", "helix",
    "httpcomponents-client", "httpcomponents-core", "jena", "jspwiki",
    "knox", "kylin", "lens", "mahout", "manifoldcf", "mina-sshd", "nifi",
    "nutch", "oozie", "phoenix", "ranger", "roller",
    "samza", "storm", "streams", "systemml", "tez", "tika",
]

CODE_EXTENSIONS = {
    ".py", ".java", ".go", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp",
    ".cs", ".js", ".ts", ".rb", ".rs", ".scala", ".kt", ".swift", ".php", ".m",
}

MAX_FILE_BYTES = 1_000_000  # ~1MB — skip minified/vendored/generated blobs before Lizard
GIT_TIMEOUT = 60  # seconds — safety net against any git subprocess hanging


def clone_repo(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    print(f"  cloning {url}")
    r = subprocess.run(
        ["git", "clone", "--bare", "--quiet", url, str(dest)],
        capture_output=True,
    )
    if r.returncode != 0:
        print(f"  FAILED {url}: {r.stderr.decode()[:300]}")
        return False
    return True


def load_recovered_csv(project: str) -> list:
    path = DATASET_DIR / f"promise_{project}_recovered.csv"
    if not path.exists():
        print(f"  [{project}] missing {path} — run recover_promise_messages.py first")
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_changed_code_files(repo: Path, commit_hash: str) -> list:
    try:
        r = subprocess.run(
            ["git", "--git-dir", str(repo), "diff-tree", "--cc",
             "--no-commit-id", "-r", "--name-only", "--diff-filter=AM", commit_hash],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return []
    if r.returncode != 0:
        return []
    files = [f.strip() for f in r.stdout.strip().splitlines() if f.strip()]
    return [f for f in files if Path(f).suffix.lower() in CODE_EXTENSIONS]


def get_files_content_batch(repo: Path, commit_hash: str, filepaths: list) -> dict:
    if not filepaths:
        return {}
    requests = "".join(f"{commit_hash}:{fp}\n" for fp in filepaths)
    try:
        r = subprocess.run(
            ["git", "--git-dir", str(repo), "cat-file", "--batch"],
            input=requests.encode(), capture_output=True, timeout=GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {}
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
        if size <= MAX_FILE_BYTES:
            results[fp] = data[pos:pos + size].decode("utf-8", errors="replace")
        pos += size + 1
    return results


def run_lizard(filepath: str, source: str):
    try:
        info  = lizard.analyze_file.analyze_source_code(filepath, source)
        funcs = info.function_list
    except Exception:
        return None
    if not funcs:
        return None
    return {
        "avg_ccn":    sum(fn.cyclomatic_complexity for fn in funcs) / len(funcs),
        "avg_nloc":   sum(fn.nloc                  for fn in funcs) / len(funcs),
        "avg_params": sum(fn.parameter_count       for fn in funcs) / len(funcs),
        "func_count": len(funcs),
    }


def process_one_commit(args):
    commit_hash, project, buggy, repo = args
    base  = {"commit_id": commit_hash, "project": project, "buggy": buggy}
    empty = {**base, "avg_ccn": "", "avg_nloc": "", "func_count": 0,
             "avg_params": "", "file_count": 0}

    code_files = get_changed_code_files(repo, commit_hash)
    if not code_files:
        return empty, "no_code"

    contents = get_files_content_batch(repo, commit_hash, code_files)
    per_file = [m for m in (run_lizard(fp, c) for fp, c in contents.items()) if m]

    if not per_file:
        return {**empty, "file_count": len(code_files)}, "no_funcs"

    n = len(per_file)
    return {
        **base,
        "avg_ccn":    round(sum(m["avg_ccn"]    for m in per_file) / n, 4),
        "avg_nloc":   round(sum(m["avg_nloc"]   for m in per_file) / n, 4),
        "func_count": sum(m["func_count"] for m in per_file),
        "avg_params": round(sum(m["avg_params"] for m in per_file) / n, 4),
        "file_count": n,
    }, "ok"


FIELDS = ["commit_id", "project", "avg_ccn", "avg_nloc",
          "func_count", "avg_params", "file_count", "buggy"]


def process_project(project: str, workers: int, verify_only: bool) -> Path | None:
    print(f"\n{'='*60}\nProject: {project}")

    url = f"https://github.com/apache/{project}"
    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    repo = CLONE_DIR / project
    if not clone_repo(url, repo):
        return None

    rows = load_recovered_csv(project)
    if not rows:
        return None
    print(f"  {len(rows)} recovered commits")

    if verify_only:
        sample = rows[:50]
        ok = sum(1 for r in sample if get_changed_code_files(repo, r["commit_hash"]))
        print(f"  [{project}] sample: {ok}/{len(sample)} commits have code-file changes")
        return None

    out_path = DATASET_DIR / f"promise_{project}_lizard.csv"
    done_ids = set()
    if out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            done_ids = {r["commit_id"] for r in csv.DictReader(f)}
        if done_ids:
            print(f"  [{project}] resuming — {len(done_ids)} commits already done, skipping them")

    todo = [r for r in rows if r["commit_hash"] not in done_ids]
    if not todo:
        print(f"  [{project}] nothing left to do")
        return out_path

    work = [(r["commit_hash"], project, r["buggy"], repo) for r in todo]
    counts = {"ok": 0, "no_code": 0, "no_funcs": 0, "error": 0}

    write_mode = "a" if done_ids else "w"
    with open(out_path, write_mode, newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDS)
        if write_mode == "w":
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_one_commit, item): item for item in work}
            for fut in tqdm(as_completed(futures), total=len(work), desc=project):
                try:
                    row, status = fut.result()
                except Exception:
                    commit_hash, _, buggy, _ = futures[fut]
                    row = {"commit_id": commit_hash, "project": project,
                           "buggy": buggy, "avg_ccn": "", "avg_nloc": "",
                           "func_count": 0, "avg_params": "", "file_count": 0}
                    status = "error"
                writer.writerow(row)
                fout.flush()
                counts[status] = counts.get(status, 0) + 1

    total = len(todo)
    print(f"  [{project}] {counts['ok']}/{total} analyzed this run  |  "
          f"no_code={counts['no_code']}  no_funcs={counts['no_funcs']}  errors={counts['error']}")
    print(f"  -> {out_path}")
    return out_path


def parse_args():
    verify_only = "--verify" in sys.argv
    workers     = 16  # measured optimum on this machine — see bench notes
    single      = None
    if "--project" in sys.argv:
        idx = sys.argv.index("--project")
        single = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
    if "--workers" in sys.argv:
        idx = sys.argv.index("--workers")
        try:
            workers = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            pass
    return verify_only, workers, single


def main():
    verify_only, workers, single = parse_args()
    projects = [single] if single else PROJECTS

    print(f"Projects: {projects}  |  workers={workers}  |  verify_only={verify_only}")
    out_files = []
    for project in projects:
        out = process_project(project, workers, verify_only)
        if out:
            out_files.append(out)

    if verify_only:
        print(f"\n{'='*60}\nVerify done. Re-run without --verify to process.")
        return

    print(f"\n{'='*60}\nDone. Output files:")
    for f in out_files:
        print(f"  {f}")
    print()
    print("Next steps:")
    print("  1. Upload promise_*_recovered.csv + promise_*_lizard.csv to HuggingFace dataset repo.")
    print("  2. Add a notebook cell (modeled on cell 19/20 ApacheJIT loader) that:")
    print("     - loads promise_{project}_recovered.csv + promise_{project}_lizard.csv")
    print("     - joins them by commit_hash / commit_id")
    print("     - runs classify_batch() on message for sentiment_score")
    print("     - percentile-ranks entrophy/ndev/age/avg_ccn/avg_nloc per project")
    print("     - appends to the training pool alongside DeepJIT + ApacheJIT + geocabral + JIT-Fine")


if __name__ == "__main__":
    main()
