#!/usr/bin/env python3
"""
Build commit→file mapping CSVs for file-level training in gomi_train.ipynb.

For each commit in the training/validation datasets, runs:
    git diff-tree --no-commit-id -r --name-only --diff-filter=AM <hash>

Output files (upload all three to HuggingFace jit/ folder after running):
    dataset/commit_files_deepjit.csv         commit_hash, project, file_path, date
    dataset/commit_files_apache_train.csv    commit_hash, project, file_path, date
    dataset/commit_files_apache_test.csv     commit_hash, project, file_path, date

Usage:
    python scripts/build_file_mapping.py
    python scripts/build_file_mapping.py --workers 64
    python scripts/build_file_mapping.py --split deepjit
    python scripts/build_file_mapping.py --split train
    python scripts/build_file_mapping.py --split test
"""

import csv
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        total = kw.get("total", "?")
        for i, x in enumerate(it):
            if i % 2000 == 0:
                print(f"  {i}/{total}", flush=True)
            yield x

SCRIPT_DIR  = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR.parent / "dataset"
CLONE_BASE  = DATASET_DIR / ".clones"

OUTPUT_FIELDS = ["commit_hash", "project", "file_path", "date"]

CODE_EXTS = {
    ".py", ".java", ".go", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp",
    ".cs", ".js", ".ts", ".rb", ".rs", ".scala", ".kt", ".swift", ".php", ".m",
}

DEEPJIT_PROJECTS = ["qt", "openstack", "go", "jdt", "gerrit", "platform"]

APACHE_OVERRIDES = {
    "apache/hadoop-hdfs":      "apache_hadoop",
    "apache/hadoop-mapreduce": "apache_hadoop",
}

BATCH_TIMEOUT = 120


# ── Git ────────────────────────────────────────────────────────────────────────

def git_changed_files(git_dir: Path, commit_hash: str) -> list:
    try:
        r = subprocess.run(
            ["git", "--git-dir", str(git_dir), "diff-tree",
             "--no-commit-id", "-r", "--name-only", "--diff-filter=AM", commit_hash],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return []
        return [
            f.strip() for f in r.stdout.strip().splitlines()
            if f.strip() and Path(f.strip()).suffix.lower() in CODE_EXTS
        ]
    except Exception:
        return []


def hashes_in_subrepo(subrepo: Path, hashes: list) -> set:
    """Return set of hashes that exist in this bare subrepo."""
    inp = "\n".join(hashes).encode()
    try:
        r = subprocess.run(
            ["git", "--git-dir", str(subrepo),
             "cat-file", "--batch-check=%(objectname) %(objecttype)"],
            input=inp, capture_output=True, timeout=BATCH_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return set()
    found = set()
    for line in r.stdout.decode(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] != "missing":
            found.add(parts[0])
    return found


def build_hash_to_subrepo(proj_dir: Path, hashes: list) -> dict:
    """
    Searches all bare-repo subdirectories of proj_dir for each hash.
    Returns {hash: subrepo_path}.
    Each project dir (e.g. .clones/qt/) contains multiple bare subrepos.
    """
    subrepos = [p for p in proj_dir.iterdir() if p.is_dir()]
    if not subrepos:
        return {}

    hash_to_repo = {}
    with ThreadPoolExecutor(max_workers=len(subrepos)) as ex:
        futures = {ex.submit(hashes_in_subrepo, sr, hashes): sr for sr in subrepos}
        for fut in as_completed(futures):
            sr = futures[fut]
            for h in fut.result():
                if h not in hash_to_repo:
                    hash_to_repo[h] = sr

    return hash_to_repo


def process_commit(args):
    commit_hash, project, date, git_dir = args
    files = git_changed_files(git_dir, commit_hash)
    return commit_hash, project, date, files


# ── DeepJIT ───────────────────────────────────────────────────────────────────

def load_deepjit_work() -> list:
    """Returns [(commit_hash, project, date, subrepo_path), ...]"""
    work = []
    for proj in DEEPJIT_PROJECTS:
        proj_dir = CLONE_BASE / proj
        if not proj_dir.exists():
            print(f"  [skip] {proj}: not found at {proj_dir}")
            continue

        pkl_path = DATASET_DIR / "deepjit" / proj / "deepjit" / f"{proj}_test_raw.pkl"
        if not pkl_path.exists():
            print(f"  [skip] {proj}: PKL not found at {pkl_path}")
            continue

        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        hashes = raw[0]

        # Dates from k_feature CSV
        date_map = {}
        feat_csv = DATASET_DIR / "deepjit" / proj / f"{proj}_k_feature.csv"
        if feat_csv.exists():
            with open(feat_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    h = row.get("_id", "").strip()
                    d = row.get("date", "").strip()
                    if h and d:
                        try:
                            date_map[h] = int(float(d))
                        except ValueError:
                            pass

        # Index hashes across all subrepos
        print(f"  {proj}: indexing {len(hashes)} hashes across subrepos...")
        hash_to_repo = build_hash_to_subrepo(proj_dir, hashes)
        found = len(hash_to_repo)
        print(f"  {proj}: {found}/{len(hashes)} hashes found  |  {len(date_map)} with dates")

        for h in hashes:
            subrepo = hash_to_repo.get(h)
            if subrepo:
                work.append((h, proj, date_map.get(h, ""), subrepo))

    return work


# ── ApacheJIT ─────────────────────────────────────────────────────────────────

def find_apache_clone(project: str) -> Path | None:
    override = APACHE_OVERRIDES.get(project)
    if override:
        p = CLONE_BASE / "apache" / override
        return p if p.exists() else None
    if "/" in project:
        _, name = project.split("/", 1)
        p = CLONE_BASE / "apache" / f"apache_{name}"
        return p if p.exists() else None
    return None


def load_apache_work(csv_path: Path) -> list:
    """Returns [(commit_hash, project, date, git_dir), ...]"""
    if not csv_path.exists():
        print(f"  [skip] not found: {csv_path}")
        return []

    work = []
    missing_clones = set()
    clone_cache = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = row.get("commit_id", "").strip()
            p = row.get("project", "").strip()
            d = row.get("author_date", "").strip()
            if not h or not p:
                continue

            if p not in clone_cache:
                clone_cache[p] = find_apache_clone(p)
                if clone_cache[p] is None:
                    missing_clones.add(p)

            git_dir = clone_cache[p]
            if git_dir is None:
                continue

            try:
                date = int(float(d)) if d else ""
            except ValueError:
                date = ""

            work.append((h, p, date, git_dir))

    if missing_clones:
        print(f"  Missing clones: {', '.join(sorted(missing_clones))}")
    return work


# ── Run + write ────────────────────────────────────────────────────────────────

def run_and_write(work: list, out_path: Path, label: str, workers: int):
    if not work:
        print(f"  No work for {label}, skipping.")
        return

    print(f"Running diff-tree on {len(work)} {label} commits ({workers} workers)...")
    rows = []
    missing = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_commit, item): item for item in work}
        for fut in tqdm(as_completed(futures), total=len(work), desc=label):
            commit_hash, project, date, files = fut.result()
            if not files:
                missing += 1
            for fp in files:
                rows.append({
                    "commit_hash": commit_hash,
                    "project":     project,
                    "file_path":   fp,
                    "date":        date,
                })

    print(f"  No-code commits: {missing}/{len(work)}")
    print(f"  (commit, file) pairs: {len(rows)}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    workers = 64
    split   = None
    if "--workers" in sys.argv:
        idx = sys.argv.index("--workers")
        try:
            workers = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            pass
    if "--split" in sys.argv:
        idx = sys.argv.index("--split")
        if idx + 1 < len(sys.argv):
            split = sys.argv[idx + 1]
    return workers, split


def main():
    workers, split = parse_args()

    run_deepjit = split in (None, "deepjit")
    run_train   = split in (None, "train")
    run_test    = split in (None, "test")

    if run_deepjit:
        print("\n" + "=" * 60)
        print("DeepJIT")
        print("=" * 60)
        work = load_deepjit_work()
        run_and_write(work, DATASET_DIR / "commit_files_deepjit.csv", "deepjit", workers)

    if run_train:
        print("\n" + "=" * 60)
        print("ApacheJIT train")
        print("=" * 60)
        work = load_apache_work(DATASET_DIR / "apachejit_train_with_messages.csv")
        run_and_write(work, DATASET_DIR / "commit_files_apache_train.csv", "apache-train", workers)

    if run_test:
        print("\n" + "=" * 60)
        print("ApacheJIT test")
        print("=" * 60)
        work = load_apache_work(DATASET_DIR / "apachejit_test_with_messages.csv")
        run_and_write(work, DATASET_DIR / "commit_files_apache_test.csv", "apache-test", workers)

    print("\n" + "=" * 60)
    print("Done. Upload to HuggingFace:")
    for name in ["commit_files_deepjit", "commit_files_apache_train", "commit_files_apache_test"]:
        p = DATASET_DIR / f"{name}.csv"
        if p.exists():
            print(f"  huggingface-cli upload GitRatBCSAD/gomi-datasets {p} jit/{name}.csv --repo-type dataset")


if __name__ == "__main__":
    main()
