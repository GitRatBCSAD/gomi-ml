#!/usr/bin/env python3
"""
Fetch commit messages for apachejit_test_small.csv and write
apachejit_test_with_messages.csv — matching the column layout of
apachejit_train_with_messages.csv.

For each commit the subject line is retrieved via:
    git log -1 --format=%s <hash>

Output columns (same as train):
    commit_id, project, buggy, fix, year, author_date, la, ld, nf, nd, ns,
    ent, ndev, age, nuc, aexp, arexp, asexp, message

Usage:
    python scripts/fetch_apache_test_messages.py
    python scripts/fetch_apache_test_messages.py --workers 64
    python scripts/fetch_apache_test_messages.py --local   # read/write datasets/ only, no HF
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
            if i % 500 == 0:
                print(f"  {desc}: {i}/{total}")
            yield x

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR.parent / "datasets"
CLONE_DIR   = DATASET_DIR / ".clones"

HF_DATASET_REPO = "GitRatBCSAD/gomi-datasets"
HF_INPUT_PATH   = "jit/apachejit_test_small.csv"

INPUT_CSV  = DATASET_DIR / "apachejit_test_small.csv"
OUTPUT_CSV = DATASET_DIR / "apachejit_test_with_messages.csv"

# Same as extract_apache_lizard.py
REPO_OVERRIDES = {
    "apache/hadoop-hdfs":      "apache/hadoop",
    "apache/hadoop-mapreduce": "apache/hadoop",
}

# Column order mirrors apachejit_train_with_messages.csv
OUTPUT_FIELDS = [
    "commit_id", "project", "buggy", "fix", "year", "author_date",
    "la", "ld", "nf", "nd", "ns", "ent", "ndev", "age",
    "nuc", "aexp", "arexp", "asexp", "message",
]


# ── HF token ──────────────────────────────────────────────────────────────────

def get_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = SCRIPT_DIR.parent / ".env"
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


def build_hash_index(rows: list, repo_map: dict) -> dict:
    """Build hash→repo map directly from the project column — no git batch-check needed."""
    index = {}
    missing = 0
    for r in rows:
        proj = r["project"].strip()
        repo = repo_map.get(proj)
        if repo:
            index[r["commit_id"]] = repo
        else:
            missing += 1
    print(f"  mapped {len(index)}/{len(rows)}  |  {missing} with no repo")
    return index


def get_commit_subject(repo: Path, commit_hash: str) -> str:
    """Returns the subject line (first line) of the commit message."""
    r = subprocess.run(
        ["git", "--git-dir", str(repo), "log", "-1", "--format=%s", commit_hash],
        capture_output=True, text=True, errors="replace",
    )
    if r.returncode != 0:
        return ""
    return r.stdout.strip()


# ── Per-commit worker ──────────────────────────────────────────────────────────

def process_one_commit(args):
    idx, row, hash_index = args
    repo = hash_index.get(row["commit_id"])
    msg  = get_commit_subject(repo, row["commit_id"]) if repo else ""
    return idx, {**row, "message": msg}


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    workers   = 64
    local     = "--local" in sys.argv
    if "--workers" in sys.argv:
        idx = sys.argv.index("--workers")
        try:
            workers = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            pass
    return workers, local


def main():
    workers, local = parse_args()
    CLONE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load input CSV ────────────────────────────────────────────────────────
    if INPUT_CSV.exists():
        print(f"Reading input from {INPUT_CSV}")
        with open(INPUT_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    elif not local:
        token = get_hf_token()
        if not token:
            print("ERROR: No HF token. Fix: export HF_TOKEN=... OR huggingface-cli login")
            sys.exit(1)
        from huggingface_hub import hf_hub_download
        print(f"Downloading {HF_INPUT_PATH} from HF...")
        path = hf_hub_download(HF_DATASET_REPO, HF_INPUT_PATH,
                               repo_type="dataset", token=token)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        print(f"ERROR: {INPUT_CSV} not found and --local specified.")
        sys.exit(1)

    print(f"Loaded {len(rows)} commits from test split.")

    # ── Clone repos ───────────────────────────────────────────────────────────
    projects = {r["project"].strip() for r in rows}
    print(f"\nCloning {len(projects)} repos...")
    repo_map = {}
    for proj in sorted(projects):
        repo = clone_repo(proj)
        if repo:
            repo_map[proj] = repo
            canonical = REPO_OVERRIDES.get(proj, proj)
            repo_map[canonical] = repo
    print(f"  {len(set(repo_map.values()))} distinct repos ready")

    # ── Build hash index ──────────────────────────────────────────────────────
    print(f"\nBuilding hash index for {len(rows)} commits...")
    hash_index = build_hash_index(rows, repo_map)

    # ── Fetch messages in parallel ────────────────────────────────────────────
    print(f"\nFetching commit messages ({workers} workers)...")
    work    = [(i, r, hash_index) for i, r in enumerate(rows)]
    results = [None] * len(rows)
    missing = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one_commit, item): item[0] for item in work}
        for fut in tqdm(as_completed(futures), total=len(rows), desc="messages"):
            try:
                idx, out_row = fut.result()
            except Exception as e:
                idx = futures[fut]
                out_row = {**rows[idx], "message": ""}
            results[idx] = out_row
            if not out_row["message"]:
                missing += 1

    print(f"  Messages found: {len(rows) - missing}/{len(rows)}  |  empty: {missing}")

    # ── Write output ──────────────────────────────────────────────────────────
    # Use OUTPUT_FIELDS to guarantee column order matches train CSV.
    # Any extra columns in the source are dropped; missing ones get empty string.
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in OUTPUT_FIELDS})

    print(f"\nWrote {len(results)} rows → {OUTPUT_CSV}")
    print("\nNext steps:")
    print("  1. Upload datasets/apachejit_test_with_messages.csv to HF:")
    print(f"     huggingface-cli upload {HF_DATASET_REPO} {OUTPUT_CSV} jit/apachejit_test_with_messages.csv --repo-type dataset")
    print("  2. Re-run the notebook — cell 15 will auto-detect the new file.")


if __name__ == "__main__":
    main()
