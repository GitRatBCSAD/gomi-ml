#!/usr/bin/env python3
"""
Build commit-hash -> file_path mapping CSVs for geocabral/jitfine/promise,
matching the existing deepjit_files.csv / apachejit_{train,test}_files.csv
convention so gomi_train.ipynb's file-level aggregation cell (load_mapping()/
build_file_level()) can group these three sources by file too — currently it
can't, since no mapping exists for them and every commit rides through as a
raw commit-level row instead of an aggregated 6-month file-window.

Reuses --cc (not plain diff-tree), same as extract_geocabral_lizard.py —
geocabral has real, non-trivial merge commits that plain diff-tree silently
returns nothing for. build_file_mapping.py (the original DeepJIT/ApacheJIT
script) doesn't need --cc because those two sources are confirmed merge-free.

For each commit already in dataset/{source}_{project}_commits.csv:
    git diff-tree --cc --no-commit-id -r --name-only --diff-filter=AM <hash>

Output: dataset/{source}_files.csv (one combined file per source, all projects)
Columns: commit_hash, project, file_path, date

Usage:
    python scripts/build_file_mapping_new_sources.py                     # all 3 sources
    python scripts/build_file_mapping_new_sources.py --source geocabral  # one source
    python scripts/build_file_mapping_new_sources.py --source geocabral --project npm
    python scripts/build_file_mapping_new_sources.py --workers 16
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
        desc = kw.get("desc", "")
        for i, x in enumerate(it):
            if i % 500 == 0:
                print(f"  {desc}: {i}/{total}")
            yield x

DATASET_DIR = Path(__file__).parent.parent / "dataset"
CLONE_DIR = DATASET_DIR / ".clones"

CODE_EXTENSIONS = {
    ".py", ".java", ".go", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp",
    ".cs", ".js", ".ts", ".rb", ".rs", ".scala", ".kt", ".swift", ".php", ".m",
}

GIT_TIMEOUT = 60
SOURCES = ["geocabral", "jitfine", "promise"]
FIELDS = ["commit_hash", "project", "file_path", "date"]


def discover_projects(source: str) -> list:
    """Find every {source}_{project}_commits.csv already on disk."""
    prefix = f"{source}_"
    suffix = "_commits.csv"
    projects = []
    for f in DATASET_DIR.glob(f"{source}_*_commits.csv"):
        name = f.name[len(prefix):-len(suffix)]
        projects.append(name)
    return sorted(projects)


def get_changed_files(repo: Path, commit_hash: str) -> list:
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
    return [f.strip() for f in r.stdout.strip().splitlines()
            if f.strip() and Path(f.strip()).suffix.lower() in CODE_EXTENSIONS]


def process_one(args):
    commit_hash, project, date, repo = args
    files = get_changed_files(repo, commit_hash)
    return commit_hash, project, date, files


def load_commits(source: str, project: str) -> list:
    """Returns [(commit_hash, project, date), ...] from the recovered commits CSV."""
    path = DATASET_DIR / f"{source}_{project}_commits.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = row.get("commit_hash", "").strip()
            d = row.get("author_date_unix_timestamp", "").strip()
            if h:
                rows.append((h, project, d))
    return rows


def build_source(source: str, workers: int, single_project: str = None):
    print(f"\n{'='*60}\nSource: {source}")
    projects = [single_project] if single_project else discover_projects(source)
    if not projects:
        print(f"  no {source}_*_commits.csv files found — skip")
        return

    out_path = DATASET_DIR / f"{source}_files.csv"

    # Resume support, same convention as the lizard extraction scripts:
    # track which (commit_hash, project) pairs already have rows written.
    done_keys = set()
    if out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_keys.add((row["commit_hash"], row["project"]))
        if done_keys:
            print(f"  resuming — {len(done_keys)} (commit,project) pairs already mapped")

    write_mode = "a" if done_keys else "w"
    with open(out_path, write_mode, newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDS)
        if write_mode == "w":
            writer.writeheader()

        for project in projects:
            repo = CLONE_DIR / project
            if not repo.exists():
                print(f"  [{project}] no clone at {repo} — skip")
                continue

            commits = load_commits(source, project)
            todo = [(h, p, d) for h, p, d in commits if (h, p) not in done_keys]
            if not todo:
                print(f"  [{project}] nothing left to do ({len(commits)} total)")
                continue

            print(f"  [{project}] {len(todo)}/{len(commits)} commits to map")
            work = [(h, p, d, repo) for h, p, d in todo]
            no_code = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(process_one, item): item for item in work}
                for fut in tqdm(as_completed(futures), total=len(work), desc=f"{source}/{project}"):
                    h, p, d, files = fut.result()
                    if not files:
                        no_code += 1
                    for fp in files:
                        writer.writerow({"commit_hash": h, "project": p, "file_path": fp, "date": d})
                    fout.flush()
            print(f"  [{project}] no_code={no_code}/{len(todo)}")

    print(f"  -> {out_path}")


def parse_args():
    workers = 16
    source = None
    project = None
    if "--workers" in sys.argv:
        i = sys.argv.index("--workers")
        try:
            workers = int(sys.argv[i + 1])
        except (IndexError, ValueError):
            pass
    if "--source" in sys.argv:
        i = sys.argv.index("--source")
        if i + 1 < len(sys.argv):
            source = sys.argv[i + 1]
    if "--project" in sys.argv:
        i = sys.argv.index("--project")
        if i + 1 < len(sys.argv):
            project = sys.argv[i + 1]
    if project and not source:
        print("ERROR: --project requires --source (project names aren't unique across sources)")
        sys.exit(1)
    return workers, source, project


def main():
    workers, source, project = parse_args()
    sources = [source] if source else SOURCES
    for s in sources:
        build_source(s, workers, single_project=project)

    print(f"\n{'='*60}\nDone.")
    print("Next: upload {source}_files.csv to HuggingFace jit/{source}/ alongside the commits/lizard CSVs.")
    print("Then update gomi_train.ipynb's load_mapping() calls to also load these 3 new mapping files.")


if __name__ == "__main__":
    main()
