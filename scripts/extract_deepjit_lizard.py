#!/usr/bin/env python3
"""
Extract Lizard structural complexity metrics for every DeepJIT commit.

For each commit:
  1. Batch-searches all repos for the project (one cat-file --batch-check per repo)
  2. Lists changed source files (git diff-tree --diff-filter=AM)
  3. Reads file content at that commit (git show <hash>:<file>)
  4. Runs Lizard in-memory → AvgCCN, AvgNLOC, FuncCount, AvgParams
  5. Averages metrics across all changed code files in the commit

Each DeepJIT project spans MULTIPLE repos (from ISSTA21-JIT-DP/Data_Extraction/git_base/git_datasets/).
"platform" = Eclipse Platform (NOT Android).

Output: dataset/deepjit_{project}_lizard.csv
Columns: commit_id, message, avg_ccn, avg_nloc, func_count, avg_params, file_count, buggy

Usage:
    python scripts/extract_deepjit_lizard.py              # full run, all projects
    python scripts/extract_deepjit_lizard.py --verify     # check repos + hash coverage
    python scripts/extract_deepjit_lizard.py --project qt # single project
    python scripts/extract_deepjit_lizard.py --workers 32 # override thread count (default: CPU*2)

Repos cloned bare with --filter=blob:none (small initial clone, blobs fetched lazily).
"""

import csv
import os
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
CLONE_DIR   = DATASET_DIR / ".clones"

# ── Repo URLs (from ISSTA21-JIT-DP/Data_Extraction/git_base/git_datasets/) ────
REPO_MAP = {
    "gerrit": [
        "https://gerrit-review.googlesource.com/gerrit",
        "https://gerrit-review.googlesource.com/gerrit-ci-scripts",
        "https://gerrit-review.googlesource.com/git-repo",
        "https://gerrit-review.googlesource.com/gitiles",
        "https://gerrit-review.googlesource.com/homepage",
        "https://gerrit-review.googlesource.com/plugins/delete-project",
        "https://gerrit-review.googlesource.com/plugins/events-log",
        "https://gerrit-review.googlesource.com/plugins/high-availability",
        "https://gerrit-review.googlesource.com/plugins/lfs",
        "https://gerrit-review.googlesource.com/plugins/replication",
        "https://gerrit-review.googlesource.com/plugins/reviewers",
        "https://gerrit-review.googlesource.com/zoekt",
    ],
    "go": [
        "https://go-review.googlesource.com/build",
        "https://go-review.googlesource.com/crypto",
        "https://go-review.googlesource.com/exp",
        "https://go-review.googlesource.com/go",
        "https://go-review.googlesource.com/gofrontend",
        "https://go-review.googlesource.com/mobile",
        "https://go-review.googlesource.com/net",
        "https://go-review.googlesource.com/pkgsite",
        "https://go-review.googlesource.com/protobuf",
        "https://go-review.googlesource.com/sys",
        "https://go-review.googlesource.com/text",
        "https://go-review.googlesource.com/tools",
        "https://go-review.googlesource.com/vscode-go",
    ],
    "jdt": [
        # Eclipse migrated from git.eclipse.org/r/ to GitHub (eclipse-jdt org)
        "https://github.com/eclipse-jdt/eclipse.jdt.core",
        "https://github.com/eclipse-jdt/eclipse.jdt.debug",
        "https://github.com/eclipse-jdt/eclipse.jdt.ui",
    ],
    "openstack": [
        # review.opendev.org is Gerrit web UI — actual git clone URL is opendev.org or GitHub mirror
        "https://github.com/openstack/neutron",
        "https://github.com/openstack/nova",
        "https://github.com/openstack/swift",
        "https://github.com/openstack/cinder",
        "https://github.com/openstack/glance",
    ],
    "platform": [
        # Eclipse migrated from git.eclipse.org/r/ to GitHub (eclipse-platform org)
        "https://github.com/eclipse-platform/eclipse.platform",
        "https://github.com/eclipse-platform/eclipse.platform.common",
        "https://github.com/eclipse-platform/eclipse.platform.debug",
        "https://github.com/eclipse-platform/eclipse.platform.images",
        "https://github.com/eclipse-platform/eclipse.platform.releng",
        "https://github.com/eclipse-platform/eclipse.platform.releng.aggregator",
        "https://github.com/eclipse-platform/eclipse.platform.resources",
        "https://github.com/eclipse-platform/eclipse.platform.runtime",
        "https://github.com/eclipse-platform/eclipse.platform.swt",
        "https://github.com/eclipse-platform/eclipse.platform.swt.binaries",
        "https://github.com/eclipse-platform/eclipse.platform.team",
        "https://github.com/eclipse-platform/eclipse.platform.text",
        "https://github.com/eclipse-platform/eclipse.platform.ua",
        "https://github.com/eclipse-platform/eclipse.platform.ui",
        "https://github.com/eclipse-platform/eclipse.platform.ui.tools",
    ],
    "qt": [
        "https://codereview.qt-project.org/qt/qtbase",
        "https://codereview.qt-project.org/qt/qtconnectivity",
        "https://codereview.qt-project.org/qt/qtdeclarative",
        "https://codereview.qt-project.org/qt/qtwebkit",
        "https://codereview.qt-project.org/qt/qtserialport",
        "https://codereview.qt-project.org/qt/qtwebchannel",
        "https://codereview.qt-project.org/qt/qtsensors",
        "https://codereview.qt-project.org/qt/qtquickcontrols",
        "https://codereview.qt-project.org/qt/qtmultimedia",
        "https://codereview.qt-project.org/qt/qttools",
        "https://codereview.qt-project.org/qt/qtlocation",
        "https://codereview.qt-project.org/qt/qtandroidextras",
        "https://codereview.qt-project.org/qt/qtimageformats",
        "https://codereview.qt-project.org/qt/qtwinextras",
        "https://codereview.qt-project.org/qt/qtrepotools",
        "https://codereview.qt-project.org/qt/qtquick1",
        "https://codereview.qt-project.org/qt/qtscript",
        "https://codereview.qt-project.org/qt/qtxmlpatterns",
        "https://codereview.qt-project.org/qt/qtqa",
        "https://codereview.qt-project.org/qt/qtsvg",
        "https://codereview.qt-project.org/qt/qtdoc",
        "https://codereview.qt-project.org/qt/qtwebsockets",
        "https://codereview.qt-project.org/qt/qtactiveqt",
    ],
}

DEEPJIT_PKLS = {
    "qt":        "qt_test_raw.pkl",
    "openstack": "openstack_test_raw.pkl",
    "go":        "go_test_raw.pkl",
    "jdt":       "jdt_test_raw.pkl",
    "gerrit":    "gerrit_test_raw.pkl",
    "platform":  "platform_test_raw.pkl",
}

CODE_EXTENSIONS = {
    ".py", ".java", ".go", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp",
    ".cs", ".js", ".ts", ".rb", ".rs", ".scala", ".kt", ".swift", ".php", ".m",
}


# ── Repo name from URL ─────────────────────────────────────────────────────────

def repo_name(url: str) -> str:
    parts = url.rstrip("/").split("/")
    return "_".join(parts[-2:]).replace(".", "_") if len(parts) >= 2 else parts[-1].replace(".", "_")


# ── Git helpers ────────────────────────────────────────────────────────────────

def clone_repo(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    # Full bare clone — blobs stored locally so git show reads from disk, not network.
    r = subprocess.run(
        ["git", "clone", "--bare", url, str(dest)],
        capture_output=True,
    )
    if r.returncode != 0:
        print(f"    FAILED {url}: {r.stderr.decode()[:200]}")
        return False
    return True


def clone_all_repos(project: str, clone_workers: int = 8) -> list:
    """Clone all repos for a project in parallel. Returns list of (url, repo_path)."""
    proj_dir = CLONE_DIR / project
    proj_dir.mkdir(parents=True, exist_ok=True)

    if (proj_dir / "HEAD").exists():
        print(f"  [{project}] WARNING: old single-repo bare clone at {proj_dir}")
        print(f"             Delete it first: rm -rf {proj_dir}")
        return []

    urls = REPO_MAP[project]
    print(f"  [{project}] cloning {len(urls)} repos (up to {clone_workers} parallel)...")

    def _clone(url):
        dest = proj_dir / repo_name(url)
        if clone_repo(url, dest):
            return (url, dest)
        return None

    with ThreadPoolExecutor(max_workers=min(len(urls), clone_workers)) as ex:
        raw = list(ex.map(_clone, urls))

    repos = [r for r in raw if r is not None]
    print(f"  [{project}] {len(repos)}/{len(urls)} repos ready")
    return repos


# ── Batch hash indexing (one subprocess per repo instead of one per hash) ──────

BATCH_TIMEOUT = 120  # seconds per repo batch-check


def hashes_in_repo(repo: Path, hashes: list) -> tuple:
    """
    Check which hashes exist in repo using a single batch subprocess call.
    Returns (repo_path, set_of_found_hashes).
    """
    inp = "\n".join(hashes).encode()
    try:
        r = subprocess.run(
            ["git", "--git-dir", str(repo),
             "cat-file", "--batch-check=%(objectname) %(objecttype)"],
            input=inp, capture_output=True, timeout=BATCH_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT indexing {repo.name} (>{BATCH_TIMEOUT}s) — skipping")
        return repo, set()
    found = set()
    for line in r.stdout.decode(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] != "missing":
            found.add(parts[0])
    return repo, found


def build_hash_index(project: str, repos: list, hashes: list) -> dict:
    """
    Map each commit hash → repo_path using parallel batch cat-file calls.
    Complexity: O(n_repos) subprocess calls, not O(n_hashes × n_repos).
    """
    repo_paths = [rp for _, rp in repos]
    print(f"  [{project}] batch-indexing {len(hashes)} hashes across {len(repo_paths)} repos...")

    index = {}
    done = 0
    with ThreadPoolExecutor(max_workers=len(repo_paths)) as ex:
        futures = {ex.submit(hashes_in_repo, rp, hashes): rp for rp in repo_paths}
        for fut in as_completed(futures):
            rp, found = fut.result()
            done += 1
            for h in found:
                if h not in index:
                    index[h] = rp
            print(f"  [{project}] indexed {done}/{len(repo_paths)} repos  "
                  f"(found so far: {len(index)})", end="\r", flush=True)

    print()  # newline after \r
    found_n   = len(index)
    missing_n = len(hashes) - found_n
    print(f"  [{project}] found {found_n}/{len(hashes)}  |  {missing_n} not in any repo")
    return index


def verify_coverage(project: str, repos: list, hashes: list, n: int = 100):
    sample = hashes[:n]
    index, _ = build_hash_index.__wrapped__(project, repos, sample) if hasattr(build_hash_index, "__wrapped__") else ({}, None)
    # Just reuse build_hash_index on the sample
    idx = build_hash_index(project, repos, sample)
    found = len(idx)
    print(f"  [{project}] sample coverage: {found}/{n} ({found/n*100:.0f}%)")
    if found < n:
        missing = [h for h in sample if h not in idx]
        print(f"             first missing: {missing[0]}")


# ── Per-commit processing ──────────────────────────────────────────────────────

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
    """
    Fetch multiple files at a commit in ONE subprocess call via git cat-file --batch.
    Returns {filepath: content_str}. Missing files are omitted.
    """
    if not filepaths:
        return {}
    requests = "".join(f"{commit_hash}:{fp}\n" for fp in filepaths)
    r = subprocess.run(
        ["git", "--git-dir", str(repo), "cat-file", "--batch"],
        input=requests.encode(),
        capture_output=True,
    )
    if r.returncode != 0:
        return {}

    results = {}
    data = r.stdout
    pos = 0
    for fp in filepaths:
        if pos >= len(data):
            break
        nl = data.find(b'\n', pos)
        if nl == -1:
            break
        header = data[pos:nl].decode(errors="replace")
        pos = nl + 1
        parts = header.split()
        # missing object: "<request> missing"
        if len(parts) < 3 or parts[1] == "missing":
            continue
        size = int(parts[2])
        content_bytes = data[pos:pos + size]
        results[fp] = content_bytes.decode("utf-8", errors="replace")
        pos += size + 1  # skip content + trailing newline

    return results


def run_lizard(filepath: str, source: str):
    info = lizard.analyze_file.analyze_source_code(filepath, source)
    funcs = info.function_list
    if not funcs:
        return None
    return {
        "avg_ccn":    sum(fn.cyclomatic_complexity for fn in funcs) / len(funcs),
        "avg_nloc":   sum(fn.nloc                  for fn in funcs) / len(funcs),
        "avg_params": sum(fn.parameter_count       for fn in funcs) / len(funcs),
        "func_count": len(funcs),
    }


def process_one_commit(args):
    """Worker function for ThreadPoolExecutor. Returns (index, row_dict, status)."""
    idx, h, label, msg, hash_index = args
    base  = {"commit_id": h, "message": (msg or "").strip(), "buggy": int(label)}
    empty = {**base, "avg_ccn": "", "avg_nloc": "", "func_count": 0, "avg_params": "", "file_count": 0}

    repo = hash_index.get(h)
    if repo is None:
        return idx, empty, "no_repo"

    code_files = get_changed_code_files(repo, h)
    if not code_files:
        return idx, empty, "no_code"

    # One subprocess call for all files in this commit
    contents = get_files_content_batch(repo, h, code_files)

    per_file = []
    for fp, content in contents.items():
        m = run_lizard(fp, content)
        if m:
            per_file.append(m)

    if not per_file:
        return idx, {**empty, "file_count": len(code_files)}, "no_funcs"

    n = len(per_file)
    row = {
        **base,
        "avg_ccn":    round(sum(m["avg_ccn"]    for m in per_file) / n, 4),
        "avg_nloc":   round(sum(m["avg_nloc"]   for m in per_file) / n, 4),
        "func_count": sum(m["func_count"] for m in per_file),
        "avg_params": round(sum(m["avg_params"] for m in per_file) / n, 4),
        "file_count": n,
    }
    return idx, row, "ok"


def process_project(project: str, hash_index: dict,
                    hashes: list, labels: list, messages: list,
                    workers: int) -> Path:
    out_path = DATASET_DIR / f"deepjit_{project}_lizard.csv"
    fields   = ["commit_id", "message", "avg_ccn", "avg_nloc",
                "func_count", "avg_params", "file_count", "buggy"]

    total   = len(hashes)
    results = [None] * total
    counts  = {"ok": 0, "no_repo": 0, "no_code": 0, "no_funcs": 0, "error": 0}

    work_items = [
        (i, hashes[i], labels[i], messages[i], hash_index)
        for i in range(total)
    ]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one_commit, item): item[0] for item in work_items}
        pbar = tqdm(as_completed(futures), total=total, desc=project)
        for fut in pbar:
            try:
                idx, row, status = fut.result()
            except Exception as e:
                idx = futures[fut]
                h, label, msg = hashes[idx], labels[idx], messages[idx]
                row = {"commit_id": h, "message": (msg or "").strip(), "buggy": int(label),
                       "avg_ccn": "", "avg_nloc": "", "func_count": 0, "avg_params": "", "file_count": 0}
                status = "error"
            results[idx] = row
            counts[status] = counts.get(status, 0) + 1

    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"  [{project}] {counts['ok']}/{total} analyzed  |  "
          f"no_repo={counts['no_repo']}  no_code={counts['no_code']}  "
          f"no_funcs={counts['no_funcs']}  errors={counts['error']}")
    print(f"  → {out_path}")
    return out_path


# ── Pkl loader ─────────────────────────────────────────────────────────────────

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


def load_pkl(project: str):
    for candidate in [
        DATASET_DIR / DEEPJIT_PKLS[project],
        DATASET_DIR / "jit" / DEEPJIT_PKLS[project],
    ]:
        if candidate.exists():
            print(f"  [{project}] loading from local: {candidate.name}")
            with open(candidate, "rb") as f:
                raw = pickle.load(f)
            return raw[0], raw[1], raw[2]

    token = get_hf_token()
    if not token:
        print(f"  [{project}] No HF token. Fix: export HF_TOKEN=... OR huggingface-cli login")
        return None, None, None

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            "GitRatBCSAD/gomi-datasets",
            f"jit/{DEEPJIT_PKLS[project]}",
            repo_type="dataset", token=token,
        )
        with open(path, "rb") as f:
            raw = pickle.load(f)
        return raw[0], raw[1], raw[2]
    except Exception as e:
        print(f"  [{project}] pkl load failed: {e}")
        return None, None, None


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    verify_only    = "--verify"  in sys.argv
    single_project = None
    workers        = 64  # I/O-bound: threads spend most time waiting on subprocess, not CPU

    if "--project" in sys.argv:
        idx = sys.argv.index("--project")
        single_project = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None

    if "--workers" in sys.argv:
        idx = sys.argv.index("--workers")
        try:
            workers = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            pass

    return verify_only, single_project, workers


def main():
    verify_only, single_project, workers = parse_args()

    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    projects  = [single_project] if single_project else list(DEEPJIT_PKLS.keys())
    out_files = []

    print(f"Workers: {workers}  |  Projects: {projects}")

    for project in projects:
        print(f"\n{'='*60}")
        print(f"Project: {project}")

        hashes, labels, messages = load_pkl(project)
        if hashes is None:
            print(f"  Skipping — pkl not loadable.")
            continue

        print(f"  Commits: {len(hashes)}  Buggy rate: {sum(labels)/len(labels):.2%}")

        repos = clone_all_repos(project)
        if not repos:
            print(f"  Skipping — no repos available.")
            continue

        if verify_only:
            verify_coverage(project, repos, hashes, n=100)
            continue

        hash_index = build_hash_index(project, repos, hashes)
        out = process_project(project, hash_index, hashes, labels, messages, workers)
        out_files.append(out)

    if verify_only:
        print(f"\n{'='*60}")
        print("Verify done. Re-run without --verify to process.")
        return

    print(f"\n{'='*60}")
    print("Done.")
    for f in out_files:
        print(f"  {f}")
    print()
    print("Next steps:")
    print("  1. Upload deepjit_*_lizard.csv to HuggingFace dataset repo.")
    print("  2. Update notebook FEATURE_NAMES:")
    print("     ['sentiment_score', 'avg_ccn', 'avg_nloc', 'func_count', 'avg_params']")
    print("  3. Apply percentile_rank normalization per project before training.")


if __name__ == "__main__":
    main()
