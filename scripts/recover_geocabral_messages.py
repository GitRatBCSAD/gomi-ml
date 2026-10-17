#!/usr/bin/env python3
"""
Recover commit hash + message for the geocabral-spdisc-icse19 JIT-SDP dataset.

geocabral's *.arff files (Cabral et al., "Class Imbalance Evolution and
Verification Latency in Just-in-Time Software Defect Prediction," ICSE'19)
contain the classic Kamei et al. 2013 process metrics + a `contains_bug`
label, per commit — but NO commit hash or message. Only
`author_date_unix_timestamp` identifies the commit.

Recovery strategy: bare-clone the project's repo, walk its default-branch
history, and match ARFF rows to real commits by EXACT author-date timestamp
(no fuzzy tolerance — a false match would poison labels, better to drop an
unmatched row than mismatch one). Matched rows get a real commit_hash + real
commit message + the original 14 numeric features + the buggy label, ready
to feed into the same feature schema as DeepJIT/ApacheJIT.

Usage:
    python scripts/recover_geocabral_messages.py                 # pilot: npm + wagtail
    python scripts/recover_geocabral_messages.py --project npm   # single project
    python scripts/recover_geocabral_messages.py --all           # all 14 (needs URLs filled in)
    python scripts/recover_geocabral_messages.py --verify        # match-rate only, no CSV written

Output: dataset/geocabral_{project}_recovered.csv
Columns: commit_hash, project, message, buggy, author_date_unix_timestamp,
         fix, ns, nd, nf, entrophy, la, ld, lt, ndev, age, nuc, exp, rexp, sexp
"""

import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.parent.parent   # gomi/
ARFF_DIR    = REPO_ROOT / "references" / "geocabral-spdisc-icse19-0a7955c" / "datasets"
DATASET_DIR = Path(__file__).parent.parent / "dataset"
CLONE_DIR   = DATASET_DIR / ".clones"

# All 14 cross-validated against IRJIT (arXiv:2210.02435, Fig. 1) project
# names + a real clone/commit-count sanity check per repo (see chat log).
# "fabric" ARFF file == "Fabric8" (fabric8io org) in the paper's figure —
# NOT the Python "fabric" deploy tool, that guess was wrong, fixed below.
# Repo commit counts are all >= ARFF row counts for actively-maintained
# projects (matplotlib, zulip, nova, tomcat, sentry, camel) because the ARFF
# mining cutoff is ~2017-2019 and these repos kept growing after — expected,
# not a mismatch (max author_date in each ARFF confirms the ~2018 cutoff).
REPO_MAP = {
    "npm":                 "https://github.com/npm/npm",                             # verified: 9258 rows / 8342 commits
    "wagtail":             "https://github.com/wagtail/wagtail",                     # verified: 9095 rows / 20187 commits
    "brackets":            "https://github.com/adobe/brackets",                      # verified: 21348 rows / 17847 commits
    "broadleaf":           "https://github.com/BroadleafCommerce/BroadleafCommerce",  # verified: 17430 rows / 18776 commits
    "camel":               "https://github.com/apache/camel",                        # verified: 36753 rows / 82174 commits
    "fabric":              "https://github.com/fabric8io/fabric8",                   # verified: 15592 rows / 13497 commits (paper calls it "Fabric8")
    "jgroups":             "https://github.com/belaban/JGroups",                     # verified: 21469 rows / 21049 commits
    "matplotlib":          "https://github.com/matplotlib/matplotlib",               # verified: 22096 rows / 54900 commits
    "neutron":             "https://github.com/openstack/neutron",                   # verified: 24058 rows / 30545 commits
    "nova":                "https://github.com/openstack/nova",                      # verified: 61365 rows / 62353 commits
    "sentry":              "https://github.com/getsentry/sentry",                    # verified: 26604 rows / 106869 commits
    "spring-integration":  "https://github.com/spring-projects/spring-integration",  # verified: 11000 rows / 13051 commits
    "tomcat":              "https://github.com/apache/tomcat",                       # verified: 24043 rows / 28831 commits
    "zulip":               "https://github.com/zulip/zulip",                         # verified: 22965 rows / 71220 commits
}

PILOT_PROJECTS = ["npm", "wagtail"]

ARFF_COLUMNS = [
    "fix", "ns", "nd", "nf", "entrophy", "la", "ld", "lt",
    "ndev", "age", "nuc", "exp", "rexp", "sexp",
    "contains_bug", "author_date_unix_timestamp", "commit_type",
]


def parse_arff(project: str) -> list:
    path = ARFF_DIR / f"{project}.arff"
    rows = []
    in_data = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if s.lower() == "@data":
                in_data = True
                continue
            if s.startswith("@"):
                continue
            if not in_data:
                continue
            parts = s.split(",")
            if len(parts) != len(ARFF_COLUMNS):
                continue
            row = dict(zip(ARFF_COLUMNS, parts))
            rows.append(row)
    return rows


def clone_repo(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    # Full clone, no --filter=blob:none — matches extract_deepjit_lizard.py /
    # extract_apache_lizard.py's clone_repo. This script itself only needs
    # git-log metadata, but the SAME clone gets reused by
    # extract_geocabral_lizard.py to read file content for Lizard — a
    # blobless clone there means lazy per-blob network fetches mixed into
    # what should be pure-CPU timing. Pay the full clone cost once, upfront.
    r = subprocess.run(
        ["git", "clone", "--bare", url, str(dest)],
        capture_output=True,
    )
    if r.returncode != 0:
        print(f"    FAILED clone {url}: {r.stderr.decode()[:300]}")
        return False
    return True


def load_git_log(repo: Path) -> dict:
    """Returns {unix_timestamp: [(hash, subject), ...]} for default-branch history."""
    r = subprocess.run(
        ["git", "--git-dir", str(repo), "log",
         "--format=%H%x1f%at%x1f%s", "HEAD"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        print(f"    git log failed: {r.stderr[:300]}")
        return {}

    by_ts = {}
    for line in r.stdout.splitlines():
        parts = line.split("\x1f", 2)
        if len(parts) != 3:
            continue
        h, ts, subject = parts
        by_ts.setdefault(int(ts), []).append((h, subject))
    return by_ts


def recover_project(project: str, verify_only: bool) -> None:
    url = REPO_MAP.get(project)
    if not url:
        print(f"  [{project}] no repo URL configured — skip")
        return

    print(f"\n{'='*60}\nProject: {project}")
    rows = parse_arff(project)
    print(f"  ARFF rows: {len(rows)}")
    if not rows:
        print(f"  Skipping — no ARFF data.")
        return

    proj_dir = CLONE_DIR / project
    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Cloning {url} ...")
    if not clone_repo(url, proj_dir):
        return

    print(f"  Indexing commit history by timestamp ...")
    by_ts = load_git_log(proj_dir)
    total_commits = sum(len(v) for v in by_ts.values())
    print(f"  Repo commits (HEAD history): {total_commits}")

    matched, ambiguous, unmatched = 0, 0, 0
    used_hashes = set()
    out_rows = []

    for row in rows:
        ts = int(row["author_date_unix_timestamp"])
        candidates = [c for c in by_ts.get(ts, []) if c[0] not in used_hashes]
        if not candidates:
            unmatched += 1
            continue
        if len(candidates) > 1:
            ambiguous += 1  # still usable, just note the collision
        h, subject = candidates[0]
        used_hashes.add(h)
        matched += 1
        out_rows.append({
            "commit_hash": h,
            "project": project,
            "message": subject.strip(),
            "buggy": 1 if row["contains_bug"].strip().lower() == "true" else 0,
            "author_date_unix_timestamp": ts,
            **{col: row[col] for col in ARFF_COLUMNS if col not in
               ("contains_bug", "author_date_unix_timestamp", "commit_type")},
        })

    rate = matched / len(rows) * 100
    print(f"  Matched: {matched}/{len(rows)} ({rate:.1f}%)  "
          f"ambiguous_ts_collisions={ambiguous}  unmatched={unmatched}")

    if verify_only:
        return

    out_path = DATASET_DIR / f"geocabral_{project}_recovered.csv"
    fields = ["commit_hash", "project", "message", "buggy", "author_date_unix_timestamp",
              "fix", "ns", "nd", "nf", "entrophy", "la", "ld", "lt",
              "ndev", "age", "nuc", "exp", "rexp", "sexp"]
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"  -> {out_path}")


def parse_args():
    verify_only = "--verify" in sys.argv
    run_all     = "--all" in sys.argv
    single      = None
    if "--project" in sys.argv:
        idx = sys.argv.index("--project")
        single = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
    return verify_only, run_all, single


def main():
    verify_only, run_all, single = parse_args()

    if single:
        projects = [single]
    elif run_all:
        projects = [p for p, u in REPO_MAP.items() if u]
    else:
        projects = PILOT_PROJECTS

    print(f"Projects: {projects}  |  verify_only={verify_only}")
    for project in projects:
        recover_project(project, verify_only)


if __name__ == "__main__":
    main()
