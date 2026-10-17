#!/usr/bin/env python3
"""
Extract commit hash + message + Kamei features + label for the PROMISE 2021
Defect Prediction Challenge dataset (SmartSHARK, Trautsch & Herbold).

Source: https://github.com/smartshark/promise-challenge

The provided CSVs are FILE-level (one row per file changed per commit), with
real commit hashes, a Kamei feature set (`kamei_*` columns, identical values
across every file-row of the same commit — verified), and a sparse per-bug
label matrix (`induces__<JIRAKEY>__<hash>__<date>` columns, '1'/'0' strings,
one column per known bug in that project). There is NO commit message column
anywhere in the data (searched all ~4000+ columns across multiple projects,
confirmed empty) — recovered here via `git log`, same as
recover_geocabral_messages.py, except simpler: the commit hash is already
given directly, no timestamp-matching/ambiguity needed at all.

4 of the 39 projects (activemq, kafka, zeppelin, zookeeper) already exist in
ApacheJIT — excluded here to avoid duplicate-commit risk. The remaining 35
are confirmed zero-overlap with DeepJIT/ApacheJIT/geocabral/JIT-Fine.

Steps per project:
  1. Parse the file-level CSV, group rows by commit hash.
  2. Collapse the buggy label: 1 if ANY induces__* column is '1' for ANY
     row (file) in that commit's group, else 0.
  3. Take Kamei features from the group's first row (identical across all
     files of the same commit, verified).
  4. Clone the repo (needed for message recovery; also reused by
     extract_promise_lizard.py — same shared dataset/.clones/ dir).
  5. Recover the message via `git log -1 --format=%s <hash>` — deterministic,
     no ambiguity, since the hash is already known.

Usage:
    python scripts/recover_promise_messages.py

Output: dataset/promise_{project}_recovered.csv
Columns: commit_hash, project, message, buggy, author_date_unix_timestamp,
         fix, ns, nd, nf, entrophy, la, ld, lt, ndev, age, nuc, exp, rexp, sexp
"""

import csv
import gzip
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.parent.parent   # gomi/
PROMISE_DIR = REPO_ROOT / "temp" / "promise-challenge" / "data"
DATASET_DIR = Path(__file__).parent.parent / "dataset"
CLONE_DIR   = DATASET_DIR / ".clones"  # shared with the other extract/recover scripts

# Excludes activemq/kafka/zeppelin/zookeeper — already in ApacheJIT.
# Also excludes pig and pdfbox — their GitHub mirror history was rewritten/
# re-migrated at some point (~April 2014, confirmed via a clean date-boundary
# split on manifoldcf's match/mismatch pattern), so SmartSHARK's mined commit
# hashes no longer exist in the current repo at all (pig: 0/2059 recoverable,
# pdfbox: 9/5349). manifoldcf has the SAME issue for its pre-2014 commits —
# kept, but only the 1,497 post-migration commits that actually recover.
# All URLs verified via `git ls-remote` before writing this script.
PROJECTS = [
    "archiva", "calcite", "cayenne", "commons-jexl", "deltaspike", "derby",
    "directory-fortress-core", "eagle", "falcon", "flume", "helix",
    "httpcomponents-client", "httpcomponents-core", "jena", "jspwiki",
    "knox", "kylin", "lens", "mahout", "manifoldcf", "mina-sshd", "nifi",
    "nutch", "oozie", "phoenix", "ranger", "roller",
    "samza", "storm", "streams", "systemml", "tez", "tika",
]

FIELDS = ["commit_hash", "project", "message", "buggy", "author_date_unix_timestamp",
          "fix", "ns", "nd", "nf", "entrophy", "la", "ld", "lt",
          "ndev", "age", "nuc", "exp", "rexp", "sexp"]

GIT_TIMEOUT = 30


def clone_repo(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    print(f"  cloning {url}")
    r = subprocess.run(
        ["git", "clone", "--bare", "--quiet", url, str(dest)],
        capture_output=True,
    )
    if r.returncode != 0:
        print(f"    FAILED clone {url}: {r.stderr.decode()[:300]}")
        return False
    return True


def parse_committer_date(s: str) -> int:
    # e.g. "2007-11-03 04:41:15+00:00"
    return int(datetime.fromisoformat(s).timestamp())


def to_bool_int(v) -> int:
    return 1 if str(v).strip().lower() in ("true", "1") else 0


def load_and_group(project: str):
    """Returns {commit_hash: {kamei fields..., 'buggy': int, 'author_date_unix_timestamp': int}}"""
    path = PROMISE_DIR / f"{project}.csv.gz"
    if not path.exists():
        print(f"  [{project}] missing {path}")
        return {}

    groups = {}
    induces_cols = None
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if induces_cols is None:
            induces_cols = [c for c in reader.fieldnames if c.startswith("induces__")]

        for row in reader:
            h = row["commit"]
            if h not in groups:
                groups[h] = {
                    "author_date_unix_timestamp": parse_committer_date(row["committer_date"]),
                    "fix":     to_bool_int(row.get("kamei_fix", "")),
                    "ns":      row.get("kamei_ns", ""),
                    "nd":      row.get("kamei_nd", ""),
                    "nf":      row.get("kamei_nf", ""),
                    "entrophy": row.get("kamei_entropy", ""),
                    "la":      row.get("kamei_la", ""),
                    "ld":      row.get("kamei_ld", ""),
                    "lt":      row.get("kamei_lt", ""),
                    "ndev":    row.get("kamei_ndev", ""),
                    "age":     row.get("kamei_age", ""),
                    "nuc":     row.get("kamei_nuc", ""),
                    "exp":     row.get("kamei_exp", ""),
                    "rexp":    row.get("kamei_rexp", ""),
                    "sexp":    row.get("kamei_sexp", ""),
                    "buggy":   0,
                }
            if groups[h]["buggy"] == 0 and any(row[c] == "1" for c in induces_cols):
                groups[h]["buggy"] = 1

    return groups


def recover_message(repo: Path, commit_hash: str):
    """Returns the message string on success, or None on failure — distinct
    from a genuinely-empty-but-successful git log (rare, but a real message
    can be an empty string, so failure must be a separate sentinel, not '')."""
    try:
        r = subprocess.run(
            ["git", "--git-dir", str(repo), "log", "-1", "--format=%s", commit_hash],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def recover_project(project: str) -> None:
    print(f"\n{'='*60}\nProject: {project}")

    groups = load_and_group(project)
    print(f"  {len(groups)} distinct commits")
    if not groups:
        return

    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    repo = CLONE_DIR / project
    url = f"https://github.com/apache/{project}"
    if not clone_repo(url, repo):
        return

    out_path = DATASET_DIR / f"promise_{project}_recovered.csv"
    matched, unmatched = 0, 0
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDS)
        writer.writeheader()
        for h, feats in groups.items():
            msg = recover_message(repo, h)
            if msg is None:
                unmatched += 1
                continue
            matched += 1
            writer.writerow({
                "commit_hash": h,
                "project": project,
                "message": msg,
                **feats,
            })

    print(f"  matched={matched}  unmatched={unmatched}")
    print(f"  -> {out_path}")


def main():
    for project in PROJECTS:
        recover_project(project)

    print(f"\n{'='*60}\nDone.")
    print("Next: run scripts/extract_promise_lizard.py to add Lizard complexity metrics.")


if __name__ == "__main__":
    main()
