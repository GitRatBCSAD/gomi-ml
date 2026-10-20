#!/usr/bin/env python3
"""
Replace DeepJIT's tokenized message bodies with real git subject lines.

deepjit_{p}_commits.csv carries DeepJIT's preprocessed text (lower-cased, punctuation
space-separated, full body). The runtime classifies `git log %s` (subject only), so the
text must match. Same idea as recover_promise_messages.py, but one `git log --no-walk
--stdin` per repo instead of one subprocess per hash.

Reuses REPO_MAP, clone_all_repos() and build_hash_index() from extract_deepjit_lizard.py
(shared dataset/.clones/ dir). Commits whose hash is in no repo are DROPPED: their
tokenized text is not in the runtime format and must not leak into the pool.

Usage:
    python scripts/recover_deepjit_subjects.py
    python scripts/recover_deepjit_subjects.py --project qt

Rewrites in place: gomi-datasets/jit/deepjit/deepjit_{project}_commits.csv
"""

import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from extract_deepjit_lizard import (
    DEEPJIT_DIR, DEEPJIT_PROJECTS, build_hash_index, clone_all_repos,
)

SEP = "\x1f"
GIT_TIMEOUT = 600  # per repo batch


def subjects_in_repo(repo: Path, hashes: list) -> dict:
    """{hash: subject} for hashes in one repo, via a single git log call."""
    r = subprocess.run(
        ["git", "--git-dir", str(repo), "log", "--no-walk=unsorted", "--stdin",
         f"--format=%H{SEP}%s"],
        input="\n".join(hashes).encode(), capture_output=True, timeout=GIT_TIMEOUT,
    )
    out = {}
    for line in r.stdout.decode("utf-8", errors="replace").splitlines():
        h, _, subj = line.partition(SEP)
        if h:
            out[h] = subj.strip()
    return out


def recover_project(project: str) -> None:
    print(f"\n{'='*60}\nProject: {project}")
    path = DEEPJIT_DIR / f"deepjit_{project}_commits.csv"
    if not path.exists():
        print(f"  missing {path}")
        return

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    hashes = [r["commit_hash"] for r in rows]
    print(f"  {len(rows)} commits")

    repos = clone_all_repos(project)
    if not repos:
        print("  Skipping, no repos available. File left unchanged.")
        return

    index = build_hash_index(project, repos, hashes)
    by_repo = defaultdict(list)
    for h, rp in index.items():
        by_repo[rp].append(h)

    subjects = {}
    for rp, hs in by_repo.items():
        subjects.update(subjects_in_repo(rp, hs))

    kept = []
    for r in rows:
        subj = subjects.get(r["commit_hash"])
        if subj is None:
            continue
        r["message"] = subj
        kept.append(r)

    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    tmp.replace(path)

    print(f"  recovered={len(kept)}  dropped={len(rows) - len(kept)}")
    print(f"  -> {path}")


def main():
    projects = DEEPJIT_PROJECTS
    if "--project" in sys.argv:
        projects = [sys.argv[sys.argv.index("--project") + 1]]
    for p in projects:
        recover_project(p)
    print(f"\n{'='*60}\nDone. Re-run the dataset audit, then upload deepjit_*_commits.csv.")


if __name__ == "__main__":
    main()
