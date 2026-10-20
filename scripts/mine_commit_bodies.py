#!/usr/bin/env python3
"""
Add a cleaned commit-message body (`message_body`) to every JIT commits CSV.

The notebook's commit-body experiment compares DistilBERT on the subject line (what the
runtime classifies today) with subject + body (what openreview, DistilBERT's only
commit-message training set, contains). `message` (the subject) is left untouched, so
the fix-keyword label and fix_ratio keep their definitions.

Body = everything after the first blank line of `git log %B` (the part `%s` leaves out),
minus trailer lines (Signed-off-by, Change-Id, Fixes:, ...), "Fixes #123"-style issue
closers, "(cherry picked from ...)" lines and URLs, whitespace-collapsed.

Repos: DeepJIT reuses its full clones (extract_deepjit_lizard.py). Everything else gets a
commits-only clone (--filter=tree:0) under dataset/.clones/msg/, reused across sources
(camel, nova, tomcat ... appear in more than one).

Usage:
    python scripts/mine_commit_bodies.py                          # all sources
    python scripts/mine_commit_bodies.py --source geocabral
    python scripts/mine_commit_bodies.py --source deepjit --project jdt --verify

--verify samples 100 hashes per project, prints coverage and example bodies, writes nothing.

Rewrites in place: gomi-datasets/jit/<source>/*_commits.csv (adds/updates `message_body`).
"""

import argparse
import csv
import random
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from extract_apache_lizard import REPO_OVERRIDES
from extract_deepjit_lizard import CLONE_DIR, clone_all_repos, hashes_in_repo, repo_name
from recover_geocabral_messages import REPO_MAP as GEOCABRAL_REPOS

JIT_DIR = Path(__file__).resolve().parents[2] / "gomi-datasets" / "jit"
MSG_CLONE_DIR = CLONE_DIR / "msg"
SOURCES = ["deepjit", "apachejit", "geocabral", "jitfine", "promise"]
GIT_TIMEOUT = 1800

TRAILER_RE = re.compile(
    r"^\s*(signed-off-by|change-id|reviewed-on|reviewed-by|tested-by|acked-by|co-authored-by|cc"
    r"|task-number|run-trybot|trybot-result|closes-bug|partial-bug|related-bug|implements"
    r"|depends-on|fixes|updates|bug|conflicts|git-svn-id)\s*:",
    re.IGNORECASE,
)
CLOSER_RE = re.compile(r"^\s*(fix(e[sd])?|close[sd]?|resolve[sd]?|updates?)\s+\S*#\d+\W*$", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+")
# Pasted "git log" headers (ported/cherry-picked commits): metadata, not tone. The quoted message stays.
HEADER_RE = re.compile(r"^\s*(commit\s+[0-9a-f]{7,40}\b|author:|date:)", re.IGNORECASE)

csv.field_size_limit(1 << 30)


def clean_body(full: str) -> str:
    parts = full.replace("\r\n", "\n").split("\n\n", 1)
    if len(parts) < 2:
        return ""
    keep = [line for line in parts[1].splitlines()
            if not (TRAILER_RE.match(line) or CLOSER_RE.match(line) or HEADER_RE.match(line)
                    or line.lstrip().lower().startswith("(cherry picked from"))]
    return " ".join(URL_RE.sub(" ", " ".join(keep)).split())


def bodies_in_repo(repo: Path, hashes: list) -> dict:
    """{hash: cleaned body} for hashes present in repo, one git log call."""
    r = subprocess.run(
        ["git", "--git-dir", str(repo), "log", "--no-walk=unsorted", "--stdin", "--format=%H%x1f%B%x1e"],
        input="\n".join(hashes).encode(), capture_output=True, timeout=GIT_TIMEOUT,
    )
    out = {}
    for rec in r.stdout.decode("utf-8", errors="replace").split("\x1e"):
        h, _, full = rec.strip("\n").partition("\x1f")
        if h:
            out[h.strip()] = clean_body(full)
    return out


def msg_clone(url: str) -> Path | None:
    dest = MSG_CLONE_DIR / repo_name(url)
    if dest.exists():
        return dest
    MSG_CLONE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"    cloning {url} (commits only)")
    r = subprocess.run(["git", "clone", "--bare", "--quiet", "--filter=tree:0", url, str(dest)],
                       capture_output=True)
    if r.returncode != 0:
        print(f"    FAILED {url}: {r.stderr.decode()[:200]}")
        return None
    return dest


def repos_for(source: str, project: str) -> list:
    if source == "deepjit":
        return [p for _, p in clone_all_repos(project)]
    if source == "geocabral":
        url = GEOCABRAL_REPOS[project]
    elif source == "apachejit":
        url = f"https://github.com/{REPO_OVERRIDES.get(project, project)}"
    else:  # promise, jitfine: all apache/<project> (GitHub redirects renamed repos, e.g. parquet-mr)
        url = f"https://github.com/apache/{project}"
    repo = msg_clone(url)
    return [repo] if repo else []


def mine_project(source: str, project: str, hashes: list) -> dict:
    bodies = {}
    remaining = set(hashes)
    for repo in repos_for(source, project):
        _, found = hashes_in_repo(repo, sorted(remaining))
        if found:
            bodies.update(bodies_in_repo(repo, sorted(found)))
            remaining -= found
    return bodies


def report(source: str, project: str, hashes: list, bodies: dict, examples: int = 0):
    non_empty = [b for b in bodies.values() if b]
    med = statistics.median(len(b.split()) for b in non_empty) if non_empty else 0
    print(f"  {source}/{project}: found {len(bodies)}/{len(hashes)}  "
          f"with body {len(non_empty) / max(1, len(bodies)):.1%}  median body words {med}")
    for b in random.Random(0).sample(non_empty, min(examples, len(non_empty))):
        print(f"      | {b[:200]}")


def process_file(path: Path, source: str, only_project: str | None, verify: bool):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)

    by_project = defaultdict(list)
    for r in rows:
        if only_project is None or r["project"] == only_project:
            by_project[r["project"]].append(r["commit_hash"])
    if not by_project:
        return

    mined = {}
    for project, hashes in sorted(by_project.items()):
        if verify:
            hashes = random.Random(0).sample(hashes, min(100, len(hashes)))
        bodies = mine_project(source, project, hashes)
        report(source, project, hashes, bodies, examples=10 if verify else 0)
        mined.update({h: bodies.get(h, "") for h in hashes})
    if verify:
        return

    if "message_body" not in fields:
        fields.insert(fields.index("message") + 1, "message_body")
    for r in rows:
        r["message_body"] = mined.get(r["commit_hash"], r.get("message_body") or "")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=SOURCES)
    ap.add_argument("--project")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    for source in [args.source] if args.source else SOURCES:
        print(f"\n{'=' * 60}\n{source}")
        for path in sorted((JIT_DIR / source).glob("*_commits.csv")):
            process_file(path, source, args.project, args.verify)


if __name__ == "__main__":
    assert clean_body("subj\n\nWhy it broke.\n\nChange-Id: I12\nSigned-off-by: a <b>\nFixes #12") == "Why it broke."
    assert clean_body("subject only") == ""
    assert clean_body("s\n\nPort of:\ncommit 2622bc3f65162bf6d6cb5838da6999f8b5ca75cf\nAuthor: A <a@b>\nDate:   Sun Apr 6 2014\n\n    Clean up regex") == "Port of: Clean up regex"
    assert clean_body("s\n\nsee https://x.y/z for details\n(cherry picked from commit abc)") == "see for details"
    main()
