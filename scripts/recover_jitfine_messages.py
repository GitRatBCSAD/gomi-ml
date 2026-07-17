#!/usr/bin/env python3
"""
Extract commit hash + message + Kamei features + label for the JIT-Fine
dataset (JIT-Defect4J, built on LLTC4J / Herbold et al.).

Source: https://github.com/jacknichao/JIT-Fine
Paper: "The Best of Both Worlds: Integrating Semantic Features with Expert
Features for Defect Prediction and Localization"

Unlike geocabral, NO recovery/matching step is needed here — JIT-Fine's own
`features_{train,valid,test}.pkl` files already contain real commit_hash,
commit_message, the full Kamei feature set, project name, and a ready-made
`is_buggy_commit` label, all pre-joined in a pandas DataFrame. This script is
purely a reformat: read the 3 splits, pool them (JIT-Fine's own train/valid/
test split is for their own paper's evaluation, not something gomi needs to
preserve — DeepJIT/ApacheJIT are pooled the same way), group by project, and
write one CSV per project matching the exact column layout
recover_geocabral_messages.py produces, so extract_jitfine_lizard.py can
reuse that script's structure unchanged.

21 projects, 27,319 commits total, verified zero duplicate hashes across
splits, zero project overlap with DeepJIT/ApacheJIT/geocabral.

Usage:
    python scripts/recover_jitfine_messages.py

Output: dataset/jitfine_{project}_recovered.csv
Columns: commit_hash, project, message, buggy, author_date_unix_timestamp,
         fix, ns, nd, nf, entrophy, la, ld, lt, ndev, age, nuc, exp, rexp, sexp
"""

import csv
from pathlib import Path

import pandas as pd

REPO_ROOT   = Path(__file__).parent.parent.parent   # gomi/
JITFINE_DIR = REPO_ROOT / "temp" / "JIT-Fine" / "data" / "data" / "jitfine"
DATASET_DIR = Path(__file__).parent.parent / "dataset"

SPLITS = ["train", "valid", "test"]

FIELDS = ["commit_hash", "project", "message", "buggy", "author_date_unix_timestamp",
          "fix", "ns", "nd", "nf", "entrophy", "la", "ld", "lt",
          "ndev", "age", "nuc", "exp", "rexp", "sexp"]


def load_all_splits() -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        path = JITFINE_DIR / f"features_{split}.pkl"
        if not path.exists():
            print(f"  missing {path} — skipping split")
            continue
        df = pd.read_pickle(path)
        print(f"  {split}: {len(df)} rows")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def to_bool_int(v) -> int:
    return 1 if str(v).strip().lower() in ("true", "1", "1.0") else 0


def main():
    print("Loading JIT-Fine splits...")
    df = load_all_splits()
    total = len(df)
    dupes = df["commit_hash"].duplicated().sum()
    print(f"Total rows: {total}  |  duplicate hashes: {dupes}")
    if dupes:
        df = df.drop_duplicates(subset="commit_hash", keep="first")
        print(f"  dropped duplicates, {len(df)} remain")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    projects = sorted(df["project"].unique())
    print(f"Projects ({len(projects)}): {projects}")

    for project in projects:
        sub = df[df["project"] == project]
        out_path = DATASET_DIR / f"jitfine_{project}_recovered.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=FIELDS)
            writer.writeheader()
            for _, row in sub.iterrows():
                writer.writerow({
                    "commit_hash":                row["commit_hash"],
                    "project":                     project,
                    "message":                     str(row["commit_message"]).strip(),
                    "buggy":                       int(row["is_buggy_commit"]),
                    "author_date_unix_timestamp":  int(float(row["author_date_unix_timestamp"])),
                    "fix":                         to_bool_int(row["fix"]),
                    "ns":                          row["ns"],
                    "nd":                          row["nd"],
                    "nf":                          row["nf"],
                    "entrophy":                    row["entropy"],
                    "la":                          row["la"],
                    "ld":                          row["ld"],
                    "lt":                          row["lt"],
                    "ndev":                        row["ndev"],
                    "age":                         row["age"],
                    "nuc":                         row["nuc"],
                    "exp":                         row["exp"],
                    "rexp":                        row["rexp"],
                    "sexp":                        row["sexp"],
                })
        print(f"  [{project}] {len(sub)} rows -> {out_path}")

    print(f"\nDone. {len(projects)} projects, {total} commits total.")
    print("Next: run scripts/extract_jitfine_lizard.py to add Lizard complexity metrics.")


if __name__ == "__main__":
    main()
