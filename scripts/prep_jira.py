"""
Convert collab-uniba EMTK Jira emotion CSVs → gomi-compatible CSV.

Source:  ../temp/collab-uniba-EMTK_datasets-c200f78/jira/emotions/
Output:  ../dataset/sentiment/jira_labeled.csv

Merge strategy per item (keyed on shared numeric ID prefix):
  - Positive emotions: joy, love  → satisfaction
  - Negative emotions: anger, sadness → caution
  - Keep only if exactly one polarity fires (no ambiguous cross-firing)
  - Discard if nothing fires (would add to neutral problem)
  - Deduplicate on text after merge

Mapping:
  joy=YES or love=YES (no negative)  → satisfaction
  anger=YES or sadness=YES (no positive) → caution
"""

import csv
from pathlib import Path
from collections import Counter, defaultdict

SRC = Path(__file__).parent.parent.parent / "temp/collab-uniba-EMTK_datasets-c200f78/jira/emotions"
OUT = Path(__file__).parent.parent / "dataset/sentiment/jira_labeled.csv"

POSITIVE_EMOTIONS = {"joy", "love"}
NEGATIVE_EMOTIONS = {"anger", "sadness"}


def load_emotion_csv(path):
    """Returns dict of id → (label YES/NO, text)."""
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["id"]] = {"label": row["label"].strip().upper(), "text": row["text"].strip()}
    return result


def main():
    emotions = {}
    for name in ["anger", "joy", "love", "sadness"]:
        emotions[name] = load_emotion_csv(SRC / f"{name}.csv")
        print(f"Loaded {name}: {len(emotions[name])} rows")

    # Collect all IDs across all files
    all_ids = set()
    for data in emotions.values():
        all_ids.update(data.keys())
    print(f"\nUnique IDs across all files: {len(all_ids)}")

    rows = []
    seen_texts = set()
    stats = Counter()

    for item_id in all_ids:
        # Collect text (use any file that has this ID)
        text = None
        for data in emotions.values():
            if item_id in data:
                text = data[item_id]["text"]
                break
        if not text:
            continue

        # Check which emotions fire YES
        pos_fires = any(
            emotions[e].get(item_id, {}).get("label") == "YES"
            for e in POSITIVE_EMOTIONS
        )
        neg_fires = any(
            emotions[e].get(item_id, {}).get("label") == "YES"
            for e in NEGATIVE_EMOTIONS
        )

        if pos_fires and neg_fires:
            stats["ambiguous"] += 1
            continue
        if not pos_fires and not neg_fires:
            stats["nothing_fires"] += 1
            continue

        label = "satisfaction" if pos_fires else "caution"

        # Deduplicate on text
        key = text.lower().strip()
        if key in seen_texts:
            stats["duplicate"] += 1
            continue
        seen_texts.add(key)

        rows.append({"message": text, "reconciled_emotion": label})
        stats[label] += 1

    print(f"\nFilter stats:")
    for k, v in stats.items():
        print(f"  {k:<16} {v}")

    print(f"\nWriting {len(rows)} rows to {OUT}")
    dist = Counter(r["reconciled_emotion"] for r in rows)
    for lbl, cnt in sorted(dist.items()):
        print(f"  {lbl:<14} {cnt:>5}  ({100*cnt/len(rows):.1f}%)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["message", "reconciled_emotion"])
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")


if __name__ == "__main__":
    main()
