"""
Convert opus-research PRemo dataset.json → gomi-compatible CSV.

Source:  ../temp/opus-research-sentiment-dataset/data/dataset.json
Output:  ../dataset/sentiment/premo_labeled.csv

Mapping:
  positive  → satisfaction
  negative  → caution
  neutral   → neutral
  undefined → skipped
"""

import json
import csv
from pathlib import Path
from collections import Counter

SRC = Path(__file__).parent.parent.parent / "temp/opus-research-sentiment-dataset/data/dataset.json"
OUT = Path(__file__).parent.parent / "dataset/sentiment/premo_labeled.csv"

POLARITY_MAP = {
    "positive": "satisfaction",
    "negative": "caution",
    "neutral":  "neutral",
}

def main():
    data = json.loads(SRC.read_text())
    print(f"Loaded {len(data)} records from dataset.json")

    rows = []
    skipped = 0

    for item in data:
        polarity = item.get("part2_aggregate", {}).get("polarity", "")
        label = POLARITY_MAP.get(polarity)
        if label is None:
            skipped += 1
            continue
        msg = item.get("raw_message", "").strip()
        if not msg:
            skipped += 1
            continue
        rows.append({"message": msg, "reconciled_emotion": label})

    print(f"Skipped: {skipped}")
    print(f"Writing {len(rows)} rows to {OUT}")

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
