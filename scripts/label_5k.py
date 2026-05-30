"""
label_5k.py — Apply trained DistilBERT sentiment model to 5k unlabeled commits.

Reads:  datasets/openreview/cleaned20k.csv           (full cleaned corpus)
        datasets/openreview/openreview_labeled_2k.csv (already labeled — excluded)
Model:  scripts/datasets/distilbert_sentiment

Writes: datasets/openreview/openreview_labeled_5k_auto.csv
        Columns: commit, author, date, repo, message, reconciled_emotion, confidence
"""

import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
DATASET_DIR       = ROOT / "datasets"
SCRIPTS_DS_DIR    = ROOT / "scripts" / "datasets"
CLEANED_CSV       = DATASET_DIR / "cleaned20k.csv"
LABELED_CSV       = DATASET_DIR / "openreview_labeled_2k.csv"
MODEL_DIR         = SCRIPTS_DS_DIR / "distilbert_sentiment"
OUTPUT_CSV        = DATASET_DIR / "openreview_labeled_5k_auto.csv"

SAMPLE_SIZE  = 5000
BATCH_SIZE   = 64

# Hugging Face Hub (optional)
load_dotenv()
HF_SENTIMENT_MODEL_REPO = os.getenv("GOMI_SENTIMENT_MODEL_REPO")
HF_SENTIMENT_MODEL_REVISION = os.getenv("GOMI_SENTIMENT_MODEL_REVISION")
HF_DATASET_REPO = os.getenv("GOMI_DATASET_REPO", "GitRatBCSAD/gomi-datasets")
HF_DATASET_REVISION = os.getenv("GOMI_DATASET_REVISION")


def _get_hf_token():
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _dataset_subdir_for_filename(filename: str) -> str | None:
    if filename in {"cleaned20k.csv", "openreview_labeled_2k.csv", "openreview_labeled_5k_auto.csv"}:
        return "openreview"
    if filename.startswith("openreview_"):
        return "openreview"
    if filename == "apachejit_commits.csv":
        return "jit"
    if filename.endswith("_test_raw.pkl") or filename.endswith("_k_feature.csv"):
        return "jit"
    return None


def _resolve_dataset_file(path: Path) -> Path | None:
    from hf_datasets import hub_download_dataset_file

    if not HF_DATASET_REPO:
        return None
    try:
        downloaded = hub_download_dataset_file(
            HF_DATASET_REPO,
            path.name,
            revision=HF_DATASET_REVISION,
            token=_get_hf_token(),
        )
    except ImportError as e:
        print(f"\nERROR: Missing dependency — {e}")
        print("Install with: pip install huggingface_hub\n")
        return None
    if not downloaded:
        return None
    return Path(downloaded)


def main():
    resolved_labeled = _resolve_dataset_file(LABELED_CSV)
    if not resolved_labeled:
        print(f"ERROR: labeled dataset not found at {LABELED_CSV}")
        if HF_DATASET_REPO:
            print(f"  Also checked Hugging Face dataset repo: {HF_DATASET_REPO}")
        sys.exit(1)

    resolved_cleaned = _resolve_dataset_file(CLEANED_CSV)
    if not resolved_cleaned:
        print(f"ERROR: cleaned dataset not found at {CLEANED_CSV}")
        if HF_DATASET_REPO:
            print(f"  Also checked Hugging Face dataset repo: {HF_DATASET_REPO}")
        sys.exit(1)

    model_source = MODEL_DIR if MODEL_DIR.is_dir() else HF_SENTIMENT_MODEL_REPO
    if not model_source:
        print(f"ERROR: model not found at {MODEL_DIR}")
        print("  Set GOMI_SENTIMENT_MODEL_REPO to load from Hugging Face Hub.")
        sys.exit(1)
    if model_source == HF_SENTIMENT_MODEL_REPO and _get_hf_token() is None:
        print("NOTE: HF_TOKEN or HUGGINGFACE_HUB_TOKEN not set; private repos will fail.")

    # Load commits already in 2k labeled set
    already_labeled: set[str] = set()
    with open(resolved_labeled, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            already_labeled.add(row["commit"])
    print(f"Excluding {len(already_labeled)} already-labeled commits")

    # Collect unlabeled rows from cleaned20k
    candidates = []
    with open(resolved_cleaned, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["commit"] not in already_labeled and row.get("message", "").strip():
                candidates.append(row)
    print(f"Unlabeled candidates: {len(candidates)}")

    # Take first SAMPLE_SIZE (preserves original ordering)
    sample = candidates[:SAMPLE_SIZE]
    print(f"Sampling: {len(sample)}")

    # Load model
    source_desc = str(model_source)
    if model_source == HF_SENTIMENT_MODEL_REPO and HF_SENTIMENT_MODEL_REVISION:
        source_desc = f"{model_source}@{HF_SENTIMENT_MODEL_REVISION}"
    print(f"\nLoading DistilBERT from {source_desc} ...")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline as hf_pipeline
    token = _get_hf_token()
    revision = HF_SENTIMENT_MODEL_REVISION if model_source == HF_SENTIMENT_MODEL_REPO else None
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_source),
        revision=revision,
        token=token,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_source),
        revision=revision,
        token=token,
    )
    classifier = hf_pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        truncation=True,
        max_length=128,
        device=-1,   # CPU; change to 0 for GPU
    )
    print("Model loaded.\n")

    # Run batch inference
    messages = [row["message"][:512] for row in sample]
    results  = []
    total    = len(messages)

    for start in range(0, total, BATCH_SIZE):
        batch = messages[start : start + BATCH_SIZE]
        preds = classifier(batch)
        results.extend(preds)
        done = min(start + BATCH_SIZE, total)
        print(f"  [{done}/{total}]", end="\r", flush=True)

    print(f"\nInference done. Writing {OUTPUT_CSV} ...")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["commit", "author", "date", "repo",
                        "message", "reconciled_emotion", "confidence"],
        )
        writer.writeheader()

        label_counts: dict[str, int] = {}
        for row, pred_list in zip(sample, results):
            best   = max(pred_list, key=lambda x: x["score"])
            label  = best["label"].lower().replace("label_", "")
            conf   = round(best["score"], 4)
            label_counts[label] = label_counts.get(label, 0) + 1
            writer.writerow({
                "commit":             row["commit"],
                "author":             row["author"],
                "date":               row["date"],
                "repo":               row["repo"],
                "message":            row["message"],
                "reconciled_emotion": label,
                "confidence":         conf,
            })

    print(f"\nDone. Label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<14} {count:>5}  ({100*count/len(sample):.1f}%)")
    print(f"\nOutput: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
