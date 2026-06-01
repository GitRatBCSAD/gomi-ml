# train_sentiment.py — One-time offline training script
#
# Fine-tunes distilbert-base-uncased on the combined OpenReview 2025 dataset:
#   - openreview_labeled_2k.csv  (2,000 human-labeled — gold standard)
#   - openreview_labeled_5k_auto.csv (5,000 auto-labeled by DistilBERT, filtered
#     by confidence >= MIN_CONFIDENCE to keep only reliable labels)
#
# Combined: ~7,000 training samples (2.5x more than the 2k-only run)
#
# Labels: frustration | caution | neutral | satisfaction
# Risk-positive labels (used by gomi.py): frustration, caution
#
# Output: saved HuggingFace model + tokenizer at datasets/distilbert_sentiment/
#         This directory is what gomi.py loads at runtime — run this script once.
#
# Usage:
#   pip install transformers datasets scikit-learn torch huggingface_hub
#   python train_sentiment.py
#
# Expected CSV columns in openreview_labeled_2k.csv:
#   message               — the raw commit message text
#   reconciled_emotion    — one of: frustration, caution, neutral, satisfaction

import csv
import os
import sys

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ─── PATHS ────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
DATASET_DIR  = os.path.join(SCRIPT_DIR, "datasets")
OPENREVIEW_CSV = "openreview_labeled_2k.csv"  # Hub: openreview/openreview_labeled_2k.csv
OUTPUT_DIR   = os.path.join(DATASET_DIR, "distilbert_sentiment")

# 5k auto-labeled dataset (produced by label_5k.py)
OPENREVIEW_5K_CSV = "openreview_labeled_5k_auto.csv"  # Hub: openreview/openreview_labeled_5k_auto.csv

BASE_MODEL   = "distilbert-base-uncased"

# Hugging Face Hub (optional)
load_dotenv()
HF_DATASET_REPO = os.getenv("GOMI_DATASET_REPO", "GitRatBCSAD/gomi-datasets")
HF_DATASET_REVISION = os.getenv("GOMI_DATASET_REVISION")
HF_SENTIMENT_MODEL_REPO = os.getenv("GOMI_SENTIMENT_MODEL_REPO")

# ─── LABEL SCHEME ─────────────────────────────────────────────────────────────

# Canonical label order — must stay consistent between training and gomi.py
LABELS      = ["frustration", "caution", "neutral", "satisfaction"]
LABEL2ID    = {l: i for i, l in enumerate(LABELS)}
ID2LABEL    = {i: l for i, l in enumerate(LABELS)}

VALID_EMOTIONS = set(LABELS)

# ─── HYPERPARAMETERS ──────────────────────────────────────────────────────────

MAX_LENGTH    = 128      # commit messages rarely exceed 128 tokens
BATCH_SIZE    = 32       # larger batch for ~7k dataset (was 16)
NUM_EPOCHS    = 5        # 5 epochs on ~7k combined dataset (was 4 on 2k)
LEARNING_RATE = 2e-5    # standard for DistilBERT fine-tuning
WEIGHT_DECAY  = 0.01
TEST_SIZE     = 0.15     # 15% held out for evaluation reporting (not used by gomi.py)
RANDOM_SEED   = 42

# Confidence threshold for auto-labeled samples from label_5k.py
# Rows where DistilBERT confidence < MIN_CONFIDENCE are excluded from training.
# 0.50 keeps ~95% of 5k rows; raise to 0.60 for a stricter quality filter.
MIN_CONFIDENCE = 0.90

# ─── LOAD DATASET ─────────────────────────────────────────────────────────────

def _get_hf_token():
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _download_dataset_csv(filename: str) -> str | None:
    from hf_datasets import hub_download_dataset_file

    if not HF_DATASET_REPO:
        return None
    try:
        return hub_download_dataset_file(
            HF_DATASET_REPO,
            filename,
            revision=HF_DATASET_REVISION,
            token=_get_hf_token(),
        )
    except ImportError as e:
        print(f"\nERROR: Missing dependency — {e}")
        print("Install with: pip install huggingface_hub\n")
        return None


def load_openreview(
    filename: str = OPENREVIEW_CSV,
    min_confidence: float | None = None,
) -> tuple[list[str], list[int]]:
    """
    Reads a labeled OpenReview CSV and returns (messages, label_ids).

    For the human-labeled 2k dataset, min_confidence is ignored (no column).
    For the auto-labeled 5k dataset, rows below min_confidence are skipped
    to filter out low-quality auto-labels.

    Rows with missing/invalid labels are always skipped.
    """
    resolved_csv = _download_dataset_csv(filename)
    if not resolved_csv:
        rev = f" (revision={HF_DATASET_REVISION})" if HF_DATASET_REVISION else ""
        print(f"\nERROR: Could not download dataset from Hugging Face.")
        print(f"  Repo: {HF_DATASET_REPO}{rev}")
        print(f"  File: openreview/{filename}")
        print("  Set GOMI_DATASET_REPO and HF_TOKEN if the repo is private.")
        print("  Clear GOMI_DATASET_REVISION until that tag exists on the Hub.\n")
        sys.exit(1)

    messages, label_ids = [], []
    skipped = 0
    low_conf = 0

    with open(resolved_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            msg     = row.get("message", "").strip()
            emotion = row.get("reconciled_emotion", "").strip().lower()
            if not msg or emotion not in VALID_EMOTIONS:
                skipped += 1
                continue
            # Confidence filtering: only applies when the column exists
            if min_confidence is not None and "confidence" in row:
                try:
                    conf = float(row["confidence"])
                    if conf < min_confidence:
                        low_conf += 1
                        continue
                except (ValueError, TypeError):
                    pass
            messages.append(msg)
            label_ids.append(LABEL2ID[emotion])

    note = ""
    if low_conf:
        note = f", {low_conf} below confidence {min_confidence}"
    print(f"  Loaded {len(messages)} rows  ({skipped} skipped — missing/invalid label{note})")

    # Print label distribution so class imbalance is visible
    from collections import Counter
    dist = Counter(LABELS[i] for i in label_ids)
    for label, count in sorted(dist.items()):
        pct = 100 * count / len(label_ids)
        print(f"    {label:<14} {count:>4}  ({pct:.1f}%)")

    return messages, label_ids


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train():
    # ── Import heavy deps here so the file can be read without them installed ──
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            DataCollatorWithPadding,
        )
        from datasets import Dataset
    except ImportError as e:
        print(f"\nERROR: Missing dependency — {e}")
        print("Install with: pip install transformers datasets torch scikit-learn\n")
        sys.exit(1)

    print("=" * 60)
    print("  GOMI — DistilBERT Sentiment Fine-tuning")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Dataset    : {HF_DATASET_REPO}/openreview/{OPENREVIEW_CSV} + {OPENREVIEW_5K_CSV} (conf>={MIN_CONFIDENCE})")
    print(f"  Output     : {OUTPUT_DIR}")
    print(f"  Epochs     : {NUM_EPOCHS}  |  LR: {LEARNING_RATE}  |  Batch: {BATCH_SIZE}")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/4] Loading combined OpenReview dataset (2k human + 5k auto)...")

    # Source 1: 2k human-labeled — gold standard, no confidence filtering
    print("  [2k human-labeled]")
    msgs_2k, ids_2k = load_openreview(OPENREVIEW_CSV, min_confidence=None)

    # Source 2: 5k auto-labeled — filter by confidence to keep reliable labels
    print(f"  [5k auto-labeled, confidence >= {MIN_CONFIDENCE}]")
    # Check locally first, then fall back to HuggingFace
    local_5k = os.path.join(DATASET_DIR, "openreview", OPENREVIEW_5K_CSV)
    if os.path.isfile(local_5k):
        # Load directly from local file (already downloaded by label_5k.py)
        import csv as _csv
        msgs_5k, ids_5k = [], []
        skipped_5k, low_conf_5k = 0, 0
        with open(local_5k, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                msg     = row.get("message", "").strip()
                emotion = row.get("reconciled_emotion", "").strip().lower()
                if not msg or emotion not in VALID_EMOTIONS:
                    skipped_5k += 1
                    continue
                try:
                    conf = float(row.get("confidence", 1.0))
                except (ValueError, TypeError):
                    conf = 1.0
                if conf < MIN_CONFIDENCE:
                    low_conf_5k += 1
                    continue
                msgs_5k.append(msg)
                ids_5k.append(LABEL2ID[emotion])
        print(f"  Loaded {len(msgs_5k)} rows from local file "
              f"({skipped_5k} invalid, {low_conf_5k} below confidence {MIN_CONFIDENCE})")
        from collections import Counter as _Counter
        dist_5k = _Counter(LABELS[i] for i in ids_5k)
        for label, count in sorted(dist_5k.items()):
            pct = 100 * count / max(len(ids_5k), 1)
            print(f"    {label:<14} {count:>4}  ({pct:.1f}%)")
    else:
        msgs_5k, ids_5k = load_openreview(OPENREVIEW_5K_CSV, min_confidence=MIN_CONFIDENCE)

    print("  [1.6k SentiCR human-labeled]")
    try:
        msgs_scr, ids_scr = load_openreview("senticr_labeled.csv", min_confidence=None)
    except Exception as e:
        print(f"    [skip] SentiCR not found: {e}")
        msgs_scr, ids_scr = [], []

    # Combine: human-labeled first (higher quality), then auto-labeled
    messages  = msgs_2k + msgs_5k + msgs_scr
    label_ids = ids_2k  + ids_5k  + ids_scr
    print(f"\n  Combined: {len(messages)} total samples "
          f"({len(msgs_2k)} OR human + {len(msgs_5k)} auto-labeled + {len(msgs_scr)} SentiCR)")

    train_msgs, val_msgs, train_labels, val_labels = train_test_split(
        messages, label_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=label_ids,      # preserve class balance in both splits
    )
    print(f"  Train: {len(train_msgs)}  |  Val: {len(val_msgs)}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"\n[2/4] Loading tokenizer ({BASE_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_ds = Dataset.from_dict({"text": train_msgs, "label": train_labels})
    val_ds   = Dataset.from_dict({"text": val_msgs,   "label": val_labels})

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds   = val_ds.map(tokenize,   batched=True)

    train_ds = train_ds.remove_columns(["text"])
    val_ds   = val_ds.remove_columns(["text"])

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[3/4] Loading {BASE_MODEL} and attaching classification head...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n[4/4] Fine-tuning for {NUM_EPOCHS} epochs on {len(train_msgs)} samples...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        report = classification_report(
            labels, preds,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        )
        return {
            "accuracy":  report["accuracy"],
            "f1_macro":  report["macro avg"]["f1-score"],
            "precision": report["macro avg"]["precision"],
            "recall":    report["macro avg"]["recall"],
        }

    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=20,
        report_to="none",       # disable wandb/mlflow
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving fine-tuned model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\nFinal evaluation on validation set:")
    preds_out = trainer.predict(val_ds)
    preds     = np.argmax(preds_out.predictions, axis=-1)
    print(classification_report(
        val_labels, preds,
        target_names=LABELS,
        zero_division=0,
    ))

    if HF_SENTIMENT_MODEL_REPO:
        print("\nUploading model to Hugging Face Hub...")
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            print(f"  ERROR: Missing dependency — {e}")
            print("  Install with: pip install huggingface_hub")
        else:
            token = _get_hf_token()
            if not token:
                print("  Skipping upload: HF_TOKEN or HUGGINGFACE_HUB_TOKEN not set.")
            else:
                api = HfApi()
                api.create_repo(
                    repo_id=HF_SENTIMENT_MODEL_REPO,
                    repo_type="model",
                    exist_ok=True,
                    token=token,
                )
                api.upload_folder(
                    repo_id=HF_SENTIMENT_MODEL_REPO,
                    repo_type="model",
                    folder_path=OUTPUT_DIR,
                    path_in_repo=".",
                    token=token,
                )
                print(f"  Uploaded to: {HF_SENTIMENT_MODEL_REPO}")

    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  Dataset    : {len(messages)} samples ({len(msgs_2k)} human + {len(msgs_5k)} auto)")
    print(f"  Model saved → {OUTPUT_DIR}")
    print("  Next:  uv run python scripts/train_risk_model.py")
    print("  Then:  python gomi.py <repo_path>")
    print("=" * 60)


if __name__ == "__main__":
    train()
