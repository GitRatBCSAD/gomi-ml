# train_sentiment.py — CPU OPTIMIZED One-time offline training script
#
# Fine-tunes distilbert-base-uncased on a balanced, multi-source dataset.
# Optimized specifically for local execution on standard laptop CPUs.
#
# Usage:
#   pip install transformers datasets scikit-learn torch huggingface_hub nlpaug nltk
#   python train_sentiment.py

import csv
import os
import sys
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# NLTK resources required for Data Augmentation
import nltk
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    print("Downloading required NLTK resources for augmentation...")
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

import nlpaug.augmenter.word as naw

# ─── PATHS ────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
DATASET_DIR  = os.path.join(SCRIPT_DIR, "datasets")
OUTPUT_DIR   = os.path.join(DATASET_DIR, "distilbert_sentiment")

OPENREVIEW_CSV = "openreview_labeled_2k.csv"
SENTICR_CSV    = "senticr_labeled.csv"
SO_CSV         = "StackOverflow.csv"

BASE_MODEL   = "distilbert-base-uncased"

# Hugging Face Hub
load_dotenv()
HF_DATASET_REPO = os.getenv("GOMI_DATASET_REPO", "GitRatBCSAD/gomi-datasets")
HF_DATASET_REVISION = os.getenv("GOMI_DATASET_REVISION")
HF_SENTIMENT_MODEL_REPO = os.getenv("GOMI_SENTIMENT_MODEL_REPO")

# ─── LABEL SCHEME ─────────────────────────────────────────────────────────────

LABELS      = ["frustration", "caution", "neutral", "satisfaction"]
LABEL2ID    = {l: i for i, l in enumerate(LABELS)}
ID2LABEL    = {i: l for i, l in enumerate(LABELS)}
VALID_EMOTIONS = set(LABELS)

# ─── CPU HYPERPARAMETERS ──────────────────────────────────────────────────────

MAX_LENGTH    = 128
BATCH_SIZE    = 8         # REDUCED for Laptop CPU RAM safety
NUM_EPOCHS    = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
TEST_SIZE     = 0.15
RANDOM_SEED   = 42

# ─── LOAD DATASET ─────────────────────────────────────────────────────────────

def _get_hf_token():
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

def _download_dataset_csv(filename: str) -> str | None:
    from huggingface_hub import hf_hub_download
    if not HF_DATASET_REPO:
        return None
    try:
        return hf_hub_download(
            repo_id=HF_DATASET_REPO, 
            filename=f"openreview/{filename}", 
            repo_type="dataset", 
            revision=HF_DATASET_REVISION,
            token=_get_hf_token()
        )
    except Exception as e:
        print(f"\nERROR: Could not download {filename} — {e}")
        return None

def load_labeled_dataset(filename: str, so_mapping: bool = False) -> tuple[list[str], list[int]]:
    resolved_csv = _download_dataset_csv(filename)
    if not resolved_csv:
        sys.exit(1)

    messages, label_ids = [], []
    skipped = 0

    with open(resolved_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if so_mapping:
                msg = row.get("text", "").strip()
                oracle = row.get("oracle", "").strip()
                if oracle == "1": emotion = "satisfaction"
                elif oracle == "0": emotion = "neutral"
                elif oracle == "-1": emotion = "frustration"
                else:
                    skipped += 1
                    continue
            else:
                msg = row.get("message", "").strip()
                emotion = row.get("reconciled_emotion", "").strip().lower()

            if not msg or emotion not in VALID_EMOTIONS:
                skipped += 1
                continue
            
            messages.append(msg)
            label_ids.append(LABEL2ID[emotion])

    print(f"  {filename}: Loaded {len(messages)} rows ({skipped} skipped)")
    return messages, label_ids


# ─── CUSTOM TRAINER ───────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """Subclassed Trainer to apply custom class weights to the CrossEntropyLoss"""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Safely map weights to the device (guaranteed CPU in this config)
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


# ─── TRAINING PIPELINE ────────────────────────────────────────────────────────

def train():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
        from datasets import Dataset
    except ImportError as e:
        print(f"\nERROR: Missing dependency — {e}")
        sys.exit(1)

    print("=" * 60)
    print("  GOMI — DistilBERT Fine-tuning (LOCAL CPU MODE)")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Output     : {OUTPUT_DIR}")
    print("=" * 60)

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading datasets from HuggingFace...")
    msgs_2k, ids_2k = load_labeled_dataset(OPENREVIEW_CSV)
    msgs_scr, ids_scr = load_labeled_dataset(SENTICR_CSV)
    msgs_so, ids_so = load_labeled_dataset(SO_CSV, so_mapping=True)

    # ── 2. Data Augmentation (Satisfaction) ───────────────────────────────────
    print("\n[2/5] Augmenting Satisfaction Data (nlpaug)...")
    aug = naw.SynonymAug(aug_src='wordnet')
    satisfaction_id = LABEL2ID['satisfaction']

    original_satisfaction_msgs = [m for m, i in zip(msgs_2k, ids_2k) if i == satisfaction_id]
    augmented_msgs, augmented_ids = [], []
    
    print(f"  Generating synonyms for {len(original_satisfaction_msgs)} commits. This may take 1-2 minutes...")
    for msg in original_satisfaction_msgs:
        new_msg = aug.augment(msg)[0]
        augmented_msgs.append(new_msg)
        augmented_ids.append(satisfaction_id)

    msgs_2k += augmented_msgs
    ids_2k += augmented_ids
    print(f"  Satisfaction pool doubled. OpenReview pool is now {len(msgs_2k)} samples.")

    # ── 3. Data Balancing (Neutral Undersampling) ─────────────────────────────
    print("\n[3/5] Balancing datasets (Undersampling Neutral)...")
    combined_msgs = msgs_scr + msgs_so
    combined_ids = ids_scr + ids_so

    neutral_id = LABEL2ID['neutral']
    neutral_msgs, neutral_ids = [], []
    non_neutral_msgs, non_neutral_ids = [], []

    for m, i in zip(combined_msgs, combined_ids):
        if i == neutral_id:
            neutral_msgs.append(m)
            neutral_ids.append(i)
        else:
            non_neutral_msgs.append(m)
            non_neutral_ids.append(i)

    # Cap neutral non-OpenReview data to 1000 samples
    random.seed(RANDOM_SEED)
    sample_size = min(1000, len(neutral_msgs))
    sampled_indices = random.sample(range(len(neutral_msgs)), sample_size)
    sampled_neutral_msgs = [neutral_msgs[i] for i in sampled_indices]
    sampled_neutral_ids = [neutral_ids[i] for i in sampled_indices]

    # Final Merge
    messages = msgs_2k + non_neutral_msgs + sampled_neutral_msgs
    label_ids = ids_2k + non_neutral_ids + sampled_neutral_ids

    print(f"  Combined and Balanced Dataset: {len(messages)} total samples.")
    dist = Counter(LABELS[i] for i in label_ids)
    for label, count in sorted(dist.items()):
        print(f"    {label:<14} {count:>4}  ({100 * count / len(label_ids):.1f}%)")

    # ── Split & Format ────────────────────────────────────────────────────────
    train_msgs, val_msgs, train_labels, val_labels = train_test_split(
        messages, label_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=label_ids,
    )

    print(f"\n[4/5] Preparing Tokenizer & Class Weights...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_dict({"text": train_msgs, "label": train_labels}).map(tokenize, batched=True).remove_columns(["text"])
    val_ds = Dataset.from_dict({"text": val_msgs, "label": val_labels}).map(tokenize, batched=True).remove_columns(["text"])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    # Calculate exact class weights
    c_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    weights_tensor = torch.tensor(c_weights, dtype=torch.float32)
    print(f"  Calculated Loss Weights: {dict(zip(LABELS, np.round(c_weights, 3)))}")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Model & Fine-Tuning ───────────────────────────────────────────────────
    print(f"\n[5/5] Loading {BASE_MODEL} and beginning CPU training...")
    print("  ⚠️ WARNING: Training on a CPU will take several hours. Keep your laptop plugged in.")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        report = classification_report(labels, preds, target_names=LABELS, output_dict=True, zero_division=0)
        return {
            "accuracy":  report["accuracy"],
            "f1_macro":  report["macro avg"]["f1-score"],
            "precision": report["macro avg"]["precision"],
            "recall":    report["macro avg"]["recall"],
        }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ⚠️ CPU SAFE TRAINING ARGUMENTS ⚠️
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
        report_to="none",
        seed=RANDOM_SEED,
        use_cpu=True,              # Forces Hugging Face to stick to CPU
        dataloader_num_workers=0,  # Prevents freezing/crashing on laptop OS
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.class_weights_tensor = weights_tensor
    trainer.train()

    # ── Save & Evaluate ───────────────────────────────────────────────────────
    print(f"\nSaving fine-tuned model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nFinal evaluation on validation set:")
    preds_out = trainer.predict(val_ds)
    preds     = np.argmax(preds_out.predictions, axis=-1)
    print(classification_report(val_labels, preds, target_names=LABELS, zero_division=0))

    if HF_SENTIMENT_MODEL_REPO:
        print("\nUploading model to Hugging Face Hub...")
        try:
            from huggingface_hub import HfApi
            token = _get_hf_token()
            if token:
                api = HfApi()
                api.create_repo(repo_id=HF_SENTIMENT_MODEL_REPO, repo_type="model", exist_ok=True, token=token)
                api.upload_folder(repo_id=HF_SENTIMENT_MODEL_REPO, repo_type="model", folder_path=OUTPUT_DIR, path_in_repo=".", token=token)
                print(f"  Uploaded to: {HF_SENTIMENT_MODEL_REPO}")
        except Exception as e:
            print(f"  ERROR Uploading: {e}")

if __name__ == "__main__":
    train()
