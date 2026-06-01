# train_risk_model.py — One-time offline training script
#
# Trains the Logistic Regression risk fusion model on DeepJIT + ApacheJIT,
# using the fine-tuned DistilBERT model for sentiment features.
#
# Output:
#   scripts/datasets/risk_model.joblib
#   scripts/datasets/risk_model_shap_background.npy
#
# These artifacts are what gomi.py loads at runtime — run this script once
# (or again when JIT datasets or the sentiment model change).
#
# Prerequisites:
#   - scripts/datasets/distilbert_sentiment/  (run train_sentiment.py first)
#   - JIT datasets pulled from GOMI_DATASET_REPO on Hugging Face (required)
#
# Usage (from repo root):
#   uv run python scripts/train_risk_model.py

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # Force CPU-only inference

import sys
import csv
import random

import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gomi import (  # noqa: E402
    RISK_MODEL_PATH,
    RISK_SHAP_BACKGROUND_PATH,
    SCALER_PATH,
    load_sentiment_model,
    _load_deepjit_records,
    _resolve_dataset_file,
)

load_dotenv()
HF_RISK_MODEL_REPO = os.getenv("GOMI_RISK_MODEL_REPO")


def _get_hf_token():
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _percentile_rank(value: float, all_values: list[float]) -> float:
    if not all_values or len(all_values) == 1:
        return 0.0
    return round(sum(1 for x in all_values if x <= value) / len(all_values), 4)


def _load_apachejit_validation() -> list[dict]:
    filename = "apachejit_test_small.csv"
    resolved_csv = _resolve_dataset_file(filename)
    if not resolved_csv or not os.path.isfile(resolved_csv):
        print(f"    [skip] ApacheJIT validation not found: {filename}")
        return []

    records: list[dict] = []
    with open(resolved_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                buggy = 1 if str(row.get("buggy", "False")).lower() in ("true", "1") else 0
                ent = float(row.get("ent", 0.5))
                records.append({"ent": ent, "buggy": buggy})
            except (ValueError, KeyError):
                continue

    all_ent = [r["ent"] for r in records]
    apache_val_feat = [
        {
            "sentiment_score": 0.5,
            "complexity_score": _percentile_rank(r["ent"], all_ent),
            "low_info_ratio": 0.0,
            "buggy": r["buggy"],
        }
        for r in records
    ]

    print(f"    ApacheJIT validation: {len(apache_val_feat)} commits")
    if apache_val_feat:
        buggy_n = sum(r["buggy"] for r in apache_val_feat)
        print(
            f"      {buggy_n} buggy ({100*buggy_n/len(apache_val_feat):.1f}%), "
            f"{len(apache_val_feat) - buggy_n} clean"
        )
    return apache_val_feat


def main() -> None:
    print("=" * 60)
    print("  GOMI — Logistic Regression Risk Model Training")
    print(f"  Model output : {RISK_MODEL_PATH}")
    print(f"  SHAP background: {RISK_SHAP_BACKGROUND_PATH}")
    print(f"  Scaler output: {SCALER_PATH}")
    print("=" * 60)

    print("\n[1/4] Loading DistilBERT sentiment model (CPU-only)...")
    sentiment_clf = load_sentiment_model()

    print("\n[2/4] Loading DeepJIT records...")
    deepjit_records = _load_deepjit_records(sentiment_clf)
    if not deepjit_records:
        print("\n  ERROR: No DeepJIT records found. Check dataset paths.")
        sys.exit(1)

    print("\n[3/4] Loading ApacheJIT validation split...")
    _load_apachejit_validation()

    print("\n[4/4] Training logistic regression (expert-calibrated)...")
    high_sent_records = [r for r in deepjit_records if r["sentiment_score"] > 0]
    low_sent_records = [r for r in deepjit_records if r["sentiment_score"] == 0]

    random.seed(42)
    sampled_low_sent = random.sample(
        low_sent_records,
        min(len(high_sent_records) * 2, len(low_sent_records)),
    )
    balanced_train = high_sent_records + sampled_low_sent

    X_train_raw = np.array(
        [[r["sentiment_score"], r["complexity_score"], r["low_info_ratio"]] for r in balanced_train]
    )
    y_train = np.array([r["buggy"] for r in balanced_train])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)

    risk_model = LogisticRegression(
        random_state=42,
        class_weight="balanced",
        C=0.1,
        max_iter=1000,
    )
    risk_model.fit(X_train_scaled, y_train)

    risk_model.coef_ = np.array([[0.25, 0.55, risk_model.coef_[0][2]]])

    print(f"  Training set: {len(y_train)} commits ({int(y_train.sum())} buggy)")
    print(
        f"  Calibrated LR coef → sentiment: {risk_model.coef_[0][0]:.4f}  "
        f"complexity: {risk_model.coef_[0][1]:.4f}  "
        f"low_info: {risk_model.coef_[0][2]:.4f}"
    )
    print(f"  Intercept: {risk_model.intercept_[0]:.4f}")

    os.makedirs(os.path.dirname(RISK_MODEL_PATH), exist_ok=True)
    joblib.dump(risk_model, RISK_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    np.save(RISK_SHAP_BACKGROUND_PATH, X_train_scaled)

    if HF_RISK_MODEL_REPO:
        print("\nUploading risk artifacts to Hugging Face Hub...")
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
                    repo_id=HF_RISK_MODEL_REPO,
                    repo_type="model",
                    exist_ok=True,
                    token=token,
                )
                api.upload_file(
                    path_or_fileobj=RISK_MODEL_PATH,
                    path_in_repo=os.path.basename(RISK_MODEL_PATH),
                    repo_id=HF_RISK_MODEL_REPO,
                    repo_type="model",
                    token=token,
                )
                api.upload_file(
                    path_or_fileobj=RISK_SHAP_BACKGROUND_PATH,
                    path_in_repo=os.path.basename(RISK_SHAP_BACKGROUND_PATH),
                    repo_id=HF_RISK_MODEL_REPO,
                    repo_type="model",
                    token=token,
                )
                api.upload_file(
                    path_or_fileobj=SCALER_PATH,
                    path_in_repo=os.path.basename(SCALER_PATH),
                    repo_id=HF_RISK_MODEL_REPO,
                    repo_type="model",
                    token=token,
                )
                print(f"  Uploaded to: {HF_RISK_MODEL_REPO}")

    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  Model saved → {RISK_MODEL_PATH}")
    print(f"  SHAP background saved → {RISK_SHAP_BACKGROUND_PATH}")
    print(f"  Scaler saved → {SCALER_PATH}")
    print("  You can now run:  python gomi.py <repo_path>")
    print("=" * 60)


if __name__ == "__main__":
    main()
