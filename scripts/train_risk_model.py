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
#   - DeepJIT .pkl files and/or apachejit_commits.csv under scripts/datasets/
#     (or set GOMI_DATASET_REPO to pull from Hugging Face)
#
# Usage (from repo root):
#   uv run python scripts/train_risk_model.py

import os
import sys

import joblib
import numpy as np
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gomi import (  # noqa: E402
    RISK_MODEL_PATH,
    RISK_SHAP_BACKGROUND_PATH,
    load_sentiment_model,
    train_risk_model,
)

load_dotenv()
HF_RISK_MODEL_REPO = os.getenv("GOMI_RISK_MODEL_REPO")


def _get_hf_token():
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def main() -> None:
    print("=" * 60)
    print("  GOMI — Logistic Regression Risk Model Training")
    print(f"  Model output : {RISK_MODEL_PATH}")
    print(f"  SHAP background: {RISK_SHAP_BACKGROUND_PATH}")
    print("=" * 60)

    print("\n[1/2] Loading DistilBERT sentiment model...")
    sentiment_clf = load_sentiment_model()

    print("\n[2/2] Training risk model on DeepJIT + ApacheJIT...")
    risk_model, X_train = train_risk_model(sentiment_clf)

    os.makedirs(os.path.dirname(RISK_MODEL_PATH), exist_ok=True)
    joblib.dump(risk_model, RISK_MODEL_PATH)
    np.save(RISK_SHAP_BACKGROUND_PATH, X_train)

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
                print(f"  Uploaded to: {HF_RISK_MODEL_REPO}")

    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  Model saved → {RISK_MODEL_PATH}")
    print(f"  SHAP background saved → {RISK_SHAP_BACKGROUND_PATH}")
    print("  You can now run:  python gomi.py <repo_path>")
    print("=" * 60)


if __name__ == "__main__":
    main()
