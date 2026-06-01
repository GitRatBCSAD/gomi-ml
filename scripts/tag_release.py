# tag_release.py — Tag trained models on HuggingFace Hub
#
# Run this AFTER validating model results on a real repo.
# Tags are immutable — they freeze the exact model version on HF.
#
# Usage:
#   uv run python scripts/tag_release.py v2.0.0

import os
import sys
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN          = os.getenv("HF_TOKEN")
HF_SENTIMENT_REPO = os.getenv("GOMI_SENTIMENT_MODEL_REPO", "GitRatBCSAD/gomi-sentiment")
HF_RISK_REPO      = os.getenv("GOMI_RISK_MODEL_REPO",      "GitRatBCSAD/gomi-risk")

if len(sys.argv) < 2:
    print("Usage: uv run python scripts/tag_release.py <tag>")
    print("  e.g. uv run python scripts/tag_release.py v2.0.0")
    sys.exit(1)

tag = sys.argv[1]

if not tag.startswith("v"):
    print(f"WARNING: tag '{tag}' doesn't start with 'v' — are you sure? (e.g. v2.0.0)")
    confirm = input("Continue? [y/N] ").strip().lower()
    if confirm != "y":
        sys.exit(0)

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set in .env")
    sys.exit(1)

from huggingface_hub import HfApi
api = HfApi()

print(f"Tagging models as {tag}...")

for repo, repo_type in [(HF_SENTIMENT_REPO, "model"), (HF_RISK_REPO, "model")]:
    try:
        api.create_tag(repo, tag=tag, repo_type=repo_type, token=HF_TOKEN)
        print(f"  ✓  {repo}@{tag}")
    except Exception as e:
        print(f"  ✗  {repo}: {e}")

print(f"""
Done. Update your .env with:
  GOMI_SENTIMENT_MODEL_REVISION={tag}
  GOMI_RISK_MODEL_REVISION={tag}
""")
