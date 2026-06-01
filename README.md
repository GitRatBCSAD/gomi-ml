# Gomi Prototype

> **Prototype only — not production-ready**  
> This repo is from an initial vibecoded commit: training scripts, inference (`gomi.py`), and docs are meant for **experimentation and thesis prototyping**, not as a polished release. Pipelines, dataset wiring, train/runtime feature alignment, and validation still need refinement. Treat outputs as exploratory until we harden the stack.

## One-time setup

```sh
# Install dependences
uv sync
```

Train and save the models once before analyzing repos:

```sh
# Layer 1: DistilBERT on OpenReview 2025
uv run python scripts/train_sentiment.py

# Layer 3: Logistic Regression on DeepJIT + ApacheJIT
uv run python scripts/train_risk_model.py
```

Re-run `train_risk_model.py` only if you change the sentiment model or JIT training datasets.

## Usage

```sh
uv run python gomi.py <git-project>
```

Example:

```sh
uv run python gomi.py ../my-project
```

### Environment variables
| Variable | Purpose |
| --- | --- |
| `GOMI_SENTIMENT_MODEL_REPO` | HF model repo for the sentiment model (e.g. `org/gomi-sentiment`). |
| `GOMI_SENTIMENT_MODEL_REVISION` | Optional tag/commit for the sentiment model (e.g. `v1.0.0`). |
| `GOMI_RISK_MODEL_REPO` | HF model repo for risk artifacts (e.g. `org/gomi-risk`). |
| `GOMI_RISK_MODEL_REVISION` | Optional tag/commit for the risk model (e.g. `v1.0.0`). |
| `GOMI_DATASET_REPO` | HF dataset repo containing OpenReview/DeepJIT/ApacheJIT files (defaults to `GitRatBCSAD/gomi-datasets`). |
| `GOMI_DATASET_REVISION` | Optional tag/commit for datasets. |
| `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` | Access token for private repos. |

Training datasets are **always downloaded from Hugging Face** (`GOMI_DATASET_REPO`, default `GitRatBCSAD/gomi-datasets`). No local dataset copies are used.

Place env vars in a `.env` at the repo root — scripts auto-load it via `python-dotenv`. Clear `GOMI_DATASET_REVISION` until that tag exists on the Hub.

### Inference workflow (pull on startup)
- `gomi.py` pulls JIT/OpenReview datasets from the HF dataset repo; sentiment and risk **models** load from `scripts/datasets/` after training, or from HF model repos when `GOMI_*_MODEL_REPO` is set.

### Stable release tagging

For tagging a release, run this script:

```sh
 uv run python scripts/tag_release.py v2.0.0
```

