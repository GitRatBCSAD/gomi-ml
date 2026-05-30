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

Make sure to set the environment variables as our models and datasets are being loaded from our huggingface repo.

If the `GOMI_*_REPO` variables are not set, the scripts fall back to local files under `scripts/datasets/` or `datasets/`, including `openreview/` and `jit/` subfolders.
You can place these variables in a `.env` at the repo root — the scripts auto-load it via `python-dotenv`.

### Inference workflow (pull on startup)
- `gomi.py` loads the sentiment and risk models from local paths when available; otherwise it pulls from HF Hub.

