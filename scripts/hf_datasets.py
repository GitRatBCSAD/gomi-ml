"""Download Gomi dataset files from Hugging Face Hub only."""

from __future__ import annotations


def _hub_paths_for_filename(filename: str) -> list[str]:
    paths: list[str] = []
    if filename in {
        "cleaned20k.csv",
        "openreview_labeled_2k.csv",
        "openreview_labeled_5k_auto.csv",
        "senticr_labeled.csv",
        "StackOverflow.csv",
        "github.csv",
    } or filename.startswith("openreview_") or filename.startswith("kaggle_100k_labeled_"):
        paths.append(f"sentiment/{filename}")
    elif filename.startswith("apachejit") or "apache" in filename:
        paths.append(f"jit/apachejit/{filename}")
    elif (
        filename.endswith("_test_raw.pkl")
        or filename.endswith("_k_feature.csv")
        or "deepjit" in filename
        or filename == "qt_dict.pkl"
        or filename == "commit_files_deepjit.csv"
    ):
        paths.append(f"jit/deepjit/{filename}")
    elif filename.startswith("unlabeled_") or filename in {"oneline.csv", "full.csv"}:
        paths.append(f"unlabeled/{filename}")
    paths.append(filename)
    seen: set[str] = set()
    ordered: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def hub_download_dataset_file(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> str | None:
    import os
    
    # 1. Check locally first for people training locally
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, "datasets")
    
    paths = _hub_paths_for_filename(filename)
    for path in paths:
        local_path = os.path.join(datasets_dir, path)
        if os.path.exists(local_path):
            return local_path
            
    # 2. Fall back to Hugging Face
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RevisionNotFoundError

    def _try_download(rev: str | None) -> str | None:
        revision_error: RevisionNotFoundError | None = None
        for path in paths:
            try:
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=path,
                    repo_type="dataset",
                    revision=rev,
                    token=token,
                )
            except RevisionNotFoundError as e:
                revision_error = e
                break
            except EntryNotFoundError:
                continue
        if revision_error:
            raise revision_error
        return None

    if revision:
        try:
            found = _try_download(revision)
            if found:
                return found
        except RevisionNotFoundError:
            print(
                f"  WARNING: dataset revision '{revision}' not found on {repo_id}; "
                "using default branch instead."
            )

    return _try_download(None)
