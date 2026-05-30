"""Download Gomi dataset files from Hugging Face Hub only."""

from __future__ import annotations


def _hub_paths_for_filename(filename: str) -> list[str]:
    paths: list[str] = []
    if filename in {
        "cleaned20k.csv",
        "openreview_labeled_2k.csv",
        "openreview_labeled_5k_auto.csv",
    } or filename.startswith("openreview_"):
        paths.append(f"openreview/{filename}")
    if (
        filename.endswith("_test_raw.pkl")
        or filename.endswith("_k_feature.csv")
        or filename.startswith("apachejit")
    ):
        paths.append(f"jit/{filename}")
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
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RevisionNotFoundError

    paths = _hub_paths_for_filename(filename)

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
