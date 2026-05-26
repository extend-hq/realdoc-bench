"""Download a layout dataset from the Hugging Face Hub.

The HF repo is expected to mirror what :mod:`realdoc_bench.layout.data.loader`
consumes:

    manifest.csv
    images/<page_id>.{png,jpg,jpeg}
    annotations/<page_id>.json

Without ``out_dir`` we just prewarm the HF cache (so ``layout eval`` doesn't
pay the network cost mid-run). With ``out_dir`` we materialize a filtered
snapshot into that directory — point ``REALDOC_BENCH_DATASET_<NAME>`` at it
to use it without touching the cache.
"""

from __future__ import annotations

import csv
import shutil
from collections.abc import Iterable
from pathlib import Path


def download_dataset(*,
                     repo_id: str,
                     out_dir: Path | None = None,
                     revision: str | None = None,
                     domains: Iterable[str] | None = None,
                     limit: int | None = None,
                     force: bool = False) -> dict[str, int | str]:
    """Fetch the layout dataset snapshot.

    Returns ``{"images": N, "annotations": M, "path": <str>}`` for a sane
    CLI summary.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required for `layout download`; "
            "`uv sync` should install it (declared in pyproject.toml)"
        ) from e

    filtered = domains is not None or limit is not None

    if out_dir is not None:
        if force and out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    if not filtered:
        # Full snapshot — let HF handle skip-if-present.
        dest = Path(snapshot_download(
            repo_id=repo_id, repo_type="dataset", revision=revision,
            **({"local_dir": str(out_dir)} if out_dir is not None else {}),
        ))
    else:
        # Pull manifest first to learn which page ids to fetch.
        keep_domains = set(domains) if domains is not None else None
        manifest_snap = Path(snapshot_download(
            repo_id=repo_id, repo_type="dataset", revision=revision,
            allow_patterns=["manifest.csv"],
        ))
        with (manifest_snap / "manifest.csv").open() as f:
            rows = list(csv.DictReader(f))
        if keep_domains is not None:
            rows = [r for r in rows if r["domain"] in keep_domains]
        if limit is not None:
            rows = rows[:limit]
        page_ids = [r["pageId"] for r in rows]
        patterns = ["manifest.csv"] + [
            f"images/{pid}.*" for pid in page_ids
        ] + [
            f"annotations/{pid}.json" for pid in page_ids
        ]
        dest = Path(snapshot_download(
            repo_id=repo_id, repo_type="dataset", revision=revision,
            allow_patterns=patterns,
            **({"local_dir": str(out_dir)} if out_dir is not None else {}),
        ))

    n_images = sum(1 for _ in (dest / "images").glob("*")) \
        if (dest / "images").exists() else 0
    n_annotations = sum(1 for _ in (dest / "annotations").glob("*.json")) \
        if (dest / "annotations").exists() else 0
    return {"images": n_images, "annotations": n_annotations, "path": str(dest)}
