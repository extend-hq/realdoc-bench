"""HuggingFace Hub loader for realdoc-bench datasets.

Datasets sit under the ``Extend-AI`` org and are public — no authentication needed.
"""

from __future__ import annotations

import os
from pathlib import Path

from realdoc_bench.shared.io.cache import default_cache_root, ensure_dir


def fetch_dataset(
    name: str,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download a HF dataset snapshot and return the local path.

    Args:
        name: ``"<org>/<dataset>"``, e.g. ``"Extend-AI/RealDocBench-Layout"``.
        revision: optional git revision (commit SHA, branch, tag). Pin for reproducibility.
        cache_dir: local cache root. Defaults to ``~/.cache/realdoc-bench/hf``.

    Local override: set ``REALDOC_BENCH_DATASET_<NAME>`` (with ``-`` and ``/`` replaced
    by ``_``, upper-cased) to point at a pre-staged local snapshot dir. Useful for
    dev before the HF dataset is published.
    """
    env_key = "REALDOC_BENCH_DATASET_" + name.replace("/", "_").replace("-", "_").upper()
    local = os.environ.get(env_key)
    if local:
        path = Path(local).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{env_key} points at non-existent path: {path}")
        return path

    from huggingface_hub import snapshot_download

    root = ensure_dir(cache_dir or (default_cache_root() / "hf"))
    return Path(
        snapshot_download(
            repo_id=name,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(root),
        )
    )
