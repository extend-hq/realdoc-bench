"""Download a QA-bench dataset from the Hugging Face Hub into a run dir.

The HF repo is expected to mirror the run-dir layout:

    bank.json
    pdfs/*.pdf

We pull both with `huggingface_hub.snapshot_download`, restricting downloads
to those patterns. Skip-if-present is automatic (HF cache is content-hashed);
pass `--force` to re-download.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from realdoc_bench.evaluate.runs import RunLayout


def download_dataset(layout: RunLayout, *,
                     repo_id: str,
                     revision: str | None = None,
                     hf_token: str | None = None,
                     force: bool = False) -> dict[str, int]:
    """Fetch `bank.json` + `pdfs/` from a HF dataset repo into `<run-dir>/`.

    Returns `{"pdfs": N, "bank": 1|0}` so the CLI can print a sane summary.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required for `evaluate download`; "
            "`uv sync` should install it (declared in pyproject.toml)"
        ) from e

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN")
    if force and layout.bank_path.exists():
        layout.bank_path.unlink()
    if force and layout.pdfs_dir.exists():
        shutil.rmtree(layout.pdfs_dir)

    layout.root.mkdir(parents=True, exist_ok=True)
    snap = Path(snapshot_download(
        repo_id=repo_id, repo_type="dataset",
        revision=revision, token=token,
        allow_patterns=["bank.json", "pdfs/**"],
        local_dir=str(layout.root),
        local_dir_use_symlinks=False,
    ))

    # snapshot_download with local_dir places files under `local_dir`, so the
    # following just counts what landed. If layout.root != snap (HF default
    # cache mode), copy across — but with local_dir set, they're the same.
    if snap != layout.root:
        # cache-mode fallback: mirror contents into the run dir
        for src in snap.rglob("*"):
            if src.is_file():
                rel = src.relative_to(snap)
                dst = layout.root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    n_pdfs = sum(1 for _ in layout.pdfs_dir.glob("*.pdf")) \
        if layout.pdfs_dir.exists() else 0
    return {"pdfs": n_pdfs, "bank": int(layout.bank_path.exists())}
