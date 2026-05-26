"""Download a QA-bench dataset from the Hugging Face Hub into a run dir.

The HF repo is expected to mirror the run-dir layout:

    qa_bank.json
    docs/*.pdf

We pull both with `huggingface_hub.snapshot_download`, restricting downloads
to those patterns. Skip-if-present is automatic (HF cache is content-hashed);
pass `--force` to re-download.

When `limit N` is set, pulls just the bank + the PDFs referenced by the
**first N bank items** — the same selection rule used by `parse --limit` and
`score --limit`, so a smoke run works end-to-end on a self-consistent slice.
The bank file itself is still complete (all items); only the docs/ subset is
limited.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from realdoc_bench.evaluate.runs import RunLayout


def download_dataset(layout: RunLayout, *,
                     repo_id: str,
                     revision: str | None = None,
                     force: bool = False,
                     limit: int | None = None) -> dict[str, int]:
    """Fetch `qa_bank.json` + `docs/` from a HF dataset repo into `<run-dir>/`.

    Returns `{"docs": N, "bank": 1|0}` so the CLI can print a sane summary.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required for `evaluate download`; "
            "`uv sync` should install it (declared in pyproject.toml)"
        ) from e

    if force and layout.bank_path.exists():
        layout.bank_path.unlink()
    if force and layout.docs_dir.exists():
        shutil.rmtree(layout.docs_dir)

    layout.root.mkdir(parents=True, exist_ok=True)

    if limit is not None and limit > 0:
        # Smoke-test mode: pull the bank first to learn which docs to pull.
        # Selection rule = first N bank items (same as score --limit).
        from realdoc_bench.evaluate.score import docs_for_items
        hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename="qa_bank.json",
            revision=revision,
            local_dir=str(layout.root), local_dir_use_symlinks=False,
        )
        bank = json.loads(layout.bank_path.read_text())
        refs = docs_for_items(bank["items"][:limit])
        patterns = ["qa_bank.json"] + [f"docs/{stem}.pdf" for stem in refs]
    else:
        patterns = ["qa_bank.json", "docs/**"]

    snap = Path(snapshot_download(
        repo_id=repo_id, repo_type="dataset",
        revision=revision,
        allow_patterns=patterns,
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

    n_docs = sum(1 for _ in layout.docs_dir.glob("*.pdf")) \
        if layout.docs_dir.exists() else 0
    return {"docs": n_docs, "bank": int(layout.bank_path.exists())}
