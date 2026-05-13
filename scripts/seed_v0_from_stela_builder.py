"""Seed datasets/extraction/v0/ from the local Stela-Builder pilot runs.

For each pilot doc that has (a) an items/<doc_id>.json schema, (b) a
truncated/<doc_id>.pdf source, and (c) a pilot/runs/<doc_id>/candidates.json,
this script:

1. Copies the schema into datasets/extraction/v0/schemas/<doc_id>.json
2. Copies the PDF into datasets/extraction/v0/sources/<doc_id>.pdf (gitignored)
3. Picks the `extend_performance` candidate's value as ground truth and writes
   datasets/extraction/v0/ground_truth/<doc_id>.json
4. Appends a manifest entry to datasets/extraction/v0/manifest.yaml

Re-running is idempotent: existing entries are merged on doc_id.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
V0_ROOT = REPO_ROOT / "datasets" / "extraction" / "v0"
STELA_BUILDER = Path("/Users/ameyajoshi/Projects/Stela-Builder")

PREFERRED_EXTRACTOR = "extend_performance"
FALLBACK_EXTRACTORS = ("claude_sonnet_4_6", "gpt_5_4")


def _pick_value(candidates_path: Path) -> tuple[str, dict] | None:
    with candidates_path.open() as f:
        candidates = json.load(f)
    by_extractor = {c["extractor"]: c for c in candidates}
    for name in (PREFERRED_EXTRACTOR, *FALLBACK_EXTRACTORS):
        entry = by_extractor.get(name)
        if entry is not None and entry.get("status") == "PROCESSED" and entry.get("value"):
            return name, entry["value"]
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dirs() -> None:
    (V0_ROOT / "schemas").mkdir(parents=True, exist_ok=True)
    (V0_ROOT / "sources").mkdir(parents=True, exist_ok=True)
    (V0_ROOT / "ground_truth").mkdir(parents=True, exist_ok=True)


def _existing_manifest() -> dict:
    path = V0_ROOT / "manifest.yaml"
    if not path.exists():
        return {"version": 1, "set_id": "v0", "defaults": {"schema_dialect": "extend-draft07"}, "docs": []}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    _ensure_dirs()
    schemas_dir = STELA_BUILDER / "stela_builder" / "schemas" / "items"
    sources_dir = STELA_BUILDER / "pilot" / "truncated"
    runs_dir = STELA_BUILDER / "pilot" / "runs"
    if not runs_dir.exists():
        print(f"no pilot runs dir at {runs_dir}")
        return 1

    manifest = _existing_manifest()
    by_doc = {d["doc_id"]: d for d in manifest.get("docs", [])}

    added = 0
    for doc_dir in sorted(runs_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        schema_src = schemas_dir / f"{doc_id}.json"
        pdf_src = sources_dir / f"{doc_id}.pdf"
        candidates_path = doc_dir / "candidates.json"
        if not schema_src.exists() or not pdf_src.exists() or not candidates_path.exists():
            print(f"skip {doc_id}: missing schema/pdf/candidates")
            continue
        picked = _pick_value(candidates_path)
        if picked is None:
            print(f"skip {doc_id}: no usable candidate")
            continue
        extractor, value = picked

        schema_dst = V0_ROOT / "schemas" / f"{doc_id}.json"
        pdf_dst = V0_ROOT / "sources" / f"{doc_id}.pdf"
        gt_dst = V0_ROOT / "ground_truth" / f"{doc_id}.json"

        shutil.copyfile(schema_src, schema_dst)
        shutil.copyfile(pdf_src, pdf_dst)
        gt_dst.write_text(json.dumps(value, indent=2, default=str))

        schema = json.loads(schema_dst.read_text())
        domain = schema.get("x-domain")
        doc_type = schema.get("x-doc-type")
        entry = {
            "doc_id": doc_id,
            "source_path": f"sources/{doc_id}.pdf",
            "source_sha256": _sha256(pdf_dst),
            "schema_path": f"schemas/{doc_id}.json",
            "ground_truth_path": f"ground_truth/{doc_id}.json",
            "metadata": {
                "domain": domain,
                "doc_type": doc_type,
                "origin": "stela-builder pilot",
                "license": "internal-only",
                "ground_truth_source": extractor,
            },
            "splits": ["pilot"],
            "scoring_rules": [],
        }
        by_doc[doc_id] = entry
        added += 1

    manifest["docs"] = sorted(by_doc.values(), key=lambda d: d["doc_id"])
    with (V0_ROOT / "manifest.yaml").open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    print(f"updated manifest with {added} doc(s); total {len(manifest['docs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
