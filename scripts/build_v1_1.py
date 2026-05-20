#!/usr/bin/env python3
"""Build datasets/extraction/v1.1/ as a domain-grouped, curated snapshot.

For every doc type configured in summarize_v1_samples.DOMAINS, copy (via
hardlinks — same filesystem, zero disk cost) all sources/*.pdf and
synthetic/*.pdf into:

  datasets/extraction/v1.1/<domain>/<type>/sources/
  datasets/extraction/v1.1/<domain>/<type>/synthetic/

Also copies the _sources_manifest.csv per type and writes an INDEX.md at
the v1.1 root with per-domain totals.

Re-running is safe: hardlinks are recreated as needed; previously-staged
files are left in place if unchanged.
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
from pathlib import Path

# Import the DOMAINS configuration from the existing summarizer so this stays
# in sync.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_v1_samples import DOMAINS  # type: ignore

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets" / "extraction"
DEST_ROOT = DATASETS / "v1.1"

# Sources whose is_filled column is in this set are excluded from v1.1.
# "blank" = template / unfilled form. "guide"/"unknown" pass through.
EXCLUDE_IS_FILLED = {"blank"}

# Override the EXCLUDE for these specific types — blanks ARE included.
# Use for tax/regulatory templates where filled examples are PII and we want
# the templates anyway as form-field benchmarks.
INCLUDE_BLANKS_FOR_TYPES = {
    "v1_irs_specialty",
    "v1_intl_tax_ca", "v1_intl_tax_uk", "v1_intl_tax_au", "v1_intl_tax_in",
    "v1_intl_tax_de", "v1_intl_tax_nz", "v1_intl_tax_sg", "v1_intl_tax_za",
    "v1_intl_tax_jp", "v1_intl_tax_br",
    "v1_ferpa_release",
    "v1_naic_annual_statement",
    "v1_ferc_form1", "v1_ferc_form2",
    # Finance bucket blanks (user-requested addition):
    "v1_acord_125_126", "v1_acord_4", "v1_acord_25",
    "v1_fec_form3", "v1_fec_form5", "v1_fec_form6", "v1_fec_form7",
    "v1_form_5500",
}

# Global page cap — any PDF longer than this is truncated to the first N pages
# (real copy, not hardlink). Keeps parse cost / time bounded.
MAX_PAGES_PER_PDF = 200

# Per-type cap on number of source PDFs, applied per domain.
# Financial domain has 23 types with up to 62 PDFs each (FEC); cap to 2 per type
# for cost discipline. Other domains: no cap (None).
PER_TYPE_CAP_BY_DOMAIN = {
    "Financial (incl. shipping/cargo)": 4,  # bumped from 2 for the new dense-tables variety
}

# Override per-type cap (None = no cap). Used for high-value new types where
# we want full diversity (e.g. all 10 IRS specialty forms, all SEC 13Fs).
PER_TYPE_CAP_OVERRIDE: dict[str, int | None] = {
    "v1_irs_specialty": None,  # keep all 10 IRS specialty forms
    "v1_intl_tax_ca": None,
    "v1_intl_tax_uk": None,
    "v1_intl_tax_in": None,
    "v1_intl_tax_de": None,
    "v1_intl_tax_au": None,
    "v1_intl_tax_nz": None,
    "v1_intl_tax_sg": None,
    "v1_intl_tax_za": None,
    "v1_intl_tax_jp": None,
    "v1_intl_tax_br": None,
    "v1_sec_13f": 6,           # cap at 6 of 8 to leave breathing room
    "v1_sec_def14a": 6,
    "v1_sec_form_3_4_5": 8,
    # ACORD insurance blanks — keep all (small set: 7-8 each)
    "v1_acord_125_126": None,
    "v1_acord_4": None,
    "v1_acord_25": None,
    # FEC — keep 10 (8 filled + room for blanks), blanks included
    "v1_fec_form3": 10,
    "v1_fec_form5": None,  # only 6
    "v1_fec_form6": None,  # only 6
    "v1_form_5500": None,  # only 2 (1 blank + 1 guide)
}

# Ranking when more sources exist than the cap allows: prefer is_filled='filled'
# over 'unknown' over 'guide'; within the same tier, prefer smaller page count
# (focused filings beat omnibus dumps).
FILLED_PRIORITY = {"filled": 0, "unknown": 1, "": 1, "guide": 2}


def safe_dirname(name: str) -> str:
    """File-system-safe folder name from a human-readable domain label."""
    out = name.replace("&", "and").replace("/", "-")
    out = "_".join(out.split())
    return out.strip("_")


def _pdf_page_count(pdf: Path) -> int:
    try:
        import fitz
        with fitz.open(pdf) as d:
            return d.page_count
    except Exception:
        return 0


def _truncate_pdf(src: Path, dest: Path, max_pages: int) -> bool:
    """Copy first max_pages pages of src into dest. Returns True if a truncated
    copy was actually written."""
    try:
        import fitz
        with fitz.open(src) as src_doc:
            if src_doc.page_count <= max_pages:
                return False
            new_doc = fitz.open()
            new_doc.insert_pdf(src_doc, from_page=0, to_page=max_pages - 1)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest.unlink()
            new_doc.save(dest, garbage=4, deflate=True)
            new_doc.close()
            return True
    except Exception as e:
        print(f"  ! truncate fail {src.name}: {e}")
        return False


def hardlink_or_copy(src: Path, dest: Path, max_pages: int | None = None) -> str:
    """Hardlink src→dest if same fs; fall back to copy. If src exceeds max_pages,
    write a truncated copy instead. Returns 'link'/'copy'/'trunc'/'skip'."""
    # Truncation needed?
    if max_pages and _pdf_page_count(src) > max_pages:
        if _truncate_pdf(src, dest, max_pages):
            return "trunc"
        # fall through to normal copy on failure
    if dest.exists():
        try:
            if dest.stat().st_ino == src.stat().st_ino:
                return "skip"
        except OSError:
            pass
        dest.unlink()
    try:
        os.link(src, dest)
        return "link"
    except OSError:
        shutil.copy2(src, dest)
        return "copy"


def count_pdfs(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.glob("*.pdf"))


def load_excluded_filenames(manifest_path: Path, type_name: str = "") -> set[str]:
    """Return source filenames whose is_filled value is in EXCLUDE_IS_FILLED.
    For types listed in INCLUDE_BLANKS_FOR_TYPES, return empty set (no exclusion)."""
    if type_name in INCLUDE_BLANKS_FOR_TYPES:
        return set()
    excluded: set[str] = set()
    if not manifest_path.exists():
        return excluded
    with manifest_path.open() as f:
        for row in csv.DictReader(f):
            tag = (row.get("is_filled") or "").strip().lower()
            if tag in EXCLUDE_IS_FILLED:
                fname = row.get("local_filename") or row.get("filename")
                if fname:
                    excluded.add(fname.strip())
    return excluded


def load_filled_tags(manifest_path: Path) -> dict[str, str]:
    """Return {filename → is_filled tag} for ranking purposes."""
    out: dict[str, str] = {}
    if not manifest_path.exists():
        return out
    with manifest_path.open() as f:
        for row in csv.DictReader(f):
            tag = (row.get("is_filled") or "").strip().lower()
            fname = row.get("local_filename") or row.get("filename")
            if fname:
                out[fname.strip()] = tag
    return out


def apply_per_type_cap(pdfs: list[Path], cap: int, filled_tags: dict[str, str]) -> list[Path]:
    """Keep top `cap` PDFs sorted by:
        1. filled priority (filled > unknown > guide)
        2. LARGER page count first — more substantive content
           (the global 200pp truncation cap bounds the cost of "too big")
        3. filename
    """
    if len(pdfs) <= cap:
        return pdfs
    def _rank(p: Path):
        tag = filled_tags.get(p.name, "")
        pri = FILLED_PRIORITY.get(tag, 1)
        return (pri, -_pdf_page_count(p), p.name)
    return sorted(pdfs, key=_rank)[:cap]


def stage_type(folder_key: str, dest_type_dir: Path, per_type_cap: int | None = None) -> dict:
    """Stage one type into dest_type_dir/{sources,synthetic}/. Returns stats.

    Sources marked is_filled='blank' (in _sources_manifest.csv) are excluded.
    If per_type_cap is set, keep only the top-N ranked sources.
    PDFs exceeding MAX_PAGES_PER_PDF are truncated on copy.
    All synthetic/ files are kept — those are filled by construction.
    """
    sources_files: list[Path] = []
    synth_files: list[Path] = []
    manifest_src: Path | None = None
    excluded_count = 0
    capped_count = 0

    if folder_key.startswith("v0/"):
        parent, doc_id = folder_key.split("/", 1)
        pdf = DATASETS / parent / "sources" / f"{doc_id}.pdf"
        if pdf.exists():
            sources_files.append(pdf)
    elif folder_key == "v0_tax":
        src_dir = DATASETS / "v0_tax" / "sources"
        if src_dir.exists():
            sources_files = sorted(src_dir.glob("*.pdf"))
    else:
        src_dir = DATASETS / folder_key
        manifest_candidate = src_dir / "_sources_manifest.csv"
        # Type name derived from folder_key (works for both v0/ and v1_ variants)
        type_name = folder_key.split("/")[-1] if folder_key.startswith("v0/") else folder_key
        excluded = load_excluded_filenames(manifest_candidate, type_name) if manifest_candidate.exists() else set()
        filled_tags = load_filled_tags(manifest_candidate) if manifest_candidate.exists() else {}
        if (src_dir / "sources").exists():
            all_src = sorted((src_dir / "sources").glob("*.pdf"))
            sources_files = [p for p in all_src if p.name not in excluded]
            excluded_count = len(all_src) - len(sources_files)
            # Apply per-type-override first (lets us bypass or tighten the domain default)
            effective_cap = per_type_cap
            if type_name in PER_TYPE_CAP_OVERRIDE:
                effective_cap = PER_TYPE_CAP_OVERRIDE[type_name]  # may be None (uncapped)
            if effective_cap is not None and len(sources_files) > effective_cap:
                before = len(sources_files)
                sources_files = apply_per_type_cap(sources_files, effective_cap, filled_tags)
                capped_count = before - len(sources_files)
        if (src_dir / "synthetic").exists():
            synth_files = sorted((src_dir / "synthetic").glob("*.pdf"))
        if manifest_candidate.exists():
            manifest_src = manifest_candidate

    # Stage
    (dest_type_dir / "sources").mkdir(parents=True, exist_ok=True)
    if synth_files:
        (dest_type_dir / "synthetic").mkdir(parents=True, exist_ok=True)

    counts = {"sources_linked": 0, "synth_linked": 0, "copied": 0,
              "skipped": 0, "truncated": 0}
    for src in sources_files:
        dst = dest_type_dir / "sources" / src.name
        action = hardlink_or_copy(src, dst, max_pages=MAX_PAGES_PER_PDF)
        if action == "link":
            counts["sources_linked"] += 1
        elif action == "copy":
            counts["copied"] += 1
        elif action == "trunc":
            counts["truncated"] += 1
        else:
            counts["skipped"] += 1
    for src in synth_files:
        dst = dest_type_dir / "synthetic" / src.name
        action = hardlink_or_copy(src, dst, max_pages=MAX_PAGES_PER_PDF)
        if action == "link":
            counts["synth_linked"] += 1
        elif action == "copy":
            counts["copied"] += 1
        elif action == "trunc":
            counts["truncated"] += 1
        else:
            counts["skipped"] += 1
    if manifest_src:
        shutil.copy2(manifest_src, dest_type_dir / "_sources_manifest.csv")

    counts["sources_count"] = len(sources_files)
    counts["synth_count"] = len(synth_files)
    counts["excluded_blank"] = excluded_count
    counts["capped"] = capped_count
    return counts


def main() -> int:
    if DEST_ROOT.exists() and any(DEST_ROOT.iterdir()):
        # Don't wipe — just re-stage on top. User can rm -rf if they want a clean slate.
        print(f"NOTE: {DEST_ROOT.relative_to(REPO)} already exists; re-staging on top "
              "(safe — hardlinks are idempotent).")
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    # Track per-domain + grand totals
    domain_rows: list[tuple[str, list[dict]]] = []
    grand = {"sources": 0, "synth": 0, "linked": 0, "copied": 0, "excluded_blank": 0}

    for domain_name, entries in DOMAINS:
        d_dir = DEST_ROOT / safe_dirname(domain_name)
        cap = PER_TYPE_CAP_BY_DOMAIN.get(domain_name)
        rows: list[dict] = []
        for folder_key, label, _hint in entries:
            type_name = folder_key.replace("v0/", "v0_").replace("/", "_")
            dest_type_dir = d_dir / type_name
            stats = stage_type(folder_key, dest_type_dir, per_type_cap=cap)
            if stats["sources_count"] == 0 and stats["synth_count"] == 0:
                # Skip and clean empty dir
                if dest_type_dir.exists() and not any(dest_type_dir.rglob("*.pdf")):
                    shutil.rmtree(dest_type_dir, ignore_errors=True)
                continue
            rows.append({
                "folder_key": folder_key,
                "label": label,
                "type_name": type_name,
                **stats,
            })
            grand["sources"] += stats["sources_count"]
            grand["synth"] += stats["synth_count"]
            grand["linked"] += stats["sources_linked"] + stats["synth_linked"]
            grand["copied"] += stats["copied"]
            grand["excluded_blank"] += stats["excluded_blank"]
        domain_rows.append((domain_name, rows))

    # INDEX.md
    md = DEST_ROOT / "INDEX.md"
    with md.open("w") as f:
        f.write("# realdoc-bench v1.1\n\n")
        f.write(f"**{grand['sources']} source PDFs · {grand['synth']} synthetic/filled PDFs · "
                f"{len([r for _, rows in domain_rows for r in rows])} doc types "
                f"across {sum(1 for _, rows in domain_rows if rows)} domains**\n\n")
        f.write("Curated snapshot grouped by domain. Source PDFs are the originals "
                "(hand-triaged real filed forms + reports). Synthetic PDFs are "
                "AI-filled variants of blank templates with diverse personas; see "
                "each `_sources_manifest.csv` for per-PDF provenance.\n\n")
        f.write("Files are hardlinked from the per-type folders in `datasets/extraction/v1_*/`; "
                "zero extra disk used. If you copy v1.1/ elsewhere, files will be duplicated.\n\n")
        for domain_name, rows in domain_rows:
            if not rows:
                continue
            f.write(f"## {domain_name}\n\n")
            f.write("| Doc type | Sources | Synthetic | Folder |\n|---|---:|---:|---|\n")
            d_src = d_syn = 0
            for r in rows:
                f.write(f"| `{r['label']}` | {r['sources_count']} | {r['synth_count']} | "
                        f"`{safe_dirname(domain_name)}/{r['type_name']}/` |\n")
                d_src += r["sources_count"]; d_syn += r["synth_count"]
            f.write(f"| **Subtotal** | **{d_src}** | **{d_syn}** | |\n\n")
        f.write("## Grand total\n\n")
        f.write(f"**{grand['sources']} source · {grand['synth']} synthetic · "
                f"{grand['linked']} hardlinked · {grand['copied']} copied (cross-fs fallback)**\n\n")
        f.write("Generated by `scripts/build_v1_1.py`.\n")

    # Console summary
    print()
    for domain_name, rows in domain_rows:
        if not rows:
            continue
        d_src = sum(r["sources_count"] for r in rows)
        d_syn = sum(r["synth_count"] for r in rows)
        print(f"  {safe_dirname(domain_name):40} sources={d_src:>4}  synth={d_syn:>4}  "
              f"types={len(rows)}")
    print(f"\n  GRAND: {grand['sources']} sources, {grand['synth']} synthetic "
          f"({grand['linked']} hardlinks, {grand['copied']} copies)")
    print(f"\nWrote {md.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
