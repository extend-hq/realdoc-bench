#!/usr/bin/env python3
"""Apply user's guide review decisions:
  - DELETE_ALL N PDFs: remove source, manifest row, v1.1 link, derived synth fills
  - KEEP_PAGES N PDFs: extract picked pages into a new excerpt PDF, mark
    is_filled='filled' (so v1.2 picks it up), remove the original guide
  - SKIPPED: leave alone
"""
import csv
import os
import sys
from datetime import date as date_cls
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "datasets" / "extraction"

# (type, filename, kept_pages_or_None)
DELETE_ALL = [
    ("v1_air_waybill", "asean_air_waybill_trainer.pdf"),
    ("v1_air_waybill", "iata_PAWB_functional_specs.pdf"),
    ("v1_air_waybill", "iata_eAWB_implementation_playbook.pdf"),
    ("v1_cms_1500",    "cms_medicare_billing_837p_cms1500.pdf"),
    ("v1_eu_sad",      "eu_taxation_sad_presentation.pdf"),
    ("v1_form_5500",   "irs_5500EZ_instructions.pdf"),
    ("v1_hud1",        "cfpb_HUD1_explainer.pdf"),
    ("v1_lloyds_mrc",  "limoss_iMRC_guidance_v1.pdf"),
    ("v1_lloyds_mrc",  "lloyds_Y4173_market_bulletin.pdf"),
    ("v1_phyto_cert",  "fao_ISPM12_phytosanitary_certificates.pdf"),
    ("v1_phyto_cert",  "exim_phytosanitary_certificate_procedure.pdf"),
    ("v1_urla_1003",   "fhlb_mpf_URLA_exhibit_Y.pdf"),
    ("v1_vaers",       "fda_how_to_complete_vaers.pdf"),
    ("v1_vaers",       "cdc_vaers_factsheet.pdf"),
]

KEEP_PAGES = [
    ("v1_clinical_crf", "bu_bedac_crf_design.pdf",           [19, 21, 22]),
    ("v1_phyto_cert",   "ippc_ISPM12_2014_en.pdf",           [22, 23]),
    ("v1_phyto_cert",   "ippc_export_program_manual.pdf",    [63, 118, 120, 248]),
    ("v1_ub04",         "ms_medicaid_UB04_form_guide.pdf",   [18, 19]),
    ("v1_w2_1099",      "sc_dor_W2_1099_filing_specs.pdf",   [11]),
]


def manifest_path(type_name: str) -> Path:
    return DATA / type_name / "_sources_manifest.csv"


def update_manifest(type_name: str, *, drop: str = None,
                    add_row: dict = None):
    """Drop a row by local_filename, optionally append a new row."""
    m = manifest_path(type_name)
    rows = list(csv.DictReader(m.open())) if m.exists() else []
    fields = list(rows[0].keys()) if rows else [
        "local_filename", "source_url", "source_site", "is_filled",
        "doc_pages", "downloaded_on", "notes",
    ]
    if drop:
        rows = [r for r in rows if r["local_filename"] != drop]
    if add_row:
        # Make sure all fields exist
        for k in fields:
            add_row.setdefault(k, "")
        rows.append(add_row)
    with m.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def delete_source_and_links(type_name: str, filename: str) -> dict:
    """Delete the source PDF, v1.1 hardlinks, and derived synth fills."""
    stats = {"src": 0, "v11_links": 0, "synth_pagewise": 0, "synth_oldbatch": 0}
    src = DATA / type_name / "sources" / filename
    if src.exists():
        src.unlink()
        stats["src"] = 1
    # v1.1 hardlinks (same filename anywhere in v1.1/)
    for p in (DATA / "v1.1").rglob(filename):
        p.unlink()
        stats["v11_links"] += 1
    # Synth fills derived from this stem
    stem = src.stem
    # gpt-image-2 / pagewise-filled
    syn_dir = DATA / "v1.1_pagewise_filled" / type_name / stem
    if syn_dir.exists():
        for f in syn_dir.rglob("*"):
            if f.is_file():
                f.unlink()
                stats["synth_pagewise"] += 1
        try:
            for d in sorted(syn_dir.rglob("*"), reverse=True):
                if d.is_dir(): d.rmdir()
            syn_dir.rmdir()
        except OSError:
            pass
    # earlier batch
    for f in (DATA / "v1.1_synthetic" / type_name).glob(f"{stem}*.pdf") if (DATA / "v1.1_synthetic" / type_name).exists() else []:
        f.unlink()
        stats["synth_oldbatch"] += 1
    return stats


def extract_pages(type_name: str, filename: str, pages_1based: list[int]) -> str:
    """Extract the selected pages into a new PDF, returns new filename.

    New PDF named: <stem>__excerpt_p<page1>-<page2>.pdf
    Marks is_filled='filled' in manifest so v1.2 picks it up.
    """
    src = DATA / type_name / "sources" / filename
    stem = src.stem
    page_list = sorted(set(pages_1based))
    suffix = "p" + "_p".join(str(p) for p in page_list)
    new_name = f"{stem}__excerpt_{suffix}.pdf"
    dest = src.parent / new_name

    with fitz.open(src) as srcdoc:
        out = fitz.open()
        for p in page_list:
            if 1 <= p <= srcdoc.page_count:
                out.insert_pdf(srcdoc, from_page=p - 1, to_page=p - 1)
        out.save(dest, garbage=3, deflate=True)
        out.close()

    # Pull old row's source_url/notes for context
    m = manifest_path(type_name)
    old_row = None
    if m.exists():
        for r in csv.DictReader(m.open()):
            if r["local_filename"] == filename:
                old_row = r
                break

    new_row = {
        "local_filename": new_name,
        "source_url": (old_row or {}).get("source_url", ""),
        "source_site": (old_row or {}).get("source_site", ""),
        "is_filled": "filled",
        "doc_pages": str(len(page_list)),
        "downloaded_on": date_cls.today().isoformat(),
        "notes": f"Excerpt from guide PDF {filename}; kept pages {page_list}. "
                 f"Original notes: {(old_row or {}).get('notes','')[:140]}",
    }
    return new_name, new_row


def main() -> int:
    print("=== DELETE_ALL ===")
    for t, fn in DELETE_ALL:
        s = delete_source_and_links(t, fn)
        update_manifest(t, drop=fn)
        print(f"  {t}/{fn}  src={s['src']}  v1.1={s['v11_links']}  synth={s['synth_pagewise']}+{s['synth_oldbatch']}")

    print()
    print("=== KEEP_PAGES (extract excerpt, drop original) ===")
    for t, fn, pages in KEEP_PAGES:
        try:
            new_name, new_row = extract_pages(t, fn, pages)
            # Drop original row + add new excerpt row in one manifest write
            m = manifest_path(t)
            rows = list(csv.DictReader(m.open()))
            fields = list(rows[0].keys())
            rows = [r for r in rows if r["local_filename"] != fn]
            for k in fields:
                new_row.setdefault(k, "")
            rows.append(new_row)
            with m.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            # Delete the original source PDF + v1.1 links + ANY synth derived from it
            s = delete_source_and_links(t, fn)
            print(f"  {t}/{fn}  → {new_name} ({len(pages)}p)  src={s['src']}  v1.1={s['v11_links']}  synth={s['synth_pagewise']}+{s['synth_oldbatch']}")
        except Exception as e:
            print(f"  {t}/{fn}  ERROR: {e}", file=sys.stderr)
    print()
    print("=== SKIPPED (left as-is, still marked is_filled=guide so out of v1.2) ===")
    print("  [16] v1_ub04/carelon_UB04_tips.pdf")
    print("  [17] v1_ub04/cms_UB04_billing_guide_837i.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
