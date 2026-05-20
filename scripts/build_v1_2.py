#!/usr/bin/env python3
"""Consolidate ALL real + synthetic PDFs into datasets/extraction/v1.2/.

Layout (per user direction 2026-05-18):

    v1.2/
    ├── INDEX.md
    ├── _manifest.csv          (one row per output PDF)
    ├── real/
    │   ├── supply_chain/
    │   ├── mortgage/
    │   ├── medical_healthcare/
    │   └── finance/           (incl. tax + ESG + audit + OIG + DOJ + historical_tax)
    └── synthetic/
        ├── supply_chain/
        ├── mortgage/
        ├── medical_healthcare/
        └── finance/

Rules:
  * Pagewise — every multi-page PDF is split into single-page outputs.
  * Real inclusion — manifest is_filled IN ('yes', 'filled', 'partial').
    Skip 'blank', 'guide', '_acroform_dump', '_rejected', '_quarantine_*'.
  * Synthetic inclusion — every PDF in v1.1_pagewise_filled/<type>/<template>/
    matching *__persona*.pdf, plus every PDF in v1.1_synthetic/<type>/.
  * Filename encodes origin: <type>__<doc>__p<NN>[__persona<NN>].pdf.
  * Hardlink when single-page; pdf-split when multi-page. Zero extra disk
    for single-page sources. Multi-page sources get N small new PDFs.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "datasets" / "extraction"
OUT = DATA / "v1.2"

# ───────────────────────────────────────────────────────────────────────────
# Domain mapping — 4 user-facing buckets
# ───────────────────────────────────────────────────────────────────────────
DOMAIN: dict[str, str] = {
    # Supply chain & logistics
    "v1_bol_inland": "supply_chain",
    "v1_bol_ocean": "supply_chain",
    "v1_air_waybill": "supply_chain",
    "v1_cmr": "supply_chain",
    "v1_commercial_invoice": "supply_chain",
    "v1_packing_list": "supply_chain",
    "v1_pod": "supply_chain",
    "v1_phyto_cert": "supply_chain",
    "v1_eu_sad": "supply_chain",
    "v1_loc": "supply_chain",

    # Mortgage & real estate
    "v1_fnma_710": "mortgage",
    "v1_hud1": "mortgage",
    "v1_urar": "mortgage",
    "v1_urla_1003": "mortgage",
    "v1_cd": "mortgage",
    "v1_real_estate_contract": "mortgage",

    # Medical / healthcare
    "v1_cms_1500": "medical_healthcare",
    "v1_ub04": "medical_healthcare",
    "v1_vaers": "medical_healthcare",
    "v1_patient_intake": "medical_healthcare",
    "v1_clinical_consent": "medical_healthcare",
    "v1_clinical_crf": "medical_healthcare",
    "v1_clinical_facesheet": "medical_healthcare",
    "v1_med_autopsy": "medical_healthcare",
    "v1_med_protocol": "medical_healthcare",
    "v1_eob": "medical_healthcare",

    # Finance (includes tax, ESG, audit, regulatory, govt financial reports)
    "v1_sec_10k": "finance",
    "v1_sec_10q": "finance",
    "v1_sec_13f": "finance",
    "v1_sec_def14a": "finance",
    "v1_sec_form_3_4_5": "finance",
    "v1_fec_form3": "finance",
    "v1_fec_form5": "finance",
    "v1_fec_form6": "finance",
    "v1_fec_form7": "finance",
    "v1_ffiec_call_report": "finance",
    "v1_ffiec_hmda": "finance",
    "v1_irs_specialty": "finance",
    "v1_form_5500": "finance",
    "v1_acord_125_126": "finance",
    "v1_acord_25": "finance",
    "v1_acord_4": "finance",
    "v1_bankruptcy_schedule": "finance",
    "v1_lloyds_mrc": "finance",
    "v1_esg": "finance",
    "v1_audit_report": "finance",
    "v1_budget_990": "finance",
    "v1_budget_cafr": "finance",
    "v1_budget_amendment": "finance",
    "v1_budget_sf424a": "finance",
    "v1_w2_1099": "finance",
    "v1_oig_dhs_fema": "finance",
    "v1_oig_sba": "finance",
    "v1_oig_usda": "finance",
    "v1_doj_fraud_complaint": "finance",
    "v1_historical_tax": "finance",
    "v1_politician_tax": "finance",
    "v1_sba_disaster_loan": "finance",
    # International tax — all under finance
    "v1_intl_tax_au": "finance",
    "v1_intl_tax_br": "finance",
    "v1_intl_tax_ca": "finance",
    "v1_intl_tax_de": "finance",
    "v1_intl_tax_in": "finance",
    "v1_intl_tax_jp": "finance",
    "v1_intl_tax_nz": "finance",
    "v1_intl_tax_sg": "finance",
    "v1_intl_tax_uk": "finance",
    "v1_intl_tax_za": "finance",
}

# Excluded types (don't fit any of the 4 user domains)
SKIP_TYPES = {"v1_tech_datasheet"}

# is_filled values that count as "include in real/"
REAL_INCLUDE_STATUS = {"yes", "filled", "partial"}

# Folders never to scan
SKIP_TOP_DIRS = {"_acroform_dump", "_rejected", "_quarantine_confidentiality_stamp",
                 "_quarantine_wrong_doctype", "_personas",
                 "v0", "v0_tax", "v1-customer", "v1-public",
                 "v1.1", "v1.1_pagewise", "v1.1_pagewise_rich", "v1_review", "v1_INDEX.md"}


def pad_page(n: int, total: int) -> str:
    width = max(2, len(str(total)))
    return f"{n:0{width}d}"


def safe_stem(s: str) -> str:
    s = re.sub(r"\.pdf$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    return s


def write_page(src_pdf: Path, page_idx: int, dest_pdf: Path) -> bool:
    """Write a single page from src_pdf as a standalone PDF at dest_pdf.
    If src is already 1-page, hardlink instead of re-encoding."""
    if dest_pdf.exists():
        return True
    dest_pdf.parent.mkdir(parents=True, exist_ok=True)
    try:
        with fitz.open(src_pdf) as src:
            if src.page_count == 1 and page_idx == 0:
                # hardlink: same inode, zero extra disk
                try:
                    os.link(src_pdf, dest_pdf)
                    return True
                except OSError:
                    # cross-device or other issue → fall through to copy via fitz
                    pass
            new = fitz.open()
            new.insert_pdf(src, from_page=page_idx, to_page=page_idx)
            new.save(dest_pdf, garbage=3, deflate=True)
            new.close()
        return True
    except Exception as e:
        print(f"  ! write_page fail {src_pdf.name} p{page_idx}: {type(e).__name__}: {e}", file=sys.stderr)
        return False


def collect_real() -> list[dict]:
    """Yield rows for every page of every 'filled' real source PDF."""
    rows = []
    for t in sorted(os.listdir(DATA)):
        if t in SKIP_TOP_DIRS: continue
        if not t.startswith("v1_"): continue
        if t in SKIP_TYPES: continue
        domain = DOMAIN.get(t)
        if not domain:
            print(f"  (skip — no domain mapping) {t}", file=sys.stderr)
            continue
        m = DATA / t / "_sources_manifest.csv"
        if not m.exists():
            continue
        for r in csv.DictReader(m.open()):
            st = (r.get("is_filled") or "").strip().lower()
            if st not in REAL_INCLUDE_STATUS:
                continue
            pdf = DATA / t / "sources" / r["local_filename"]
            if not pdf.exists():
                continue
            try:
                with fitz.open(pdf) as d:
                    total = d.page_count
            except Exception as e:
                print(f"  ! fitz fail {pdf}: {e}", file=sys.stderr)
                continue
            doc_stem = safe_stem(pdf.stem)
            for pi in range(total):
                pname = pad_page(pi + 1, total)
                dest_name = f"{t}__{doc_stem}__p{pname}.pdf"
                rows.append({
                    "src_pdf": str(pdf),
                    "page_idx": pi,
                    "kind": "real",
                    "domain": domain,
                    "type": t,
                    "doc_stem": doc_stem,
                    "page_num": pi + 1,
                    "doc_total_pages": total,
                    "persona": "",
                    "dest_name": dest_name,
                    "source_url": r.get("source_url", ""),
                    "is_filled": st,
                    "notes": r.get("notes", "")[:200],
                })
    return rows


def collect_synthetic() -> list[dict]:
    """Yield rows for every synthetic fill PDF.

    Two sources:
      v1.1_pagewise_filled/<type>/<template>/*__persona*.pdf  (each is 1 page)
      v1.1_synthetic/<type>/*.pdf  (mix of 1-page and multi-page)
    """
    rows = []

    # A — pagewise gpt-image-2 fills
    pw_root = DATA / "v1.1_pagewise_filled"
    if pw_root.exists():
        for t_dir in sorted(pw_root.iterdir()):
            if not t_dir.is_dir(): continue
            t = t_dir.name
            if t in SKIP_TYPES: continue
            domain = DOMAIN.get(t)
            if not domain:
                print(f"  (skip — no domain mapping) synthetic {t}", file=sys.stderr)
                continue
            for tpl_dir in sorted(t_dir.iterdir()):
                if not tpl_dir.is_dir(): continue
                tpl_stem = tpl_dir.name
                for pdf in sorted(tpl_dir.glob("*__persona*.pdf")):
                    # filename: <tpl_stem>__p<NN>__persona<NN>.pdf
                    m = re.search(r"__p(\d+)__persona(\d+)\.pdf$", pdf.name)
                    if not m: continue
                    page_n = int(m.group(1))
                    persona = m.group(2)
                    try:
                        with fitz.open(pdf) as d:
                            total = d.page_count
                    except Exception:
                        continue
                    dest_name = f"{t}__{safe_stem(tpl_stem)}__p{page_n:02d}__persona{persona}.pdf"
                    rows.append({
                        "src_pdf": str(pdf),
                        "page_idx": 0,
                        "kind": "synthetic",
                        "domain": domain,
                        "type": t,
                        "doc_stem": safe_stem(tpl_stem),
                        "page_num": page_n,
                        "doc_total_pages": total,
                        "persona": persona,
                        "dest_name": dest_name,
                        "source_url": "",
                        "is_filled": "synthetic",
                        "notes": f"gpt-image-2 fill from {tpl_stem}",
                    })

    # B — earlier v1.1_synthetic batch
    syn_root = DATA / "v1.1_synthetic"
    if syn_root.exists():
        for t_dir in sorted(syn_root.iterdir()):
            if not t_dir.is_dir(): continue
            t = t_dir.name
            if t in SKIP_TYPES: continue
            domain = DOMAIN.get(t)
            if not domain:
                print(f"  (skip — no domain mapping) synthetic {t}", file=sys.stderr)
                continue
            for pdf in sorted(t_dir.glob("*.pdf")):
                try:
                    with fitz.open(pdf) as d:
                        total = d.page_count
                except Exception:
                    continue
                stem = safe_stem(pdf.stem)
                for pi in range(total):
                    pname = pad_page(pi + 1, total)
                    # Try to keep persona suffix if filename already has it
                    m_per = re.search(r"persona[_-]?(\d+)", stem)
                    persona = m_per.group(1) if m_per else "01"
                    dest_name = f"{t}__{stem}__p{pname}__persona{persona}.pdf"
                    # dedupe within type/stem/page — earlier synth folder is messy
                    rows.append({
                        "src_pdf": str(pdf),
                        "page_idx": pi,
                        "kind": "synthetic",
                        "domain": domain,
                        "type": t,
                        "doc_stem": stem,
                        "page_num": pi + 1,
                        "doc_total_pages": total,
                        "persona": persona,
                        "dest_name": dest_name,
                        "source_url": "",
                        "is_filled": "synthetic",
                        "notes": "v1.1_synthetic earlier batch",
                    })
    return rows


def materialize(rows: list[dict], dry_run: bool) -> tuple[int, int, int]:
    n_done = n_skip = n_fail = 0
    seen_dest = {}
    for r in rows:
        dest = OUT / r["kind"] / r["domain"] / r["dest_name"]
        # collision: append a hash suffix
        if dest in seen_dest:
            base = dest.stem
            i = 2
            while True:
                cand = dest.with_name(f"{base}__d{i}.pdf")
                if cand not in seen_dest and not cand.exists():
                    dest = cand
                    break
                i += 1
        seen_dest[dest] = True
        r["dest_path"] = str(dest.relative_to(OUT))
        if dest.exists():
            n_skip += 1
            continue
        if dry_run:
            continue
        ok = write_page(Path(r["src_pdf"]), r["page_idx"], dest)
        if ok: n_done += 1
        else: n_fail += 1
    return n_done, n_skip, n_fail


def write_manifest(rows: list[dict]) -> None:
    csv_path = OUT / "_manifest.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dest_path", "kind", "domain", "type", "doc_stem", "page_num",
              "doc_total_pages", "persona", "is_filled", "source_url",
              "src_pdf", "page_idx", "notes"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL,
                          extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  manifest → {csv_path.relative_to(REPO)}  ({len(rows)} rows)")


def write_index(rows: list[dict]) -> None:
    from collections import defaultdict, Counter
    by_kd = defaultdict(Counter)  # (kind, domain) → Counter(type → count)
    by_kd_pages = defaultdict(int)
    by_kd_docs = defaultdict(set)  # set of (type, doc_stem)
    for r in rows:
        key = (r["kind"], r["domain"])
        by_kd[key][r["type"]] += 1
        by_kd_pages[key] += 1
        by_kd_docs[key].add((r["type"], r["doc_stem"]))

    lines = []
    lines.append("# realdoc-bench v1.2 — Consolidated Sample Index\n")
    total = len(rows)
    n_real = sum(by_kd_pages[k] for k in by_kd_pages if k[0] == "real")
    n_syn = sum(by_kd_pages[k] for k in by_kd_pages if k[0] == "synthetic")
    n_real_docs = sum(len(by_kd_docs[k]) for k in by_kd_docs if k[0] == "real")
    n_syn_docs = sum(len(by_kd_docs[k]) for k in by_kd_docs if k[0] == "synthetic")
    lines.append(f"**{total} pagewise PDFs** = {n_real} real (across {n_real_docs} source docs) "
                 f"+ {n_syn} synthetic (across {n_syn_docs} fills)\n")
    lines.append("Two top-level kinds × four domains. Every file is a single page. "
                 "Real sources are hand-collected filed documents. Synthetic are gpt-image-2 "
                 "fills of blank templates using diverse personas.\n")
    lines.append("- **real/**: actual filed documents from clinics, courts, IRS, SEC, EDGAR, etc.")
    lines.append("- **synthetic/**: AI-filled variants of blank templates.\n")
    lines.append("Provenance per page: see `_manifest.csv` (one row per file).\n")

    for kind in ("real", "synthetic"):
        kn = sum(by_kd_pages[k] for k in by_kd_pages if k[0] == kind)
        kd = sum(len(by_kd_docs[k]) for k in by_kd_docs if k[0] == kind)
        lines.append(f"## {kind}/ — {kn} pages from {kd} docs\n")
        for domain in ("supply_chain", "mortgage", "medical_healthcare", "finance"):
            key = (kind, domain)
            n_pages = by_kd_pages.get(key, 0)
            n_docs = len(by_kd_docs.get(key, set()))
            if n_pages == 0:
                continue
            lines.append(f"### {domain}/ — {n_pages} pages, {n_docs} docs\n")
            lines.append(f"| Type | Pages | Source docs |")
            lines.append(f"|---|---:|---:|")
            per_type = by_kd.get(key, Counter())
            for t in sorted(per_type, key=lambda x: -per_type[x]):
                n_p = per_type[t]
                n_d = len({ds for (tp, ds) in by_kd_docs[key] if tp == t})
                lines.append(f"| `{t}` | {n_p} | {n_d} |")
            lines.append("")

    index_path = OUT / "INDEX.md"
    index_path.write_text("\n".join(lines))
    print(f"  INDEX → {index_path.relative_to(REPO)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Plan but don't write files")
    ap.add_argument("--clean", action="store_true",
                    help="Wipe v1.2/ first (use after pipelines settle)")
    args = ap.parse_args()

    if args.clean:
        if OUT.exists():
            print(f"Cleaning {OUT.relative_to(REPO)} ...")
            shutil.rmtree(OUT)

    OUT.mkdir(parents=True, exist_ok=True)
    print("==> collecting real source PDFs ...")
    real_rows = collect_real()
    print(f"    {len(real_rows)} real pages")
    print("==> collecting synthetic fills ...")
    syn_rows = collect_synthetic()
    print(f"    {len(syn_rows)} synthetic pages")
    rows = real_rows + syn_rows
    print(f"==> total: {len(rows)} pages")

    print("==> materializing ...")
    n_done, n_skip, n_fail = materialize(rows, args.dry_run)
    print(f"    done {n_done}  skip {n_skip}  fail {n_fail}")

    print("==> writing manifest + index ...")
    write_manifest(rows)
    write_index(rows)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
