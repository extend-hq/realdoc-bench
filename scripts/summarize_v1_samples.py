#!/usr/bin/env python3
"""Domain-grouped summary of the full benchmark dataset (v0 + v0_tax + v1_*).

Reads each folder's _sources_manifest.csv plus v0 manifests, applies a domain
mapping, and renders:
  - console summary grouped by domain
  - datasets/extraction/v1_INDEX.md
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets" / "extraction"


# (domain, [(folder_name, display_label, doc_type_hint), ...])
DOMAINS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("Medical", [
        ("v0/med_autopsy_001", "med_autopsy (v0 hand-curated)", "autopsy"),
        ("v0/med_protocol_001", "med_protocol (v0 hand-curated)", "protocol"),
        ("v1_med_autopsy", "med_autopsy", "autopsy"),
        ("v1_med_protocol", "med_protocol", "protocol"),
        ("v1_clinical_consent", "clinical_consent", "consent"),
        ("v1_clinical_crf", "clinical_crf", "crf"),
        ("v1_clinical_facesheet", "clinical_facesheet", "facesheet"),
        ("v1_cms_1500", "cms_1500", "claim"),
    ]),
    ("Financial (incl. shipping/cargo)", [
        ("v0/fin_10k_001", "sec_10k (v0 hand-curated)", "10K"),
        ("v0/fin_audit_001", "fin_audit (v0 hand-curated)", "audit"),
        ("v0/fin_audit_002", "fin_audit (v0 hand-curated)", "audit"),
        ("v1_sec_10k", "sec_10k", "10K"),
        ("v1_audit_report", "audit_report", "audit"),
        ("v1_acord_25", "acord_25 (COI)", "insurance"),
        ("v1_acord_125_126", "acord_125_126 (commercial app)", "insurance"),
        ("v1_acord_4", "acord_4 (homeowner+loss)", "insurance"),
        ("v1_bol_inland", "bol_inland", "shipping"),
        ("v1_commercial_invoice", "commercial_invoice", "shipping"),
        ("v1_packing_list", "packing_list", "shipping"),
        ("v1_pod", "proof_of_delivery", "shipping"),
    ]),
    ("Mortgage & Legal", [
        ("v1_fnma_710", "fnma_710 (mortgage assistance)", "mortgage"),
        ("v1_urar", "urar (Fannie 1004 appraisal)", "appraisal"),
        ("v1_cd", "closing_disclosure", "mortgage"),
        ("v1_real_estate_contract", "real_estate_contract", "legal"),
    ]),
    ("Tax", [
        ("v0_tax", "presidential_tax_returns (v0)", "tax"),
    ]),
    ("Other (technical)", [
        ("v0/tech_datasheet_001", "tech_datasheet (v0 hand-curated)", "datasheet"),
        ("v1_tech_datasheet", "tech_datasheet", "datasheet"),
    ]),
    ("Budget & nonprofit financials", [
        ("v1_budget_cafr", "budget_cafr (city/county ACFRs)", "cafr"),
        ("v1_budget_sf424a", "budget_sf424a (federal grant budget)", "grant_budget"),
        ("v1_budget_amendment", "budget_amendment (transfer/amend forms)", "amendment"),
        ("v1_budget_990", "budget_990 (nonprofit IRS filings)", "990"),
    ]),
]

# Append new sub-types into existing domains
DOMAINS_BY_NAME = {name: lst for name, lst in DOMAINS}
DOMAINS_BY_NAME["Medical"].extend([
    ("v1_vaers", "vaers (vaccine adverse events)", "vaers"),
    ("v1_ub04", "ub04 (CMS-1450 institutional claim)", "claim"),
])
DOMAINS_BY_NAME["Financial (incl. shipping/cargo)"].extend([
    ("v1_eu_sad", "eu_sad (EU customs Single Admin Doc)", "customs"),
    ("v1_air_waybill", "air_waybill (IATA AWB)", "shipping"),
    ("v1_cmr", "cmr (international road transport)", "shipping"),
    ("v1_phyto_cert", "phyto_cert (phytosanitary cert)", "agri_export"),
    ("v1_loc", "loc (letter of credit / SWIFT MT700)", "trade_finance"),
    ("v1_sec_10q", "sec_10q (quarterly SEC filing)", "10Q"),
    ("v1_lloyds_mrc", "lloyds_mrc (London insurance market contract)", "insurance"),
    ("v1_esg", "esg (CDP / sustainability reports)", "esg"),
])
DOMAINS_BY_NAME["Mortgage & Legal"].extend([
    ("v1_urla_1003", "urla_1003 (Uniform Residential Loan App)", "mortgage"),
    ("v1_hud1", "hud1 (HUD-1 settlement statement)", "settlement"),
])
DOMAINS_BY_NAME["Tax"].extend([
    ("v1_w2_1099", "w2_1099 (IRS wage/info forms)", "tax"),
])
DOMAINS_BY_NAME["Financial (incl. shipping/cargo)"].extend([
    ("v1_bankruptcy_schedule", "bankruptcy_schedule (PACER Chapter 11 filings)", "court"),
    ("v1_form_5500", "form_5500 (DOL retirement plan return)", "pension"),
    ("v1_fec_form3", "fec_form3 (campaign finance receipts/disbursements)", "campaign_fin"),
])
# ─── Finance split additions (May 2026) — new financial doc types ───
DOMAINS_BY_NAME["Tax"].extend([
    ("v1_irs_specialty", "irs_specialty (Schedule K-1, Form 2848, 8949, etc.)", "tax"),
    ("v1_intl_tax_ca", "intl_tax_ca (Canada CRA — T1, T2, T4, T5)", "tax"),
    ("v1_intl_tax_uk", "intl_tax_uk (UK HMRC — SA100, CT600)", "tax"),
    ("v1_intl_tax_au", "intl_tax_au (Australia ATO)", "tax"),
    ("v1_intl_tax_in", "intl_tax_in (India ITR-1 through ITR-7)", "tax"),
    ("v1_intl_tax_de", "intl_tax_de (Germany ESt 1A + Anlagen)", "tax"),
    ("v1_intl_tax_nz", "intl_tax_nz (NZ IR3, IR4)", "tax"),
    ("v1_intl_tax_sg", "intl_tax_sg (Singapore IRAS Form B/B1/P)", "tax"),
    ("v1_intl_tax_za", "intl_tax_za (South Africa SARS)", "tax"),
    ("v1_intl_tax_jp", "intl_tax_jp (Japan NTA)", "tax"),
    ("v1_intl_tax_br", "intl_tax_br (Brazil RFB)", "tax"),
])
DOMAINS_BY_NAME["Financial (incl. shipping/cargo)"].extend([
    ("v1_sec_13f", "sec_13f (institutional holdings — dense table)", "sec"),
    ("v1_sec_def14a", "sec_def14a (proxy statements — comp tables)", "sec"),
    ("v1_sec_form_3_4_5", "sec_form_3_4_5 (insider transactions — checkbox)", "sec"),
    ("v1_sec_form_adv", "sec_form_adv (investment adviser — checkbox heavy)", "sec"),
    ("v1_fec_form5", "fec_form5 (FEC independent expenditures)", "campaign_fin"),
    ("v1_fec_form6", "fec_form6 (FEC 48-hour notice / debts)", "campaign_fin"),
    ("v1_fec_form7", "fec_form7 (FEC communications cost)", "campaign_fin"),
    ("v1_ffiec_call_report", "ffiec_call_report (UBPR bank regulatory)", "regulatory"),
    ("v1_ffiec_hmda", "ffiec_hmda (HMDA loan application register)", "regulatory"),
    ("v1_ferc_form1", "ferc_form1 (Electric utility annual report)", "regulatory"),
    ("v1_ferc_form2", "ferc_form2 (Natural gas pipeline annual)", "regulatory"),
    ("v1_naic_annual_statement", "naic_annual_statement (insurance statutory)", "regulatory"),
    ("v1_eob", "eob (Explanation of Benefits — medical insurance)", "medical_finance"),
    ("v1_bol_ocean", "bol_ocean (Ocean / master / sea waybill)", "shipping"),
])
DOMAINS_BY_NAME["Other (technical)"].extend([
    ("v1_ferpa_release", "ferpa_release (education records consent)", "education"),
])
# Rebuild DOMAINS list preserving the original order
DOMAINS = [(name, DOMAINS_BY_NAME[name]) for name, _ in DOMAINS]


def get_v1_stats(folder_name: str) -> tuple[int, int, int, int]:
    """Return (n_pdfs, total_pp, filled, blank) for a v1_* folder."""
    folder = DATASETS / folder_name
    manifest = folder / "_sources_manifest.csv"
    if not manifest.exists():
        return (0, 0, 0, 0)
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    pages = []
    filled = blank = 0
    for r in rows:
        try:
            pages.append(int(r.get("doc_pages", "")))
        except (ValueError, TypeError):
            pass
        kind = (r.get("is_filled") or "").lower()
        if "fill" in kind:
            filled += 1
        elif "blank" in kind:
            blank += 1
    return (len(rows), sum(pages), filled, blank)


def get_v0_single(folder_relpath: str) -> tuple[int, int, int, int]:
    """v0 entry like 'v0/med_autopsy_001' → look at v0/sources/<docid>.pdf and
    count as 1 doc, filled (hand-curated). Page count via pymupdf if available."""
    parent, doc_id = folder_relpath.split("/", 1)
    pdf = DATASETS / parent / "sources" / f"{doc_id}.pdf"
    if not pdf.exists():
        return (0, 0, 0, 0)
    pages = 0
    try:
        import fitz
        with fitz.open(pdf) as doc:
            pages = doc.page_count
    except Exception:
        pages = 0
    return (1, pages, 1, 0)


def get_v0_tax() -> tuple[int, int, int, int]:
    folder = DATASETS / "v0_tax"
    pdfs = list((folder / "sources").glob("*.pdf"))
    if not pdfs:
        return (0, 0, 0, 0)
    pages = 0
    try:
        import fitz
        for p in pdfs:
            try:
                with fitz.open(p) as doc:
                    pages += doc.page_count
            except Exception:
                pass
    except Exception:
        pass
    return (len(pdfs), pages, len(pdfs), 0)


def get_stats(folder_key: str) -> tuple[int, int, int, int]:
    if folder_key.startswith("v0/"):
        return get_v0_single(folder_key)
    if folder_key == "v0_tax":
        return get_v0_tax()
    return get_v1_stats(folder_key)


def main() -> int:
    grand = [0, 0, 0, 0]
    domain_rows: list[tuple[str, list[tuple[str, str, tuple[int, int, int, int]]]]] = []
    for domain_name, entries in DOMAINS:
        rows = []
        for folder_key, label, _hint in entries:
            stats = get_stats(folder_key)
            if stats[0] > 0:
                rows.append((folder_key, label, stats))
        domain_rows.append((domain_name, rows))

    # Console
    print()
    for domain_name, rows in domain_rows:
        if not rows:
            continue
        d_tot = [0, 0, 0, 0]
        print(f"=== {domain_name} ===")
        print(f"  {'doc_type':40} {'pdfs':>5} {'pages':>7} {'filled':>7} {'blank':>6}")
        for folder_key, label, (n, p, f, b) in rows:
            print(f"  {label:40} {n:>5} {p:>7} {f:>7} {b:>6}")
            d_tot[0] += n; d_tot[1] += p; d_tot[2] += f; d_tot[3] += b
        print(f"  {'-' * 70}")
        print(f"  {'SUBTOTAL':40} {d_tot[0]:>5} {d_tot[1]:>7} {d_tot[2]:>7} {d_tot[3]:>6}")
        print()
        for i in range(4):
            grand[i] += d_tot[i]
    print(f"GRAND TOTAL: {grand[0]} PDFs, {grand[1]} pages, "
          f"{grand[2]} filled / {grand[3]} blank")

    # Markdown index
    md = DATASETS / "v1_INDEX.md"
    with md.open("w") as f:
        f.write("# Benchmark dataset index\n\n")
        f.write(f"**{grand[0]} PDFs across all sets — {grand[1]:,} pages, "
                f"{grand[2]} filled / {grand[3]} blank.**\n\n")
        f.write("Grouped by domain. v0 = hand-curated docs with bespoke schemas + GT. "
                "v0_tax = presidential tax returns. v1_* = public-source samples curated for "
                "type-level diversity (provenance in each folder's `_sources_manifest.csv`).\n\n")
        for domain_name, rows in domain_rows:
            if not rows:
                continue
            d_tot = [0, 0, 0, 0]
            f.write(f"## {domain_name}\n\n")
            f.write("| Doc type | PDFs | Pages | Filled | Blank | Avg pp |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for folder_key, label, (n, p, fl, bl) in rows:
                avg = (p / n) if n else 0
                f.write(f"| `{label}` | {n} | {p} | {fl} | {bl} | {avg:.1f} |\n")
                d_tot[0] += n; d_tot[1] += p; d_tot[2] += fl; d_tot[3] += bl
            avg = (d_tot[1] / d_tot[0]) if d_tot[0] else 0
            f.write(f"| **Subtotal** | **{d_tot[0]}** | **{d_tot[1]}** | "
                    f"**{d_tot[2]}** | **{d_tot[3]}** | **{avg:.1f}** |\n\n")
        f.write(f"## Grand total\n\n")
        f.write(f"**{grand[0]} PDFs · {grand[1]:,} pages · "
                f"{grand[2]} filled · {grand[3]} blank**\n\n")
        f.write("*Generated by `scripts/summarize_v1_samples.py`. "
                "Per-PDF source URLs in each `_sources_manifest.csv`.*\n")
    print(f"\nWrote {md.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
