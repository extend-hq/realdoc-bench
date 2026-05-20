#!/usr/bin/env python3
"""Download international tax form templates from government tax authority sites.

Sources are direct PDF URLs from official government sites. All free, all blank
templates (filled returns are PII and rarely public).

Output:
  datasets/extraction/v1_intl_tax_<country>/sources/<form>.pdf
  + manifest CSV per type

Countries covered (8 plus US): CA, UK, AU, IN, DE, NZ, SG, ZA, plus extra US
specialty forms (Schedule K-1, Form 2848, Form 8949).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as date_cls
from pathlib import Path

import requests

try:
    import fitz
except ImportError:
    print("pymupdf not installed", file=sys.stderr)
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets" / "extraction"

UA = "Extend realdoc-bench/0.1 (research; ameya@extend.app)"
TIMEOUT = 60
MIN_PDF_BYTES = 20_000  # tax templates can be small (single-page)

# ─────────────────────────────────────────────────────────────────────────────
# URL catalog: (type, [(filename_stem, url, notes), ...])
# Filename will be `<stem>.pdf`. Always saved with is_filled="blank".
# ─────────────────────────────────────────────────────────────────────────────

CATALOG: dict[str, list[tuple[str, str, str]]] = {
    # USA — IRS specialty forms (extending our existing US tax coverage)
    "v1_irs_specialty": [
        ("irs_schedule_k1_form_1065_2024", "https://www.irs.gov/pub/irs-pdf/f1065sk1.pdf",
         "IRS Schedule K-1 (Form 1065) — Partner's Share of Income, Deductions, Credits (checkbox grid for partner status)"),
        ("irs_schedule_k1_form_1120s_2024", "https://www.irs.gov/pub/irs-pdf/f1120ssk.pdf",
         "IRS Schedule K-1 (Form 1120-S) — S-corp Shareholder's Share"),
        ("irs_schedule_k1_form_1041_2024", "https://www.irs.gov/pub/irs-pdf/f1041sk1.pdf",
         "IRS Schedule K-1 (Form 1041) — Beneficiary's Share from Estate/Trust"),
        ("irs_form_2848_2024", "https://www.irs.gov/pub/irs-pdf/f2848.pdf",
         "IRS Form 2848 — Power of Attorney (heavy checkboxes for representative acts)"),
        ("irs_form_8949_2024", "https://www.irs.gov/pub/irs-pdf/f8949.pdf",
         "IRS Form 8949 — Sales of Capital Assets (dense transaction table)"),
        ("irs_form_8821_2024", "https://www.irs.gov/pub/irs-pdf/f8821.pdf",
         "IRS Form 8821 — Tax Information Authorization (checkbox-heavy)"),
        ("irs_form_5471_2024", "https://www.irs.gov/pub/irs-pdf/f5471.pdf",
         "IRS Form 5471 — Information Return for Foreign Corporations (multi-page schedules + checkboxes)"),
        ("irs_form_1040_2024", "https://www.irs.gov/pub/irs-pdf/f1040.pdf",
         "IRS Form 1040 — Individual Income Tax Return (filing status + dependent checkboxes)"),
        ("irs_form_1040_sch_a_2024", "https://www.irs.gov/pub/irs-pdf/f1040sa.pdf",
         "IRS Form 1040 Schedule A — Itemized Deductions"),
        ("irs_form_1040_sch_c_2024", "https://www.irs.gov/pub/irs-pdf/f1040sc.pdf",
         "IRS Form 1040 Schedule C — Profit/Loss from Business"),
    ],

    # Canada — CRA
    "v1_intl_tax_ca": [
        ("cra_t1_general_2024", "https://www.canada.ca/content/dam/cra-arc/formspubs/pbg/5000-r/5000-r-24e.pdf",
         "Canada T1 General Income Tax and Benefit Return 2024"),
        ("cra_t2_corporation_2024", "https://www.canada.ca/content/dam/cra-arc/formspubs/pbg/t2/t2-23e.pdf",
         "Canada T2 Corporation Income Tax Return"),
        ("cra_t4_statement_remuneration", "https://www.canada.ca/content/dam/cra-arc/formspubs/pbg/t4/t4-fill-25e.pdf",
         "Canada T4 Statement of Remuneration Paid"),
        ("cra_t5_investment_income", "https://www.canada.ca/content/dam/cra-arc/formspubs/pbg/t5/t5-fill-24e.pdf",
         "Canada T5 Statement of Investment Income"),
        ("cra_t2125_self_employment", "https://www.canada.ca/content/dam/cra-arc/formspubs/pbg/t2125/t2125-fill-24e.pdf",
         "Canada T2125 Statement of Business or Professional Activities"),
    ],

    # UK — HMRC
    "v1_intl_tax_uk": [
        ("hmrc_sa100_2024", "https://assets.publishing.service.gov.uk/media/65e94b6df1cab30011fc4395/SA100_2024.pdf",
         "UK HMRC SA100 — Self Assessment Tax Return 2024"),
        ("hmrc_sa101_additional_2024", "https://assets.publishing.service.gov.uk/media/65e94c66f1cab30013b2feeb/SA101_2024.pdf",
         "UK HMRC SA101 — Additional Information Tax Return"),
        ("hmrc_sa105_uk_property_2024", "https://assets.publishing.service.gov.uk/media/65e94cd2cf7eb1000d33e317/SA105_2024.pdf",
         "UK HMRC SA105 — UK Property Pages"),
        ("hmrc_ct600_2024", "https://assets.publishing.service.gov.uk/media/64d4d3deca39c2000e1c50fd/CT600_2024_v3.pdf",
         "UK HMRC CT600 — Company Tax Return (Corporation Tax)"),
        ("hmrc_p11d", "https://assets.publishing.service.gov.uk/media/63bbe2238fa8f53999b16f9b/P11d-blank-form-2022-23.pdf",
         "UK HMRC P11D — Expenses and Benefits"),
    ],

    # Australia — ATO
    "v1_intl_tax_au": [
        ("ato_nat2541_individual_2024", "https://www.ato.gov.au/forms-and-instructions/tax-return-for-individuals-2024",
         "ATO Tax return for individuals 2024 (NAT 2541) — landing page"),
        ("ato_bas_business_activity_statement", "https://www.ato.gov.au/forms-and-instructions/business-activity-statement",
         "ATO Business Activity Statement"),
    ],

    # India — Income Tax Department
    "v1_intl_tax_in": [
        ("itd_itr1_sahaj_2024", "https://www.incometax.gov.in/iec/foportal/sites/default/files/2024-04/ITR_1_2024_25.pdf",
         "India ITR-1 (Sahaj) — Individual income up to ₹50 lakh"),
        ("itd_itr2_2024", "https://www.incometax.gov.in/iec/foportal/sites/default/files/2024-04/ITR_2_2024_25.pdf",
         "India ITR-2 — Individuals/HUFs without business income"),
        ("itd_itr3_2024", "https://www.incometax.gov.in/iec/foportal/sites/default/files/2024-04/ITR_3_2024_25.pdf",
         "India ITR-3 — Individuals/HUFs with business income"),
        ("itd_itr4_sugam_2024", "https://www.incometax.gov.in/iec/foportal/sites/default/files/2024-04/ITR_4_2024_25.pdf",
         "India ITR-4 (Sugam) — Presumptive income"),
        ("itd_itr5_2024", "https://www.incometax.gov.in/iec/foportal/sites/default/files/2024-04/ITR_5_2024_25.pdf",
         "India ITR-5 — Firms/AOPs/BOIs"),
        ("itd_itr6_2024", "https://www.incometax.gov.in/iec/foportal/sites/default/files/2024-04/ITR_6_2024_25.pdf",
         "India ITR-6 — Companies"),
        ("itd_itr7_2024", "https://www.incometax.gov.in/iec/foportal/sites/default/files/2024-04/ITR_7_2024_25.pdf",
         "India ITR-7 — Trusts/charitable orgs"),
    ],

    # Germany — Federal Ministry of Finance
    "v1_intl_tax_de": [
        ("bzst_est1a_mantelbogen_2024",
         "https://www.formulare-bfinv.de/ffw/form/download/ESt1A_2024.pdf",
         "Germany Mantelbogen ESt 1A 2024 — main income tax form (checkbox-dense)"),
        ("bzst_anlage_n_2024",
         "https://www.formulare-bfinv.de/ffw/form/download/AnlageN_2024.pdf",
         "Germany Anlage N 2024 — employment income"),
        ("bzst_anlage_kind_2024",
         "https://www.formulare-bfinv.de/ffw/form/download/AnlageKind_2024.pdf",
         "Germany Anlage Kind 2024 — children allowance"),
        ("bzst_anlage_r_2024",
         "https://www.formulare-bfinv.de/ffw/form/download/AnlageR_2024.pdf",
         "Germany Anlage R 2024 — pension income"),
        ("bzst_anlage_kap_2024",
         "https://www.formulare-bfinv.de/ffw/form/download/AnlageKAP_2024.pdf",
         "Germany Anlage KAP 2024 — investment income"),
    ],

    # New Zealand — IRD
    "v1_intl_tax_nz": [
        ("ird_ir3_individual_2024", "https://www.ird.govt.nz/-/media/project/ir/home/documents/forms-and-guides/ir3/ir3-2024.pdf",
         "NZ IR3 — Individual income tax return 2024"),
        ("ird_ir4_company_2024", "https://www.ird.govt.nz/-/media/project/ir/home/documents/forms-and-guides/ir4/ir4-2024.pdf",
         "NZ IR4 — Companies income tax return 2024"),
        ("ird_ir526_donation_rebate", "https://www.ird.govt.nz/-/media/project/ir/home/documents/forms-and-guides/ir500-599/ir526/ir526-2024.pdf",
         "NZ IR526 — Tax credit claim form (donations)"),
    ],

    # Singapore — IRAS
    "v1_intl_tax_sg": [
        ("iras_form_b_self_employed", "https://www.iras.gov.sg/IRASHome/uploadedFiles/IRASHome/Quick_Links/Forms/2024/2024_form_b.pdf",
         "Singapore IRAS Form B — Self-Employed Income"),
        ("iras_form_b1_employee", "https://www.iras.gov.sg/IRASHome/uploadedFiles/IRASHome/Quick_Links/Forms/2024/2024_form_b1.pdf",
         "Singapore IRAS Form B1 — Employee Income"),
        ("iras_form_p_partnership", "https://www.iras.gov.sg/IRASHome/uploadedFiles/IRASHome/Quick_Links/Forms/2024/2024_form_p.pdf",
         "Singapore IRAS Form P — Partnership Income"),
    ],

    # South Africa — SARS
    "v1_intl_tax_za": [
        ("sars_itr12_individuals_2024", "https://www.sars.gov.za/wp-content/uploads/Ops/Forms/Income-Tax/ITR12-Individuals-2024.pdf",
         "South Africa SARS ITR12 — Individual return 2024"),
        ("sars_it14_companies_2024", "https://www.sars.gov.za/wp-content/uploads/Ops/Forms/Income-Tax/IT14-Companies.pdf",
         "South Africa SARS IT14 — Company return"),
    ],

    # Japan — National Tax Agency
    "v1_intl_tax_jp": [
        ("nta_kakutei_shinkoku_b_2024", "https://www.nta.go.jp/taxes/shiraberu/shinkoku/yoshiki/01.pdf",
         "Japan NTA Kakutei Shinkoku B — Individual income tax return (Form B)"),
    ],

    # Brazil — Receita Federal
    "v1_intl_tax_br": [
        ("rfb_dirpf_2024", "https://www.gov.br/receitafederal/pt-br/centrais-de-conteudo/publicacoes/manuais/declaracao-irpf-2024/declaracao-irpf-2024.pdf",
         "Brazil DIRPF — Individual Income Tax Return 2024"),
    ],
}


def safe_get(url: str, **kw) -> requests.Response | None:
    try:
        r = requests.get(url, timeout=TIMEOUT,
                         headers={"User-Agent": UA, "Accept": "application/pdf,text/html,*/*"},
                         allow_redirects=True, **kw)
        if r.status_code == 200:
            return r
        print(f"  ! GET {url[:80]} → {r.status_code}")
        return None
    except Exception as e:
        print(f"  ! GET {url[:80]} failed: {e}")
        return None


def validate_pdf(path: Path) -> tuple[bool, int]:
    try:
        if path.stat().st_size < MIN_PDF_BYTES:
            return False, 0
        with fitz.open(path) as doc:
            return doc.page_count >= 1, doc.page_count
    except Exception:
        return False, 0


def download_one(type_name: str, stem: str, url: str, notes: str) -> tuple[str, bool, int, str]:
    type_dir = DATASETS / type_name
    sources_dir = type_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{stem}.pdf"
    dest = sources_dir / fname
    if dest.exists():
        ok, pp = validate_pdf(dest)
        if ok:
            return (fname, True, pp, "cached")
        dest.unlink()
    r = safe_get(url, stream=True)
    if r is None:
        return (fname, False, 0, "http_fail")
    ct = r.headers.get("content-type", "").lower()
    if "pdf" not in ct and not url.lower().endswith(".pdf"):
        # might still be PDF but content-type not set; try saving
        pass
    tmp = dest.with_suffix(".pdf.partial")
    try:
        with tmp.open("wb") as f:
            for chunk in r.iter_content(64 * 1024):
                f.write(chunk)
        ok, pp = validate_pdf(tmp)
        if not ok:
            tmp.unlink(missing_ok=True)
            return (fname, False, 0, "not_pdf_or_too_small")
        tmp.rename(dest)
        return (fname, True, pp, "downloaded")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return (fname, False, 0, f"err: {e}")


def write_manifest(type_name: str, rows: list[dict]) -> None:
    """Write/merge manifest. Each row dict has keys: filename, source_url, source_site, doc_pages, notes."""
    type_dir = DATASETS / type_name
    manifest = type_dir / "_sources_manifest.csv"
    fields = ["local_filename", "source_url", "source_site", "is_filled",
              "doc_pages", "downloaded_on", "notes"]
    existing: dict[str, dict] = {}
    if manifest.exists():
        with manifest.open() as f:
            for r in csv.DictReader(f):
                existing[r["local_filename"]] = r
    for nr in rows:
        existing[nr["filename"]] = {
            "local_filename": nr["filename"],
            "source_url": nr["source_url"],
            "source_site": nr["source_site"],
            "is_filled": "blank",
            "doc_pages": str(nr["doc_pages"]),
            "downloaded_on": date_cls.today().isoformat(),
            "notes": nr["notes"],
        }
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for k in sorted(existing):
            w.writerow(existing[k])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--type", action="append",
                    help="Restrict to one or more type folders (repeatable)")
    args = ap.parse_args()

    plan: list[tuple[str, str, str, str]] = []
    for tt, entries in CATALOG.items():
        if args.type and tt not in args.type:
            continue
        for stem, url, notes in entries:
            plan.append((tt, stem, url, notes))

    print(f"Tax form hunt: {len(plan)} URLs across {len(set(p[0] for p in plan))} types")

    t0 = time.time()
    per_type_rows: dict[str, list[dict]] = {}
    n_ok = n_fail = 0
    from urllib.parse import urlparse

    def _job(args_):
        return (args_, download_one(*args_))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_job, p) for p in plan]
        for fut in as_completed(futures):
            (tt, stem, url, notes), (fname, ok, pp, status) = fut.result()
            if ok:
                n_ok += 1
                row = {
                    "filename": fname, "source_url": url,
                    "source_site": urlparse(url).netloc,
                    "doc_pages": pp, "notes": notes,
                }
                per_type_rows.setdefault(tt, []).append(row)
                print(f"  + {tt}/{fname}  {pp}pp  ({status})")
            else:
                n_fail += 1
                print(f"  ✗ {tt}/{fname}  → {status}")

    # Write manifests per type
    for tt, rows in per_type_rows.items():
        write_manifest(tt, rows)

    print(f"\nDone in {time.time()-t0:.0f}s — {n_ok} ok, {n_fail} fail")
    for tt in sorted(per_type_rows):
        print(f"  {tt}: {len(per_type_rows[tt])} PDFs")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
