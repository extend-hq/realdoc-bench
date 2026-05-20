#!/usr/bin/env python3
"""Background hunter for real filled documents in thin v1_* categories.

Sources used (all no-auth-needed):
  - FEC: api.open.fec.gov/v1/filings  +  docquery.fec.gov PDFs
  - CourtListener: api/rest/v4/search?type=r (RECAP federal court exhibits)

Design:
  - Idempotent: skips if PDF already exists in target sources/ dir
  - Validates: PDF >= 30KB, opens with pymupdf, page_count >= 1
  - Heuristic filter: skips obvious blank-form / guidance docs
  - Appends to _sources_manifest.csv automatically (deduped by filename)
  - Logs progress to stdout (tee to file for tail-follow)

Categories targeted (priority order):
  1. v1_fec_form3        — FEC openFEC API (most reliable)
  2. v1_clinical_crf     — CourtListener: Taxotere MDL, Vioxx, opioid CRFs
  3. v1_packing_list     — CourtListener: customs/import litigation exhibits
  4. v1_commercial_invoice — same
  5. v1_w2_1099          — CourtListener: tax court exhibits
  6. v1_bol_inland       — CourtListener: maritime/cargo cases
  7. v1_air_waybill      — CourtListener: cargo claims
  8. v1_form_5500        — CourtListener: ERISA litigation (Form 5500 exhibits)

Usage:
  python -u scripts/hunt_real_docs.py --target 5 [--type v1_fec_form3 ...]

The --target N is the number of NEW PDFs per category (idempotent across runs).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests

try:
    import fitz  # pymupdf
except ImportError:
    print("pymupdf not installed", file=sys.stderr)
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets" / "extraction"

UA = "realdoc-bench/0.1 (research; contact: ameya@extend.app)"
TIMEOUT = 60
MIN_PDF_BYTES = 30_000  # 30KB — below this is almost certainly a stub

# Skip filenames / titles matching these — they're guides not filings
BLANK_GUIDE_PATTERNS = [
    r"\binstruction", r"\bguide\b", r"\btemplate\b", r"\bblank\b",
    r"\bsample\b", r"\bexample\b", r"\bworksheet\b", r"how[\s_-]?to",
    r"\bform[\s_-]?\d+[\s_-]?fillable", r"\bsupplement\b",
]
BLANK_RE = re.compile("|".join(BLANK_GUIDE_PATTERNS), re.IGNORECASE)


def safe_get(url: str, **kw) -> requests.Response | None:
    try:
        r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": UA}, **kw)
        if r.status_code == 200:
            return r
        return None
    except Exception as e:
        print(f"  ! GET {url[:80]} failed: {e}")
        return None


def slugify(s: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen]


def validate_pdf(path: Path) -> tuple[bool, int, str]:
    """Returns (valid, page_count, first_page_text_lower[:500])."""
    try:
        if path.stat().st_size < MIN_PDF_BYTES:
            return False, 0, ""
        with fitz.open(path) as doc:
            if doc.page_count < 1:
                return False, 0, ""
            text = doc[0].get_text("text")[:2000].lower()
            return True, doc.page_count, text
    except Exception:
        return False, 0, ""


def is_blank_or_guide(filename: str, first_page_text: str) -> bool:
    if BLANK_RE.search(filename):
        return True
    # Heuristic: a truly blank form has very little prose text on page 1
    # — but also has lots of field labels. Hard to disambiguate. Skip text check.
    return False


def existing_pdfs(folder: Path) -> set[str]:
    if not folder.exists():
        return set()
    return {p.name for p in folder.glob("*.pdf")}


def append_manifest(
    type_dir: Path,
    filename: str,
    source_url: str,
    doc_pages: int,
    is_filled: str,
    provenance: str,
) -> None:
    """Append a row using the existing manifest's schema (preserves column order)."""
    from datetime import date
    manifest = type_dir / "_sources_manifest.csv"
    default_fields = [
        "local_filename", "source_url", "source_site", "is_filled",
        "doc_pages", "downloaded_on", "notes",
    ]
    rows: list[dict] = []
    fieldnames = default_fields
    if manifest.exists():
        with manifest.open() as f:
            rdr = csv.DictReader(f)
            rows = list(rdr)
            if rdr.fieldnames:
                fieldnames = list(rdr.fieldnames)
        # dedupe — try both common name columns
        name_keys = [k for k in ("local_filename", "filename") if k in fieldnames]
        for r in rows:
            for nk in name_keys:
                if r.get(nk) == filename:
                    return
    # Build a row that supplies whatever the existing manifest expects.
    new_row: dict[str, str] = {fn: "" for fn in fieldnames}
    aliases = {
        "local_filename": filename, "filename": filename,
        "source_url": source_url,
        "source_site": urlparse(source_url).netloc or "",
        "site": urlparse(source_url).netloc or "",
        "is_filled": is_filled, "filled": is_filled,
        "doc_pages": str(doc_pages), "pages": str(doc_pages),
        "downloaded_on": date.today().isoformat(), "date": date.today().isoformat(),
        "license": "public_record",
        "provenance": provenance,
        "notes": f"auto-hunted; {provenance}",
    }
    for fn in fieldnames:
        if fn in aliases:
            new_row[fn] = aliases[fn]
    rows.append(new_row)
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore",
                           quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


def download_and_stage(
    url: str,
    type_dir: Path,
    suggested_name: str,
    source_label: str,
    provenance: str,
) -> bool:
    sources_dir = type_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    fname = suggested_name if suggested_name.endswith(".pdf") else f"{suggested_name}.pdf"
    if fname in existing_pdfs(sources_dir):
        return False
    if is_blank_or_guide(fname, ""):
        return False
    tmp = sources_dir / f".{fname}.partial"
    r = safe_get(url, stream=True)
    if not r:
        return False
    ct = r.headers.get("content-type", "").lower()
    if "pdf" not in ct and not url.lower().endswith(".pdf"):
        return False
    try:
        with tmp.open("wb") as f:
            for chunk in r.iter_content(64 * 1024):
                f.write(chunk)
        ok, pages, first_text = validate_pdf(tmp)
        if not ok or is_blank_or_guide(fname, first_text):
            tmp.unlink(missing_ok=True)
            return False
        final = sources_dir / fname
        tmp.rename(final)
        print(f"  + {type_dir.name}/{fname}  ({pages}pp, {final.stat().st_size//1024}KB)  [{source_label}]  src={url}")
        return True
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"  ! save {fname} failed: {e}")
        return False


# ============================================================================
# Hunter: FEC Form 3 / F3X / F3P via openFEC
# ============================================================================

def hunt_fec(target: int) -> int:
    """Pull recent Form 3 filings (F3=House, F3P=Pres, F3X=PAC) from openFEC."""
    type_dir = DATASETS / "v1_fec_form3"
    sources_dir = type_dir / "sources"
    existing = existing_pdfs(sources_dir)
    added = 0
    # mix of form types for diversity
    for form_type in ("F3", "F3P", "F3X"):
        if added >= target:
            break
        url = (
            "https://api.open.fec.gov/v1/filings/"
            f"?form_type={form_type}&api_key=DEMO_KEY"
            f"&per_page=40&sort=-receipt_date&min_receipt_date=2024-01-01"
        )
        r = safe_get(url)
        if not r:
            continue
        try:
            data = r.json()
        except Exception:
            continue
        for filing in data.get("results", []):
            if added >= target:
                break
            pdf_url = filing.get("pdf_url")
            if not pdf_url:
                continue
            cmte = filing.get("committee_name") or filing.get("committee_id", "fec")
            cycle = filing.get("election_cycle") or filing.get("report_year", "")
            report = filing.get("report_type", form_type)
            fname = f"fec_{form_type.lower()}_{slugify(cmte, 50)}_{cycle}_{slugify(str(report), 20)}.pdf"
            if fname in existing:
                continue
            if download_and_stage(pdf_url, type_dir, fname, "openFEC", f"openFEC filings API filing_id={filing.get('file_number')}"):
                added += 1
            time.sleep(0.3)  # courtesy
    return added


def hunt_fec_other_forms(target_per: int = 4) -> dict[str, int]:
    """Pull Form 5 (indep expenditures), Form 6 (debts), Form 7 (communications).
    These are typically smaller form-shaped reports (not the full Schedule A/B
    bundles which are inside Form 3 filings)."""
    added: dict[str, int] = {}
    # Map: openFEC form_type → our type folder
    form_map = [
        ("F5", "v1_fec_form5", "FEC Form 5 — Independent Expenditures"),
        ("F6", "v1_fec_form6", "FEC Form 6 — 48-Hour Notice (Loans/Late Debts)"),
        ("F7", "v1_fec_form7", "FEC Form 7 — Communications Cost"),
    ]
    for form_type, type_name, label in form_map:
        type_dir = DATASETS / type_name
        sources_dir = type_dir / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)
        existing = existing_pdfs(sources_dir)
        n = 0
        url = (
            "https://api.open.fec.gov/v1/filings/"
            f"?form_type={form_type}&api_key=DEMO_KEY"
            f"&per_page=20&sort=-receipt_date&min_receipt_date=2022-01-01"
        )
        r = safe_get(url)
        if not r:
            added[type_name] = 0
            continue
        try:
            data = r.json()
        except Exception:
            added[type_name] = 0
            continue
        for filing in data.get("results", []):
            if n >= target_per:
                break
            pdf_url = filing.get("pdf_url")
            if not pdf_url:
                continue
            cmte = filing.get("committee_name") or filing.get("committee_id") or "fec"
            year = filing.get("report_year") or filing.get("election_cycle") or ""
            fname = f"fec_{form_type.lower()}_{slugify(cmte, 50)}_{year}.pdf"
            if fname in existing:
                continue
            if download_and_stage(pdf_url, type_dir, fname, "openFEC",
                                   f"openFEC {label} filing_id={filing.get('file_number')}"):
                n += 1
            time.sleep(0.3)
        added[type_name] = n
    return added


# ============================================================================
# Hunter: CourtListener RECAP search
# ============================================================================

def courtlistener_search(query: str, max_results: int = 40) -> list[dict]:
    """Use v4 search API (no auth). type=r → RECAP documents."""
    url = (
        "https://www.courtlistener.com/api/rest/v4/search/"
        f"?type=r&q={requests.utils.quote(query)}&order_by=score+desc"
    )
    r = safe_get(url)
    if not r:
        return []
    try:
        data = r.json()
    except Exception:
        return []
    out = []
    for hit in data.get("results", []):
        # v4 search returns docket-grouped hits with nested recap_documents
        for doc in hit.get("recap_documents", [])[:5]:
            if not doc.get("is_available"):
                continue
            fp = doc.get("filepath_local") or doc.get("absolute_url")
            if not fp:
                continue
            if fp.startswith("/"):
                fp = "https://storage.courtlistener.com/" + fp.lstrip("/")
            elif fp.startswith("http"):
                pass
            else:
                fp = "https://www.courtlistener.com" + ("/" if not fp.startswith("/") else "") + fp
            out.append({
                "url": fp,
                "case": hit.get("caseName", ""),
                "court": hit.get("court", ""),
                "docket": hit.get("docketNumber", ""),
                "desc": doc.get("description") or doc.get("short_description", ""),
                "page_count": doc.get("page_count", 0),
            })
            if len(out) >= max_results:
                return out
    return out


def hunt_courtlistener_category(
    type_name: str, queries: list[str], target: int,
) -> int:
    type_dir = DATASETS / type_name
    sources_dir = type_dir / "sources"
    existing = existing_pdfs(sources_dir)
    added = 0
    for q in queries:
        if added >= target:
            break
        print(f"  CL query [{type_name}]: {q!r}")
        hits = courtlistener_search(q, max_results=30)
        time.sleep(1.0)
        for h in hits:
            if added >= target:
                break
            # filename from case + docket + desc
            base = slugify(f"cl_{h['case']}_{h['docket']}_{h['desc'][:30]}", 90)
            fname = f"{base}.pdf"
            if fname in existing:
                continue
            # page_count gate: real filled forms are 1-20 pages typically;
            # huge depositions or briefs are not what we want
            pc = h.get("page_count") or 0
            if pc and pc > 30:
                continue
            if download_and_stage(
                h["url"], type_dir, fname,
                "CourtListener", f"CourtListener RECAP: {h['case']} ({h['docket']})",
            ):
                added += 1
            time.sleep(0.5)
    return added


# ============================================================================
# Category configs
# ============================================================================

CL_CATEGORIES = {
    "v1_eob": [
        '"explanation of benefits" insurance dispute exhibit',
        '"EOB" healthcare claim exhibit',
        '"explanation of benefits" Medicare summary',
        '"explanation of benefits" filed exhibit',
    ],
    "v1_bol_ocean": [
        '"ocean bill of lading" exhibit cargo',
        '"master bill of lading" Maersk exhibit',
        '"sea waybill" container shipment exhibit',
    ],
    "v1_irs_specialty": [
        '"Schedule K-1" Form 1065 exhibit tax court',
        '"Form 2848" power of attorney IRS exhibit',
    ],
    "v1_clinical_crf": [
        '"case report form" CRF clinical trial exhibit',
        '"case report form" filed exhibit Taxotere',
        '"case report form" Vioxx exhibit',
        '"adverse event" case report form filed',
    ],
    "v1_packing_list": [
        '"packing list" exhibit customs import',
        '"packing list" maritime cargo exhibit',
        '"packing list" filed import shipment',
    ],
    "v1_commercial_invoice": [
        '"commercial invoice" exhibit import customs',
        '"commercial invoice" filed CBP duty',
        '"commercial invoice" maritime exhibit',
    ],
    "v1_w2_1099": [
        '"W-2" exhibit tax court wage',
        '"1099-MISC" exhibit filed',
        '"Form 1099" exhibit petitioner',
        '"W-2 wage and tax statement" exhibit',
    ],
    "v1_bol_inland": [
        '"bill of lading" exhibit cargo damage',
        '"straight bill of lading" exhibit',
        '"inland bill of lading" trucking exhibit',
    ],
    "v1_air_waybill": [
        '"air waybill" exhibit cargo claim',
        '"AWB" air waybill exhibit IATA',
    ],
    "v1_form_5500": [
        '"Form 5500" exhibit ERISA plan',
        '"Form 5500" annual report filed ERISA',
    ],
    "v1_clinical_facesheet": [
        '"face sheet" hospital admission exhibit',
        '"patient face sheet" admission exhibit',
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=5, help="NEW PDFs per category")
    ap.add_argument("--type", action="append", default=None,
                    help="Only hunt this type (repeatable); default = all configured")
    ap.add_argument("--workers", type=int, default=3,
                    help="Parallel category workers (be polite to APIs)")
    ap.add_argument("--skip-fec", action="store_true")
    ap.add_argument("--skip-cl", action="store_true")
    args = ap.parse_args()

    only = set(args.type) if args.type else None

    plan: list[tuple[str, callable]] = []
    if not args.skip_fec and (not only or "v1_fec_form3" in only):
        plan.append(("v1_fec_form3", lambda: hunt_fec(args.target)))
    if not args.skip_fec and (not only or any(t in only for t in ["v1_fec_form5","v1_fec_form6","v1_fec_form7"])):
        # bundled hunter that does all three Form 5/6/7
        def _fec_other():
            r = hunt_fec_other_forms(target_per=args.target)
            print(f"  FEC other forms: {r}")
            return sum(r.values())
        plan.append(("v1_fec_other_forms", _fec_other))
    if not args.skip_cl:
        for cat, queries in CL_CATEGORIES.items():
            if only and cat not in only:
                continue
            plan.append((cat, (lambda c=cat, q=queries: hunt_courtlistener_category(c, q, args.target))))

    print(f"Hunting {len(plan)} categories (target={args.target} new PDFs each, workers={args.workers})\n")

    totals: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fn): name for name, fn in plan}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                n = fut.result()
                totals[name] = n
                print(f"  → {name}: {n} new")
            except Exception as e:
                totals[name] = 0
                print(f"  ! {name}: {e}")

    print("\n=== SUMMARY ===")
    for k, v in sorted(totals.items()):
        print(f"  {k}: +{v}")
    print(f"\nGRAND TOTAL: {sum(totals.values())} new PDFs added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
