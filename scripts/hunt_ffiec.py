#!/usr/bin/env python3
"""Source FFIEC Bank Call Reports + HMDA disclosure statements.

Call Reports (FFIEC 031 / 041) = bank regulatory filings. Each is ~100-300 pages
of dense financial tables (Schedule RC-A through RC-S balance sheet, RI income
statement) plus checkbox-heavy attestation cover.

HMDA = mortgage application register, one huge table per institution-year.

FFIEC Central Data Repository (CDR) publishes filings publicly. Bank UBPR
(Uniform Bank Performance Reports) are the printable analog — they're PDFs
generated from the call report data and downloadable via the FFIEC website.

UBPR PDF URL pattern (works for any bank, public):
  https://cdr.ffiec.gov/Public/UBPR/UBPRReportConfig.aspx?rt=PDF&FIPID={cert_id}&PD={yyyymmdd}

(Use FDIC BankFind to look up cert_id: https://banks.data.fdic.gov/api/institutions)

HMDA Aggregate Reports (PDF format) at:
  https://ffiec.cfpb.gov/data-publication/aggregate-reports/{year}/

For now we focus on a curated set of banks + their UBPR + HMDA.
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
TIMEOUT = 120
MIN_PDF_BYTES = 30_000

# Banks and their FFIEC IDRSSD (registry id). Found via FFIEC NIC or FDIC BankFind.
# Note: UBPR uses cert_id (FDIC certificate number).
BANKS = [
    # (display_name, cert_id, idrssd)  — for UBPR PDF generation
    ("JPMorgan Chase Bank NA", "628", "852218"),
    ("Bank of America NA", "3510", "480228"),
    ("Wells Fargo Bank NA", "3511", "451965"),
    ("Citibank NA", "7213", "476810"),
    ("US Bank NA", "6548", "504713"),
    ("Truist Bank", "9846", "852320"),
    ("PNC Bank NA", "6384", "817824"),
    ("Goldman Sachs Bank USA", "33124", "2182786"),
    # Smaller community/regional banks for variety
    ("First Citizens Bank & Trust Co", "11063", "491224"),
    ("Silicon Valley Bridge Bank", "27", "5286084"),  # may not exist anymore
]


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


def slugify(s: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen]


def fetch_ubpr_pdf(idrssd: str, period_yyyymmdd: str) -> bytes | None:
    """Fetch the UBPR PDF for a bank at a given period (quarter-end date).
    Returns PDF bytes or None.
    UBPR URL: https://cdr.ffiec.gov/Public/UBPR/UBPRReportConfig.aspx?rt=PDF&FIPID={idrssd}&PD={yyyymmdd}"""
    url = (f"https://cdr.ffiec.gov/Public/UBPR/UBPRReportConfig.aspx?"
           f"rt=PDF&FIPID={idrssd}&PD={period_yyyymmdd}")
    r = safe_get(url)
    if r is None:
        return None
    ct = r.headers.get("content-type", "").lower()
    if "pdf" not in ct:
        return None
    return r.content


def hunt_ubpr_for_bank(bank_name: str, cert_id: str, idrssd: str, periods: list[str]) -> int:
    """Download UBPR for one bank across multiple quarter-ends. Returns count added."""
    type_dir = DATASETS / "v1_ffiec_call_report"
    sources_dir = type_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in sources_dir.glob("*.pdf")}
    added = 0
    for period in periods:
        fname = f"ubpr_{slugify(bank_name, 40)}_{period}.pdf"
        if fname in existing:
            continue
        data = fetch_ubpr_pdf(idrssd, period)
        if data is None:
            print(f"  ! ubpr {bank_name} {period}: fetch failed")
            continue
        dest = sources_dir / fname
        dest.write_bytes(data)
        ok, pp = validate_pdf(dest)
        if not ok:
            dest.unlink()
            print(f"  ! ubpr {bank_name} {period}: invalid PDF")
            continue
        write_manifest_row(
            type_dir, fname,
            f"https://cdr.ffiec.gov/Public/UBPR/UBPRReportConfig.aspx?rt=PDF&FIPID={idrssd}&PD={period}",
            "cdr.ffiec.gov", pp,
            notes=f"FFIEC UBPR for {bank_name} (cert {cert_id}, IDRSSD {idrssd}) period {period}",
        )
        added += 1
        print(f"  + v1_ffiec_call_report/{fname}  {pp}pp  {bank_name} {period}")
        time.sleep(0.3)  # be polite to FFIEC
    return added


def write_manifest_row(type_dir: Path, filename: str, source_url: str,
                       source_site: str, doc_pages: int, notes: str,
                       is_filled: str = "filled") -> None:
    manifest = type_dir / "_sources_manifest.csv"
    fields = ["local_filename", "source_url", "source_site", "is_filled",
              "doc_pages", "downloaded_on", "notes"]
    rows: list[dict] = []
    if manifest.exists():
        with manifest.open() as f:
            rows = list(csv.DictReader(f))
        if any(r.get("local_filename") == filename for r in rows):
            return
    rows.append({
        "local_filename": filename, "source_url": source_url,
        "source_site": source_site, "is_filled": is_filled,
        "doc_pages": str(doc_pages),
        "downloaded_on": date_cls.today().isoformat(),
        "notes": notes,
    })
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# HMDA Aggregate Reports — these are PDF disclosure statements per institution
# Source: https://ffiec.cfpb.gov/data-browser/data/{year}?category=states
# Aggregate national report PDF (table of all institutions): smaller, denser

HMDA_PDFS = [
    # (filename, url, notes)
    ("hmda_2023_aggregate_report",
     "https://ffiec.cfpb.gov/v2/data-browser-api/view/aggregations/aggregations.pdf?years=2023",
     "HMDA 2023 National Aggregate Report"),
    # FFIEC publishes per-MSA reports too; use a couple as additional dense-table samples
    ("hmda_2023_nyc_msa_aggregate",
     "https://ffiec.cfpb.gov/v2/data-browser-api/view/aggregations/aggregations.pdf?years=2023&msamds=35614",
     "HMDA 2023 NYC MSA Aggregate Report"),
    ("hmda_2023_la_msa_aggregate",
     "https://ffiec.cfpb.gov/v2/data-browser-api/view/aggregations/aggregations.pdf?years=2023&msamds=31084",
     "HMDA 2023 Los Angeles MSA Aggregate Report"),
]


def hunt_hmda() -> int:
    type_dir = DATASETS / "v1_ffiec_hmda"
    sources_dir = type_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in sources_dir.glob("*.pdf")}
    added = 0
    for stem, url, notes in HMDA_PDFS:
        fname = f"{stem}.pdf"
        if fname in existing:
            continue
        r = safe_get(url, stream=True)
        if r is None:
            continue
        dest = sources_dir / fname
        try:
            with dest.open("wb") as f:
                for chunk in r.iter_content(64 * 1024):
                    f.write(chunk)
            ok, pp = validate_pdf(dest)
            if not ok:
                dest.unlink()
                print(f"  ! hmda {fname}: invalid PDF")
                continue
            write_manifest_row(type_dir, fname, url, "ffiec.cfpb.gov", pp, notes=notes)
            added += 1
            print(f"  + v1_ffiec_hmda/{fname}  {pp}pp")
        except Exception as e:
            print(f"  ! hmda {fname}: {e}")
        time.sleep(0.3)
    return added


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3,
                    help="Polite parallelism (FFIEC servers are slow)")
    ap.add_argument("--periods", default="20241231,20240930",
                    help="Comma-sep quarter-end dates (YYYYMMDD) for UBPRs")
    ap.add_argument("--skip-ubpr", action="store_true")
    ap.add_argument("--skip-hmda", action="store_true")
    args = ap.parse_args()

    periods = args.periods.split(",")

    t0 = time.time()
    totals: dict[str, int] = {}

    if not args.skip_ubpr:
        print(f"\n=== UBPR Call Reports — {len(BANKS)} banks × {len(periods)} periods ===")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(hunt_ubpr_for_bank, name, cert, idrssd, periods): name
                       for name, cert, idrssd in BANKS}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    n = fut.result()
                    totals["v1_ffiec_call_report"] = totals.get("v1_ffiec_call_report", 0) + n
                except Exception as e:
                    print(f"  ! {name}: {e}")

    if not args.skip_hmda:
        print(f"\n=== HMDA Aggregate Reports ===")
        totals["v1_ffiec_hmda"] = hunt_hmda()

    print(f"\nDone in {time.time()-t0:.0f}s")
    for t, n in sorted(totals.items()):
        print(f"  {t}: +{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
