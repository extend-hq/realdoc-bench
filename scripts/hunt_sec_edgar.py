#!/usr/bin/env python3
"""Source SEC filings (10-K, 13F, DEF 14A, Form ADV, Form 3/4/5) from EDGAR.

Uses SEC's public submissions API (free, no auth — requires User-Agent header
with contact info per SEC fair-access rules).

For each company + form type:
  1. GET https://data.sec.gov/submissions/CIK{cik:010d}.json  → list of filings
  2. For recent filings of the requested type, follow accession → primary doc
  3. Download as PDF (filings are often HTML; convert via wkhtmltopdf if needed,
     OR pull the PDF exhibit directly when available)
  4. Validate + save to datasets/extraction/v1_<type>/sources/

Idempotent — skip if PDF already exists.

Doc types produced:
  v1_sec_10k        — annual reports (12 companies)
  v1_sec_13f        — institutional holdings (5 filers × 2 quarters)
  v1_sec_def14a     — proxy statements (6 companies)
  v1_sec_form_adv   — investment adviser Form ADV (6 firms)
  v1_sec_form_3_4_5 — insider transactions (10 across major companies)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests

try:
    import fitz
except ImportError:
    print("pymupdf not installed", file=sys.stderr)
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets" / "extraction"

# SEC requires a descriptive User-Agent with contact info
UA = "Extend realdoc-bench ameya@extend.app"
HTTP_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.5",
}
TIMEOUT = 90
MIN_PDF_BYTES = 30_000

# Chrome binary for headless HTML → PDF (modern SEC filings are iXBRL HTML)
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


# ─────────────────────────────────────────────────────────────────────────────
# Targets: company CIK + form types per type folder
# ─────────────────────────────────────────────────────────────────────────────

# CIK is the SEC company id (no leading zeros in url, but 10-digit zero-padded in submissions API)
COMPANIES = {
    "apple":          "0000320193",
    "microsoft":      "0000789019",
    "alphabet_a":     "0001652044",
    "amazon":         "0001018724",
    "meta":           "0001326801",
    "nvidia":         "0001045810",
    "bofa":           "0000070858",
    "wells_fargo":    "0000072971",
    "walmart":        "0000104169",
    "pfizer":         "0000078003",
    "tesla":          "0001318605",
    "berkshire":      "0001067983",
    "jpm":            "0000019617",
    "blackrock":      "0001364742",
    "renaissance":    "0001037389",
    "bridgewater":    "0001350694",
    "vanguard":       "0000102909",  # Vanguard Specialized Funds; one of several Vanguard CIKs
    # Foreign issuers (20-F instead of 10-K)
    "toyota":         "0001094517",
    "hsbc":           "0001089113",
}

# (target_doc_type, form_type_on_sec, [(company_key, n_filings), ...])
TARGETS: list[tuple[str, list[str], list[tuple[str, int]]]] = [
    # 10-K + 20-F = annual reports
    ("v1_sec_10k", ["10-K", "20-F"], [
        ("apple", 1), ("microsoft", 1), ("alphabet_a", 1), ("amazon", 1),
        ("meta", 1), ("nvidia", 1), ("bofa", 1), ("walmart", 1),
        ("pfizer", 1), ("tesla", 1),
        ("toyota", 1), ("hsbc", 1),  # 20-F foreign issuers
    ]),
    # 13F — institutional holdings, big sparse tables
    ("v1_sec_13f", ["13F-HR"], [
        ("berkshire", 2), ("blackrock", 2), ("renaissance", 2),
        ("bridgewater", 1), ("vanguard", 1),
    ]),
    # DEF 14A — proxy statements, comp tables
    ("v1_sec_def14a", ["DEF 14A"], [
        ("apple", 1), ("microsoft", 1), ("jpm", 1), ("bofa", 1),
        ("meta", 1), ("tesla", 1),
    ]),
    # Form 4 — insider transactions, checkbox-heavy
    ("v1_sec_form_3_4_5", ["4"], [
        ("apple", 3), ("tesla", 3), ("meta", 2), ("microsoft", 2),
    ]),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_get(url: str, **kw) -> requests.Response | None:
    try:
        r = requests.get(url, timeout=TIMEOUT, headers=HTTP_HEADERS, **kw)
        if r.status_code == 200:
            return r
        print(f"  ! GET {url[:80]} → {r.status_code}")
        return None
    except Exception as e:
        print(f"  ! GET {url[:80]} failed: {e}")
        return None


def render_html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Use Chrome headless to render an HTML file (iXBRL or plain) to PDF."""
    import subprocess
    if not Path(CHROME_BIN).exists():
        print(f"  ! Chrome not at {CHROME_BIN}", file=sys.stderr)
        return False
    cmd = [
        CHROME_BIN, "--headless", "--disable-gpu",
        "--no-pdf-header-footer",
        "--no-sandbox",  # required in some CI envs
        f"--print-to-pdf={pdf_path}",
        f"file://{html_path}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode != 0:
            print(f"  ! Chrome err: {proc.stderr.decode()[:200]}")
            return False
        return pdf_path.exists() and pdf_path.stat().st_size > MIN_PDF_BYTES
    except subprocess.TimeoutExpired:
        print(f"  ! Chrome timeout on {html_path.name}")
        return False
    except Exception as e:
        print(f"  ! Chrome failed: {e}")
        return False


def slugify(s: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen]


def validate_pdf(path: Path) -> tuple[bool, int]:
    try:
        if path.stat().st_size < MIN_PDF_BYTES:
            return False, 0
        with fitz.open(path) as doc:
            return doc.page_count >= 1, doc.page_count
    except Exception:
        return False, 0


def fetch_submissions(cik: str) -> dict | None:
    """Return SEC's submissions JSON for the company. cik is zero-padded 10-digit."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = safe_get(url)
    if r is None:
        return None
    try:
        return r.json()
    except Exception:
        return None


def list_recent_filings(submissions: dict, form_types: list[str], n: int) -> list[dict]:
    """Return up to n most-recent filings of the given form types."""
    if "filings" not in submissions:
        return []
    rec = submissions["filings"].get("recent") or {}
    forms = rec.get("form") or []
    accessions = rec.get("accessionNumber") or []
    primary_docs = rec.get("primaryDocument") or []
    dates = rec.get("filingDate") or []
    primary_doc_descriptions = rec.get("primaryDocDescription") or []
    out = []
    for i, f in enumerate(forms):
        if f in form_types and i < len(accessions):
            out.append({
                "form": f,
                "accession": accessions[i],
                "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
                "date": dates[i] if i < len(dates) else "",
                "primary_desc": primary_doc_descriptions[i] if i < len(primary_doc_descriptions) else "",
            })
        if len(out) >= n:
            break
    return out


def build_filing_archive_url(cik: str, accession: str) -> str:
    """Map accession to the EDGAR archives folder URL (no leading zeros, no dashes)."""
    cik_int = str(int(cik))
    acc_no_dashes = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_dashes}/"


def primary_doc_url(archive_url: str, primary_doc: str) -> str | None:
    """Resolve the primary document filename to its full URL within the archive."""
    if not primary_doc:
        return None
    return archive_url.rstrip("/") + "/" + primary_doc.lstrip("/")


def fetch_html_and_render_pdf(html_url: str, dest_pdf: Path, work_dir: Path) -> tuple[bool, int]:
    """Download iXBRL/HTML primary doc, render to PDF via Chrome headless,
    validate. Returns (ok, page_count)."""
    if dest_pdf.exists():
        ok, pp = validate_pdf(dest_pdf)
        if ok:
            return True, pp
        dest_pdf.unlink()

    r = safe_get(html_url)
    if r is None:
        return False, 0

    # Save HTML to a temp file inside the work_dir (so Chrome can resolve local file://)
    work_dir.mkdir(parents=True, exist_ok=True)
    html_file = work_dir / (dest_pdf.stem + ".htm")
    # Some files are binary-ish (gzip); use r.content + decode safely
    html_file.write_bytes(r.content)

    # Render with Chrome
    ok = render_html_to_pdf(html_file.resolve(), dest_pdf)
    if not ok:
        return False, 0
    valid, pp = validate_pdf(dest_pdf)
    if not valid:
        dest_pdf.unlink(missing_ok=True)
        return False, 0
    # Cleanup intermediate HTML
    try:
        html_file.unlink()
    except OSError:
        pass
    return True, pp


def write_manifest_row(type_dir: Path, filename: str, source_url: str,
                       source_site: str, doc_pages: int, notes: str,
                       is_filled: str = "filled") -> None:
    """Append to _sources_manifest.csv using the realdoc-bench 7-col schema."""
    from datetime import date as date_cls
    manifest = type_dir / "_sources_manifest.csv"
    fields = ["local_filename", "source_url", "source_site", "is_filled",
              "doc_pages", "downloaded_on", "notes"]
    rows: list[dict] = []
    if manifest.exists():
        with manifest.open() as f:
            rdr = csv.DictReader(f)
            rows = list(rdr)
        if any(r.get("local_filename") == filename for r in rows):
            return
    rows.append({
        "local_filename": filename,
        "source_url": source_url,
        "source_site": source_site,
        "is_filled": is_filled,
        "doc_pages": str(doc_pages),
        "downloaded_on": date_cls.today().isoformat(),
        "notes": notes,
    })
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Per-filing worker
# ─────────────────────────────────────────────────────────────────────────────

def hunt_one_company(target_type: str, form_types: list[str], company_key: str, n: int) -> int:
    """Find + render up to n filings for one company. Returns count added."""
    cik = COMPANIES.get(company_key)
    if not cik:
        print(f"  ! no CIK for {company_key}")
        return 0
    type_dir = DATASETS / target_type
    sources_dir = type_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    work_dir = type_dir / "_html_work"
    existing = {p.name for p in sources_dir.glob("*.pdf")}

    submissions = fetch_submissions(cik)
    if submissions is None:
        return 0
    name = submissions.get("name", company_key)
    filings = list_recent_filings(submissions, form_types, n + 1)

    added = 0
    for filing in filings:
        if added >= n:
            break
        accession = filing["accession"]
        date_str = filing["date"]
        form = filing["form"]
        primary_doc = filing.get("primary_doc")
        if not primary_doc:
            continue
        archive_url = build_filing_archive_url(cik, accession)
        html_url = primary_doc_url(archive_url, primary_doc)
        if html_url is None:
            continue

        slug_co = slugify(name, 40)
        slug_form = slugify(form, 12)
        fname = f"sec_{slug_co}_{slug_form}_{date_str}.pdf"
        if fname in existing:
            continue
        dest = sources_dir / fname

        ok, pp = fetch_html_and_render_pdf(html_url, dest, work_dir)
        if not ok:
            print(f"  ! {target_type}/{fname}: render failed", flush=True)
            continue
        write_manifest_row(
            type_dir, fname, html_url,
            "sec.gov",
            pp,
            notes=f"SEC EDGAR {form} for {name} filed {date_str}; accession {accession}; rendered from iXBRL HTML via Chrome headless",
        )
        added += 1
        print(f"  + {target_type}/{fname}  {pp}pp  {form} {name}", flush=True)
        time.sleep(0.15)  # polite to SEC (10 req/sec max)
    # Clean up work dir if empty
    if work_dir.exists() and not any(work_dir.iterdir()):
        work_dir.rmdir()
    return added


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel company fetchers (be polite — SEC enforces rate limit)")
    ap.add_argument("--target-type", action="append",
                    help="Restrict to one or more target types (e.g. v1_sec_10k)")
    args = ap.parse_args()

    plan: list[tuple[str, list[str], str, int]] = []
    for tt, fts, co_list in TARGETS:
        if args.target_type and tt not in args.target_type:
            continue
        for ck, n in co_list:
            plan.append((tt, fts, ck, n))

    print(f"SEC EDGAR hunt: {len(plan)} (company × type) jobs")
    for tt, fts, ck, n in plan[:6]:
        print(f"  {tt:20} {ck:14} {fts} × {n}")
    if len(plan) > 6:
        print(f"  ... ({len(plan) - 6} more)")

    t0 = time.time()
    totals: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(hunt_one_company, *args_): args_ for args_ in plan}
        for fut in as_completed(futures):
            tt, fts, ck, n = futures[fut]
            try:
                got = fut.result()
                totals[tt] = totals.get(tt, 0) + got
            except Exception as e:
                print(f"  ! {tt} {ck} crashed: {e}")

    print(f"\nDone in {time.time()-t0:.0f}s")
    for tt, n in sorted(totals.items()):
        print(f"  {tt}: +{n} PDFs")
    print(f"  GRAND TOTAL: {sum(totals.values())} new PDFs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
