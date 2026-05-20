#!/usr/bin/env python3
"""One-off: download 39 newly-found intl tax PDFs into v1_intl_tax_*/sources/
and append to each manifest."""
import csv
import os
import re
import sys
import time
from datetime import date as date_cls
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
import fitz

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets" / "extraction"
UA = "Extend realdoc-bench/0.1 (research; ameya@extend.app)"

# Pulled from agent: country|stem|url|description
ENTRIES = [
    # UK
    ("v1_intl_tax_uk", "hmrc_sa100_2026", "https://assets.publishing.service.gov.uk/media/69c14d07cfa346b9d4704a8d/SA100-2026.pdf", "UK HMRC SA100 Self Assessment tax return (2026)"),
    ("v1_intl_tax_uk", "hmrc_sa150_notes_2026", "https://assets.publishing.service.gov.uk/media/69c14ce2d588c92c483e4ba6/SA150-Notes-2026.pdf", "UK HMRC SA150 How to fill in your tax return notes (2026)"),
    ("v1_intl_tax_uk", "hmrc_sa101_2026", "https://assets.publishing.service.gov.uk/media/69c25b9ab920af63be1c770c/SA101_2026.pdf", "UK HMRC SA101 Additional information (2026)"),
    ("v1_intl_tax_uk", "hmrc_sa101_2025", "https://assets.publishing.service.gov.uk/media/67dc1e9fc5528de3aa6711e6/SA101-2025.pdf", "UK HMRC SA101 Additional information (2025)"),
    ("v1_intl_tax_uk", "hmrc_sa101_2024", "https://assets.publishing.service.gov.uk/media/660152aea6c0f7bb15ef9150/SA101_2024.pdf", "UK HMRC SA101 Additional information (2024)"),
    ("v1_intl_tax_uk", "hmrc_sa105_2026", "https://assets.publishing.service.gov.uk/media/69cd19d6eafd66b876458ba9/SA105_2026_v0.1.pdf", "UK HMRC SA105 UK Property (2026)"),
    ("v1_intl_tax_uk", "hmrc_sa105_2025", "https://assets.publishing.service.gov.uk/media/67ed5069e9c76fa33048c6b0/sa105_2025.pdf", "UK HMRC SA105 UK Property (2025)"),
    ("v1_intl_tax_uk", "hmrc_sa105_2024", "https://assets.publishing.service.gov.uk/media/65f9777e703c42001a58ef11/SA105_2024.pdf", "UK HMRC SA105 UK Property (2024)"),
    ("v1_intl_tax_uk", "hmrc_ct600_2026", "https://assets.publishing.service.gov.uk/media/69c543424a06660f085442bd/ct600.pdf", "UK HMRC CT600 Company Tax Return (2026 v3)"),
    ("v1_intl_tax_uk", "hmrc_p11d_ws1_2024_2025", "https://assets.publishing.service.gov.uk/media/67cffcb874b001c38a028795/P11D-WS1-2024-2025.pdf", "UK HMRC P11D Working Sheet 1 (Living accommodation)"),
    ("v1_intl_tax_uk", "hmrc_p11d_ws2_2024_2025", "https://assets.publishing.service.gov.uk/media/67654ba54e2d5e9c0bde9c5a/p11d-ws2-car-and-car-fuel-benefit-2024-2025.pdf", "UK HMRC P11D Working Sheet 2 (Car/fuel benefit)"),
    # Australia
    ("v1_intl_tax_au", "ato_tax_return_individuals_2024", "https://iorder.com.au/publication/Download.aspx?ProdID=2541-6.2024", "Australia ATO Tax return for individuals 2024 (NAT 2541)"),
    ("v1_intl_tax_au", "ato_tax_return_individuals_2025", "https://iorder.com.au/publication/Download.aspx?ProdID=2541-6.2025", "Australia ATO Tax return for individuals 2025 (NAT 2541)"),
    ("v1_intl_tax_au", "ato_tax_return_individuals_agents_2025", "https://iorder.com.au/publication/Download.aspx?ProdID=1371-6.2025", "Australia ATO Individual tax return for tax agents 2025 (NAT 1371)"),
    ("v1_intl_tax_au", "ato_bas_sample", "https://bas-i.com.au/ato/samplebasform.pdf", "Australia ATO BAS sample (third-party hosted)"),
    # India
    ("v1_intl_tax_in", "cbdt_itr1_sahaj_ay2023_24", "https://www.incometax.gov.in/iec/foportal/sites/default/files/2023-04/ITR1_Notified%20Form%20AY%202023-24.pdf", "India CBDT ITR-1 Sahaj AY 2023-24"),
    ("v1_intl_tax_in", "cbdt_itr2_ay2023_24", "https://www.incometax.gov.in/iec/foportal/sites/default/files/2023-04/ITR2_Notified%20Form%20AY%202023-24.pdf", "India CBDT ITR-2 AY 2023-24"),
    ("v1_intl_tax_in", "cbdt_itr3_ay2023_24", "https://www.incometax.gov.in/iec/foportal/sites/default/files/2023-04/ITR3_Notified%20Form%20AY%202023-24.pdf", "India CBDT ITR-3 AY 2023-24"),
    ("v1_intl_tax_in", "cbdt_itr4_sugam_ay2023_24", "https://www.incometax.gov.in/iec/foportal/sites/default/files/2023-04/ITR4_Notified%20Form%20AY%202023-24.pdf", "India CBDT ITR-4 Sugam AY 2023-24"),
    ("v1_intl_tax_in", "cbdt_itr1_sahaj_ay2022_23", "https://www.incometax.gov.in/iec/foportal/sites/default/files/2022-04/ITR%201%20Notified%20Form_AY%202022-23.pdf", "India CBDT ITR-1 Sahaj AY 2022-23"),
    ("v1_intl_tax_in", "cbdt_itr4_sugam_ay2022_23", "https://www.incometax.gov.in/iec/foportal/sites/default/files/2022-04/ITR%204%20Notified%20Form_AY%202022-23.pdf", "India CBDT ITR-4 Sugam AY 2022-23"),
    # New Zealand
    ("v1_intl_tax_nz", "ird_ir3_2024", "https://www.ird.govt.nz/-/media/project/ir/home/documents/forms-and-guides/ir1---ir99/ir3/ir3-2024.pdf", "NZ IRD IR3 Individual income tax return 2024"),
    ("v1_intl_tax_nz", "ird_ir3_2025", "https://www.ird.govt.nz/-/media/project/ir/home/documents/forms-and-guides/ir1---ir99/ir3/ir3-2025.pdf", "NZ IRD IR3 Individual income tax return 2025"),
    ("v1_intl_tax_nz", "ird_ir3_2023", "https://www.ird.govt.nz/-/media/project/ir/home/documents/forms-and-guides/ir1---ir99/ir3/ir3-2023.pdf", "NZ IRD IR3 Individual income tax return 2023"),
    ("v1_intl_tax_nz", "ird_ir4_2024", "https://www.ird.govt.nz/-/media/project/ir/home/documents/forms-and-guides/ir1---ir99/ir4/ir4-2024.pdf", "NZ IRD IR4 Companies income tax return 2024"),
    ("v1_intl_tax_nz", "ird_ir4_2025", "https://www.ird.govt.nz/-/media/project/ir/home/documents/forms-and-guides/ir1---ir99/ir4/ir4-2025.pdf", "NZ IRD IR4 Companies income tax return 2025"),
    # Singapore
    ("v1_intl_tax_sg", "iras_formbb1_intro", "https://file.go.gov.sg/formbb1-intro.pdf", "Singapore IRAS Tax Filing intro: Form B/B1"),
    ("v1_intl_tax_sg", "iras_formp_howtofile", "https://file.go.gov.sg/formp-howtofile.pdf", "Singapore IRAS How to file Form P (partnership)"),
    ("v1_intl_tax_sg", "iras_form_b_guide", "https://www.iras.gov.sg/docs/default-source/archive/form-b-guide-final-(english)-ext.pdf?sfvrsn=60fc105b_20", "Singapore IRAS Form B filing guide (self-employed)"),
    ("v1_intl_tax_sg", "iras_form_b1_guide", "https://www.iras.gov.sg/media/docs/default-source/uploadedfiles/pdf/guide-to-form-b1-(english)51bd.pdf?sfvrsn=5c44ab37_2", "Singapore IRAS Form B1 filing guide (tax residents)"),
]


def download(entry):
    type_name, stem, url, notes = entry
    type_dir = DATASETS / type_name
    sources = type_dir / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    fname = f"{stem}.pdf"
    dest = sources / fname
    if dest.exists():
        try:
            with fitz.open(dest) as d:
                if d.page_count >= 1 and dest.stat().st_size > 20_000:
                    return (entry, True, "cached", d.page_count)
        except Exception:
            pass
        dest.unlink(missing_ok=True)
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=90, stream=True, allow_redirects=True)
        if r.status_code != 200:
            return (entry, False, f"http {r.status_code}", 0)
        tmp = dest.with_suffix(".pdf.partial")
        with tmp.open("wb") as f:
            for c in r.iter_content(64 * 1024):
                f.write(c)
        with fitz.open(tmp) as d:
            pp = d.page_count
        if pp < 1 or tmp.stat().st_size < 10_000:
            tmp.unlink(missing_ok=True)
            return (entry, False, "invalid_pdf", 0)
        tmp.rename(dest)
        return (entry, True, "downloaded", pp)
    except Exception as e:
        return (entry, False, f"{type(e).__name__}: {str(e)[:80]}", 0)


# Run all downloads in parallel
results = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(download, e) for e in ENTRIES]
    for fut in as_completed(futures):
        results.append(fut.result())

# Update manifests per type
from collections import defaultdict
by_type = defaultdict(list)
for entry, ok, status, pp in results:
    type_name, stem, url, notes = entry
    if ok:
        by_type[type_name].append({
            "local_filename": f"{stem}.pdf",
            "source_url": url,
            "source_site": urlparse(url).netloc,
            "is_filled": "blank",
            "doc_pages": str(pp),
            "downloaded_on": date_cls.today().isoformat(),
            "notes": notes,
        })

fields = ["local_filename", "source_url", "source_site", "is_filled",
          "doc_pages", "downloaded_on", "notes"]
for type_name, new_rows in by_type.items():
    manifest = DATASETS / type_name / "_sources_manifest.csv"
    existing = []
    if manifest.exists():
        with manifest.open() as f:
            existing = list(csv.DictReader(f))
    have = {r["local_filename"] for r in existing}
    for nr in new_rows:
        if nr["local_filename"] not in have:
            existing.append(nr)
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in existing:
            w.writerow(r)

# Report
print(f"\n=== summary: {sum(1 for r in results if r[1])} ok / {sum(1 for r in results if not r[1])} fail ===")
type_counts = defaultdict(lambda: [0,0,0])  # [ok, fail, pages]
for entry, ok, status, pp in results:
    type_name = entry[0]
    if ok:
        type_counts[type_name][0] += 1
        type_counts[type_name][2] += pp
    else:
        type_counts[type_name][1] += 1
        print(f"  ✗ {type_name}/{entry[1]}: {status}")
print()
for t, (o, f, pp) in sorted(type_counts.items()):
    print(f"  {t:25} {o:>2} ok ({pp:>3} pages), {f} fail")
