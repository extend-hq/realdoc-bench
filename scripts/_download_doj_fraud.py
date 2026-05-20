#!/usr/bin/env python3
"""One-off: download 43 verified DOJ Criminal Fraud Section PPP/EIDL filings
with handwritten case-number/date/signature elements into
v1_doj_fraud_complaint/sources/ and append to manifest."""
import csv
from datetime import date as date_cls
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
import fitz

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets" / "extraction"
UA = "Extend realdoc-bench/0.1 (research; ameya@extend.app)"

TYPE = "v1_doj_fraud_complaint"

# (stem, url, notes) — from agent verified output
ENTRIES = [
    ("doj_2021_usa_v_yashica_bain_ppp_eidl_complaint_sdfl", "https://www.justice.gov/criminal/criminal-fraud/file/1380226/dl", "DOJ Criminal Fraud Section, U.S. v Yashica Bain, SDFL, 2021 - scanned AO-91 Complaint (PPP/EIDL, handwritten case numbers/dates/signatures)"),
    ("doj_2021_usa_v_shanrika_duhart_ppp_eidl_complaint_sdfl", "https://www.justice.gov/criminal/criminal-fraud/file/1380231/dl", "DOJ Criminal Fraud Section, U.S. v Shanrika Duhart, SDFL, 2021 - scanned AO-91 Complaint (PPP/EIDL)"),
    ("doj_2021_usa_v_jerrica_rosado_ppp_eidl_complaint_sdfl", "https://www.justice.gov/criminal/criminal-fraud/file/1380216/dl", "DOJ Criminal Fraud Section, U.S. v Jerrica Rosado, SDFL, 2021 - scanned AO-91 Complaint (PPP/EIDL)"),
    ("doj_2021_usa_v_devonte_thames_ppp_eidl_complaint_sdfl", "https://www.justice.gov/criminal/criminal-fraud/file/1380221/dl", "DOJ Criminal Fraud Section, U.S. v Devonte Thames, SDFL, 2021 - scanned AO-91 Complaint (PPP/EIDL)"),
    ("doj_2021_usa_v_dennes_garcia_ppp_eidl_complaint_sdfl", "https://www.justice.gov/criminal/criminal-fraud/file/1380241/dl", "DOJ Criminal Fraud Section, U.S. v Dennes Garcia, SDFL, 2021 - scanned AO-91 Complaint (PPP/EIDL)"),
    ("doj_2021_usa_v_cindi_denton_ppp_eidl_complaint_sdfl", "https://www.justice.gov/criminal/criminal-fraud/file/1380246/dl", "DOJ Criminal Fraud Section, U.S. v Cindi Denton, SDFL, 2021 - scanned AO-91 Complaint (PPP/EIDL)"),
    ("doj_2020_usa_v_keyaira_bostic_ppp_eidl_complaint_sdfl", "https://www.justice.gov/criminal/criminal-fraud/file/1315121/dl", "DOJ Criminal Fraud Section, U.S. v Keyaira Bostic, SDFL, 2020 - scanned AO-91 Complaint (PPP/EIDL)"),
    ("doj_2020_usa_v_tiara_walker_ppp_eidl_complaint_sdfl", "https://www.justice.gov/criminal/criminal-fraud/file/1315131/dl", "DOJ Criminal Fraud Section, U.S. v Tiara Walker, SDFL, 2020 - scanned AO-91 Complaint (PPP/EIDL)"),
    ("doj_2020_usa_v_andre_clark_ppp_eidl_complaint_sdfl", "https://www.justice.gov/criminal/criminal-fraud/file/1315111/dl", "DOJ Criminal Fraud Section, U.S. v Andre Clark, SDFL, 2020 - scanned AO-91 Complaint (PPP/EIDL)"),
    ("doj_2020_usa_v_damion_mckenzie_ppp_eidl_complaint_sdfl", "https://www.justice.gov/criminal/criminal-fraud/file/1315116/dl", "DOJ Criminal Fraud Section, U.S. v Damion Mckenzie, SDFL, 2020 - scanned AO-91 Complaint (PPP/EIDL)"),
    ("doj_2020_usa_v_carlos_belone_ppp_eidl_complaint_sdfl", "https://www.justice.gov/criminal/criminal-fraud/file/1315101/dl", "DOJ Criminal Fraud Section, U.S. v Carlos Belone, SDFL, 2020 - scanned AO-91 Complaint (PPP/EIDL)"),
    ("doj_2020_usa_v_dennis_nobbe_ppp_eidl_complaint_sdfl", "https://www.justice.gov/criminal/criminal-fraud/file/1315126/dl", "DOJ Criminal Fraud Section, U.S. v Dennis Nobbe, SDFL, 2020 - scanned AO-91 Complaint (PPP/EIDL)"),
    ("doj_2020_usa_v_matthew_jason_welch_ppp_indictment_dmt", "https://www.justice.gov/criminal/criminal-fraud/file/1315161/dl", "DOJ Criminal Fraud Section, U.S. v Matthew Jason Welch, District of Montana, 2020 - scanned AO-94 Indictment (PPP, HW case-number suffix, FILED stamp)"),
    ("doj_2020_usa_v_antonio_george_ppp_complaint_edmi", "https://www.justice.gov/criminal/criminal-fraud/file/1315171/dl", "DOJ Criminal Fraud Section, U.S. v Antonio George, EDMI, 2020 - scanned AO-91 Complaint (PPP)"),
    ("doj_2020_usa_v_elijah_majak_buoi_ppp_indictment_dma", "https://www.justice.gov/criminal/criminal-fraud/file/1315176/dl", "DOJ Criminal Fraud Section, U.S. v Elijah Majak Buoi, DMA, 2020 - scanned Indictment (PPP, HW signatures/stamps)"),
    ("doj_2020_usa_v_william_sadleir_ppp_complaint_cdca", "https://www.justice.gov/criminal/criminal-fraud/file/1315181/dl", "DOJ Criminal Fraud Section, U.S. v William Sadleir, CDCA, 2020 - scanned AO-91 Complaint (PPP)"),
    ("doj_2020_usa_v_shashank_shekhar_rai_ppp_complaint_edtx", "https://www.justice.gov/criminal/criminal-fraud/file/1315201/dl", "DOJ Criminal Fraud Section, U.S. v Shashank Shekhar Rai, EDTX, 2020 - scanned AO-91 Complaint (PPP)"),
    ("doj_2020_usa_v_fahad_h_shah_ppp_indictment_edtx", "https://www.justice.gov/criminal/criminal-fraud/file/1315216/dl", "DOJ Criminal Fraud Section, U.S. v Fahad H Shah, EDTX, 2020 - scanned Indictment (PPP, HW signatures/stamps)"),
    ("doj_2020_usa_v_jase_depaul_gautreaux_ppp_complaint_sdtx", "https://www.justice.gov/criminal/criminal-fraud/file/1315221/dl", "DOJ Criminal Fraud Section, U.S. v Jase DePaul Gautreaux, SDTX, 2020 - 18-page 100% scanned Complaint (PPP, HW elements)"),
    ("doj_2020_usa_v_joshua_thomas_argires_ppp_complaint_sdtx", "https://www.justice.gov/criminal/criminal-fraud/file/1315226/dl", "DOJ Criminal Fraud Section, U.S. v Joshua Thomas Argires, SDTX, 2020 - scanned AO-91 Complaint (PPP)"),
    ("doj_2020_usa_v_baoke_zhang_ppp_complaint_wdwa", "https://www.justice.gov/criminal/criminal-fraud/file/1315246/dl", "DOJ Criminal Fraud Section, U.S. v Baoke Zhang, WDWA, 2020 - 17-page 100% scanned Complaint (PPP)"),
    ("doj_2020_usa_v_brandon_casutt_ppp_complaint_dnv", "https://www.justice.gov/criminal/criminal-fraud/file/1315266/dl", "DOJ Criminal Fraud Section, U.S. v Brandon Casutt, DNV, 2020 - scanned AO-91 Complaint (PPP)"),
    ("doj_2020_usa_v_david_christopher_redfern_ppp_eidl_complaint_mdnc", "https://www.justice.gov/criminal/criminal-fraud/file/1315276/dl", "DOJ Criminal Fraud Section, U.S. v David Christopher Redfern, MDNC, 2020 - 12-page 100% scanned Complaint (PPP/EIDL)"),
    ("doj_2020_usa_v_benjamin_hayford_ppp_complaint_ndok", "https://www.justice.gov/criminal/criminal-fraud/file/1315281/dl", "DOJ Criminal Fraud Section, U.S. v Benjamin Hayford, NDOK, 2020 - 19-page 100% scanned Complaint (PPP)"),
    ("doj_2020_usa_v_lee_price_iii_ppp_eidl_complaint_sdtx", "https://www.justice.gov/criminal/criminal-fraud/file/1315361/dl", "DOJ Criminal Fraud Section, U.S. v Lee Price III, SDTX, 2020 - 20-page 100% scanned Complaint (PPP/EIDL, HW case number, FILED stamps, signatures)"),
    ("doj_2020_usa_v_joshua_bellamy_ppp_complaint_sdfl", "https://www.justice.gov/criminal/criminal-fraud/file/1315386/dl", "DOJ Criminal Fraud Section, U.S. v Joshua Bellamy, SDFL, 2020 - scanned AO-91 Complaint (PPP)"),
    ("doj_2020_usa_v_jonathan_markovich_ppp_complaint_sdfl", "https://www.justice.gov/criminal/criminal-fraud/file/1354336/dl", "DOJ Criminal Fraud Section, U.S. v Jonathan Markovich et al, SDFL, 2020 - 62-page multi-defendant Complaint (PPP, Compass Detox)"),
    ("doj_2020_usa_v_tonye_johnson_ppp_complaint_sdfl", "https://www.justice.gov/criminal/criminal-fraud/file/1354361/dl", "DOJ Criminal Fraud Section, U.S. v Tonye Johnson, SDFL, 2020 - scanned AO-91 Complaint (PPP)"),
    ("doj_2021_usa_v_izzat_freitekh_ppp_supersedingindictment_wdnc", "https://www.justice.gov/criminal/criminal-fraud/file/1354441/dl", "DOJ Criminal Fraud Section, U.S. v Izzat Freitekh, WDNC, 2021 - 23-page 100% scanned Superseding Indictment (PPP)"),
    ("doj_2021_usa_v_james_richard_stote_ppp_indictment_ndoh", "https://www.justice.gov/criminal/criminal-fraud/file/1354451/dl", "DOJ Criminal Fraud Section, U.S. v James Richard Stote et al, NDOH, 2021 - 20-page 100% scanned Indictment (PPP)"),
    ("doj_2020_usa_v_dinesh_sah_ppp_eidl_complaint_ndtx", "https://www.justice.gov/criminal/criminal-fraud/file/1354706/dl", "DOJ Criminal Fraud Section, U.S. v Dinesh Sah, NDTX, 2020 - scanned AO-91 Complaint (PPP/EIDL)"),
    ("doj_2020_usa_v_dinesh_sah_ppp_eidl_indictment_ndtx", "https://www.justice.gov/criminal/criminal-fraud/file/1354711/dl", "DOJ Criminal Fraud Section, U.S. v Dinesh Sah, NDTX, 2020 - 32-page 14.9MB 100% scanned Indictment (PPP/EIDL, HW dates, foreperson + judge signatures)"),
    ("doj_2021_usa_v_wook_chung_ppp_information_ndga", "https://www.justice.gov/criminal/criminal-fraud/file/1380196/dl", "DOJ Criminal Fraud Section, U.S. v Wook Chung, NDGA, 2021 - 10-page 100% scanned Information (PPP)"),
    ("doj_2021_usa_v_christian_johnson_ppp_eidl_complaint_wdny", "https://www.justice.gov/criminal/criminal-fraud/file/1380206/dl", "DOJ Criminal Fraud Section, U.S. v Christian Johnson, WDNY, 2021 - scanned Complaint (PPP/EIDL, mixed text+scanned)"),
    ("doj_2021_usa_v_lebnitz_tran_ppp_eidl_indictment_ndca", "https://www.justice.gov/criminal/criminal-fraud/file/1422011/dl", "DOJ Criminal Fraud Section, U.S. v Lebnitz Tran, NDCA, 2021 - 16-page 100% scanned Indictment (PPP/EIDL)"),
    ("doj_2021_usa_v_dixon_ppp_eidl_indictment_ndga", "https://www.justice.gov/criminal/criminal-fraud/file/1422076/dl", "DOJ Criminal Fraud Section, U.S. v Dixon et al, NDGA, 2021 - 82-page multi-defendant Indictment (PPP/EIDL)"),
    ("doj_2021_usa_v_arena_ppp_eidl_complaint_wdny", "https://www.justice.gov/criminal/criminal-fraud/file/1422081/dl", "DOJ Criminal Fraud Section, U.S. v Arena et al, WDNY, 2021 - 13-page 100% scanned Complaint (PPP/EIDL)"),
    ("doj_2021_usa_v_brandon_fitzgerald_holley_ppp_indictment_dmd", "https://www.justice.gov/criminal/criminal-fraud/file/1454706/dl", "DOJ Criminal Fraud Section, U.S. v Brandon Fitzgerald-Holley, DMD, 2021 - 12-page 100% scanned Indictment (PPP)"),
    ("doj_2021_usa_v_michael_tolliver_ppp_eidl_indictment_wdla", "https://www.justice.gov/criminal/criminal-fraud/file/1455841/dl", "DOJ Criminal Fraud Section, U.S. v Michael Tolliver, WDLA, 2021 - 15-page 100% scanned Indictment (PPP/EIDL)"),
    ("doj_2021_usa_v_martin_kao_ppp_indictment_dhi", "https://www.justice.gov/criminal/criminal-fraud/file/1392561/dl", "DOJ Criminal Fraud Section, U.S. v Martin Kao, DHI, 2021 - 20-page 100% scanned Indictment (PPP)"),
    ("doj_2021_usa_v_gregory_j_blotnick_ppp_eidl_complaint_dnj", "https://www.justice.gov/criminal/criminal-fraud/file/1397336/dl", "DOJ Criminal Fraud Section, U.S. v Gregory J Blotnick, DNJ, 2021 - scanned AO-91 Complaint (PPP/EIDL)"),
    ("doj_2021_usa_v_rhonda_thomas_ppp_eidl_complaint_dnj", "https://www.justice.gov/usao-nj/press-release/file/1395871/dl", "DOJ USAO NJ, U.S. v Rhonda Thomas, DNJ, 2021 - scanned AO-91 Complaint (PPP/EIDL, HW case number, ORIGINAL FILED stamp)"),
    ("doj_2022_usa_v_daniel_joseph_tisone_ppp_eidl_indictment_mdfl", "https://www.justice.gov/d9/press-releases/attachments/2022/04/22/20220422_tisone_indictment_0.pdf", "DOJ Office of Public Affairs press release attachment, U.S. v Daniel Joseph Tisone, MDFL, 2022 - 24-page Indictment (PPP/EIDL, HW case-number suffix and FILED stamp)"),
]


def download(entry):
    stem, url, notes = entry
    type_dir = DATASETS / TYPE
    sources = type_dir / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    fname = f"{stem}.pdf"
    dest = sources / fname
    if dest.exists():
        try:
            with fitz.open(dest) as d:
                if d.page_count >= 1 and dest.stat().st_size > 10_000:
                    return (entry, True, "cached", d.page_count)
        except Exception:
            pass
        dest.unlink(missing_ok=True)
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=120, stream=True, allow_redirects=True)
        if r.status_code != 200:
            return (entry, False, f"http {r.status_code}", 0)
        tmp = dest.with_suffix(".pdf.partial")
        with tmp.open("wb") as f:
            for c in r.iter_content(64 * 1024):
                f.write(c)
        with fitz.open(tmp) as d:
            pp = d.page_count
        if pp < 1 or tmp.stat().st_size < 5_000:
            tmp.unlink(missing_ok=True)
            return (entry, False, "invalid_pdf", 0)
        tmp.rename(dest)
        return (entry, True, "downloaded", pp)
    except Exception as e:
        return (entry, False, f"{type(e).__name__}: {str(e)[:80]}", 0)


# Run all downloads in parallel
results = []
with ThreadPoolExecutor(max_workers=10) as ex:
    futures = [ex.submit(download, e) for e in ENTRIES]
    for fut in as_completed(futures):
        results.append(fut.result())

# Update manifest
new_rows = []
for entry, ok, status, pp in results:
    stem, url, notes = entry
    if ok:
        new_rows.append({
            "local_filename": f"{stem}.pdf",
            "source_url": url,
            "source_site": urlparse(url).netloc,
            "is_filled": "yes",  # these are filed court docs — handwritten elements present
            "doc_pages": str(pp),
            "downloaded_on": date_cls.today().isoformat(),
            "notes": notes,
        })

fields = ["local_filename", "source_url", "source_site", "is_filled",
          "doc_pages", "downloaded_on", "notes"]
manifest = DATASETS / TYPE / "_sources_manifest.csv"
manifest.parent.mkdir(parents=True, exist_ok=True)
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
n_ok = sum(1 for r in results if r[1])
n_fail = sum(1 for r in results if not r[1])
n_pages = sum(r[3] for r in results if r[1])
print(f"\n=== summary: {n_ok} ok ({n_pages} pages) / {n_fail} fail ===")
for entry, ok, status, pp in results:
    if not ok:
        print(f"  X {entry[0]}: {status}")
print(f"\nmanifest updated: {manifest.relative_to(REPO)} ({len(existing)} total rows)")
