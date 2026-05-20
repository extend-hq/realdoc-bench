#!/usr/bin/env python3
"""Source genuine chart pages from v1.2_final (ESG / 10-K / CAFR / audit) that
are NOT already in v500. Each candidate page is rendered, then a vision call
confirms it holds a real data visualization; confirmed pages are appended to
runs/v500/sample.csv (category=charts) for the parse + chart-QA steps.
"""
from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from pilot_50_gen_qa import gemini_generate, encoded_image, load_env_file  # noqa: E402

V12F = REPO / "datasets" / "extraction" / "v1.2_final"
RUN = REPO / "runs" / "v500"
CHART_TYPES = {"v1_esg", "v1_sec_10k", "v1_sec_10q", "v1_sec_def14a",
               "v1_budget_cafr", "v1_audit_report", "v1_budget_990"}
CAP = 60

DETECT_SYSTEM = (
    "Look at this document page. Decide ONLY whether it contains a genuine "
    "DATA VISUALIZATION: a bar / line / pie / area / scatter chart, a plotted "
    "graph, a gauge, or a real diagram (org chart, flowchart, process diagram, "
    "map). A plain table, a form, a numbered list, or a block of body text is "
    'NOT one. Return {"has_chart": true|false}.'
)


def render(pdf: Path, out_png: Path) -> None:
    import pymupdf
    d = pymupdf.open(str(pdf))
    try:
        d[0].get_pixmap(matrix=pymupdf.Matrix(150 / 72, 150 / 72)).save(str(out_png))
    finally:
        d.close()


def main() -> int:
    load_env_file(REPO / ".env")
    in_v500 = {r["dest_path"] for r in csv.DictReader((RUN / "sample.csv").open())}
    manifest = list(csv.DictReader((V12F / "_manifest.csv").open()))
    cands = [r for r in manifest
             if r.get("type") in CHART_TYPES and r["dest_path"] not in in_v500]
    print(f"{len(cands)} candidate chart-type pages not already in v500", flush=True)

    imgdir = RUN / "images"
    imgdir.mkdir(exist_ok=True)

    def scan(r: dict):
        dp = r["dest_path"]
        pdf = V12F / dp
        if not pdf.exists():
            return None
        png = imgdir / f"{Path(dp).stem}.png"
        rendered_here = not png.exists()
        try:
            if rendered_here:
                render(pdf, png)
            res = gemini_generate(
                [{"text": "Does this page hold a genuine data visualization?"},
                 {"inline_data": {"mime_type": "image/png",
                                  "data": encoded_image(png)}}],
                system=DETECT_SYSTEM, temperature=0.0)
        except Exception:
            return None
        if isinstance(res, dict) and res.get("has_chart"):
            return r
        if rendered_here and png.exists():
            png.unlink()                          # not a chart — drop the render
        return None

    t0 = time.time()
    found: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(scan, r) for r in cands]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                found.append(r)
            if i % 40 == 0 or i == len(cands):
                print(f"  [{i}/{len(cands)}] charts found: {len(found)} "
                      f"({(time.time()-t0)/60:.1f}min)", flush=True)

    # cap; drop renders for the excess
    excess = found[CAP:]
    found = found[:CAP]
    for r in excess:
        p = imgdir / f"{Path(r['dest_path']).stem}.png"
        if p.exists():
            p.unlink()

    sample_path = RUN / "sample.csv"
    cols = csv.DictReader(sample_path.open()).fieldnames
    with sample_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        for r in found:
            row = {c: r.get(c, "") for c in cols}
            row["stratum"] = "chart_addition"
            row["category"] = "charts"
            w.writerow(row)
    print(f"\nappended {len(found)} confirmed chart pages to {sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
