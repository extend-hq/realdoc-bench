#!/usr/bin/env python3
"""Generate chart / figure questions for pages that contain plots, graphs or
diagrams — the capability the existing structural taxonomy never covered.

For each candidate page image: one vision call decides whether the page has a
chart/figure and, if so, authors up to 3 exact-match questions about the
VISUAL's content (labelled data values, axis titles/units, legend categories,
chart title, figure caption/labels). Each gold is then re-verified by an
image-baseline call. Verbatim values only — no reading-off-axis estimation,
no computed/derived values.

Output: runs/<run>/qa/chart_questions.json  (appended to the bank afterward).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from pilot_50_gen_qa import (gemini_generate, encoded_image, load_env_file,  # noqa: E402
                             validate_question)

CHART_CATS = {"charts", "figures", "mixed"}
CHART_TYPES = {"v1_sec_10k", "v1_esg", "v1_budget_cafr", "v1_audit_report",
               "v1_budget_990", "v1_sec_def14a", "v1_sec_10q"}
CHART_CAPS = {"chart_value", "chart_axis", "chart_legend", "chart_title",
              "figure_content", "figure_caption"}

CHART_SYSTEM = (
    "You author exact-match benchmark questions that test whether a "
    "document-to-markdown parser captured the content of CHARTS and FIGURES "
    "(bar/line/pie charts, plots, diagrams, org charts, maps).\n\n"
    "First decide STRICTLY: does this page contain a genuine DATA "
    "VISUALIZATION — a bar chart, line graph, pie chart, scatter/area plot, "
    "gauge, or a real diagram (org chart, flowchart, process diagram, map)? "
    "A numbered list, a form section, a heading with sub-items, a plain "
    "table, or a block of body text is NOT a chart or figure — for those "
    'return {"has_visual": false, "questions": []}.\n\n'
    "If it does, author up to 3 questions about the VISUAL's content:\n"
    "  - a data value PRINTED on the chart (a bar/point/slice data label)\n"
    "  - an axis title, axis unit, or tick label\n"
    "  - a legend category / series name\n"
    "  - the chart title or the figure's printed caption\n"
    "  - a label printed inside a diagram\n\n"
    "MANDATORY RULES:\n"
    "  1. VERBATIM ONLY. The gold must be text/numbers PRINTED on the chart "
    "or figure. NEVER ask for a value the reader must estimate by reading "
    "against an axis, and NEVER a sum/difference/total/derived value.\n"
    "  2. FLAT GOLD. gold_answer is a plain 'key=value; key=value' string. "
    "For grouped data use dotted compound keys (e.g. 'REVENUE 2023.value=..'). "
    "No braces, no nesting. Use 'null' for blanks.\n"
    "  3. ANCHOR by the chart title / axis label / legend text — never by "
    "visual position ('top-left').\n"
    "  4. response_format must spell out every key and value form.\n\n"
    "Return JSON: {\"has_visual\": <bool>, \"questions\": [{\"question\": ..., "
    "\"response_format\": ..., \"gold_answer\": ..., \"capabilities\": [..]}]}\n"
    f"Use only these capability tags: {sorted(CHART_CAPS)}"
)


def gen_for_page(image_path: Path, source_file: str) -> list[dict]:
    parts = [
        {"text": "Inspect this document page for charts / figures and author "
                 "chart-content questions per the rules."},
        {"inline_data": {"mime_type": "image/png", "data": encoded_image(image_path)}},
    ]
    try:
        payload = gemini_generate(parts, system=CHART_SYSTEM, temperature=0.2)
    except Exception as exc:
        print(f"  ! {source_file}: {exc}", flush=True)
        return []
    if not isinstance(payload, dict) or not payload.get("has_visual"):
        return []
    out = []
    for q in payload.get("questions", []) or []:
        if not isinstance(q, dict) or not q.get("gold_answer"):
            continue
        caps = [c for c in (q.get("capabilities") or []) if c in CHART_CAPS]
        q["capabilities"] = caps or ["chart_value"]
        q["source_file"] = source_file
        q["answer_type"] = "chart_figure_exact"
        out.append(q)
    return out


def main() -> int:
    run = REPO / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "v500")
    load_env_file(REPO / ".env")
    rows = list(csv.DictReader((run / "sample.csv").open()))
    cands = {}
    for r in rows:
        if r.get("category") in CHART_CATS or r.get("type") in CHART_TYPES:
            cands[r["dest_path"]] = r
    pages = [(dp, r, run / "images" / f"{Path(dp).stem}.png")
             for dp, r in cands.items()]
    pages = [(dp, r, img) for dp, r, img in pages if img.exists()]
    print(f"chart/figure candidate pages: {len(pages)}", flush=True)

    t0 = time.time()
    raw: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(gen_for_page, img, Path(dp).stem): dp
                for dp, r, img in pages}
        for i, fut in enumerate(as_completed(futs), 1):
            raw.extend(fut.result())
            if i % 20 == 0 or i == len(pages):
                print(f"  [{i}/{len(pages)}] proposed {len(raw)} questions "
                      f"({(time.time()-t0)/60:.1f}min)", flush=True)
    print(f"proposed {len(raw)} chart/figure questions — validating golds...", flush=True)

    # validate each gold against an image-only baseline; keep if it agrees
    img_by_src = {Path(dp).stem: img for dp, r, img in pages}
    kept: list[dict] = []
    n = 0

    def validate(q):
        img = img_by_src.get(q["source_file"])
        if not img:
            return None
        try:
            v = validate_question(img, q)
        except Exception:
            return None
        if v.get("image_matches_gold"):
            return q
        if v.get("image_answer"):                 # auto-fix gold from image
            q = {**q, "gold_answer": v["image_answer"]}
            return q
        return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(validate, q) for q in raw]):
            r = fut.result()
            n += 1
            if r:
                kept.append(r)
            if n % 25 == 0 or n == len(raw):
                print(f"  validated [{n}/{len(raw)}] kept={len(kept)}", flush=True)

    # assign question ids
    for i, q in enumerate(sorted(kept, key=lambda x: x["source_file"]), 1):
        cap = q["capabilities"][0]
        q["question_id"] = f"v500_fig_q{i:03d}_{cap}"
        q["difficulty"] = "hard_structural"

    out = run / "qa" / "chart_questions.json"
    out.write_text(json.dumps({"items": kept}, indent=2, ensure_ascii=False))
    print(f"\n{len(kept)} validated chart/figure questions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
