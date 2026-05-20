#!/usr/bin/env python3
"""Quick finalization check for a run's question bank.

Reports doc/question counts, type distribution, structural integrity (missing
gold/image, blank-page or corrupt-gold leakage), per-parser accuracy delta,
and a small sample of atomic questions for eyeballing.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_BUCKET = {
    "checkbox_state": "checkbox", "circled_choice": "checkbox",
    "handdrawn_check": "checkbox", "filled_bubble": "checkbox",
    "crossed_out": "checkbox", "ambiguous_mark": "checkbox",
    "chart_value": "chart", "chart_axis": "chart", "chart_legend": "chart",
    "chart_title": "chart", "figure_content": "figure", "figure_caption": "figure",
    "field_value_pairing": "kv", "form_region": "kv", "repeated_labels": "kv",
    "repeated_pairs": "kv", "blank_field": "kv", "scanned_form": "kv",
    "small_text": "text", "multi_field_reasoning": "text", "remarks_binding": "text",
}
PARSERS = ["extend_performance_v1_0_0", "extend_performance_v2_0_0_advanced",
           "aws_textract", "azure_di", "reducto", "llamaparse"]
LBL = {"extend_performance_v1_0_0": "Extend v1",
       "extend_performance_v2_0_0_advanced": "Extend v2(adv)",
       "aws_textract": "Textract", "azure_di": "Azure DI",
       "reducto": "Reducto", "llamaparse": "LlamaParse"}


def bucket(it: dict) -> str:
    for c in it.get("capabilities", []) or []:
        if c in _BUCKET:
            return _BUCKET[c]
    return "table"


def main() -> int:
    run = REPO / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "v500")
    bank = json.loads((run / "qa" / "bank_r1.json").read_text())
    items = bank["items"]
    docs = {it.get("source_file") for it in items}

    print("=" * 60)
    print(f"  {run.name.upper()} — FINAL VALIDATION")
    print("=" * 60)
    print(f"docs: {len(docs)}    questions: {len(items)}    "
          f"atomized: {bank.get('atomized', False)}")

    # type distribution
    bk = Counter(bucket(it) for it in items)
    print("\ntype distribution:")
    for b in ("checkbox", "chart", "figure", "kv", "table", "text"):
        n = bk.get(b, 0)
        print(f"  {b:10} {n:>5}  ({n/len(items)*100:.0f}%)")

    # structural integrity
    no_gold = sum(1 for it in items if not str(it.get("gold_answer", "")).strip())
    no_fmt = sum(1 for it in items if not str(it.get("response_format", "")).strip())
    imgdir = run / "images"
    no_img = sum(1 for it in items
                 if not (imgdir / f"{it.get('source_file','')}.png").exists())
    blank = set()
    bp = run / "qa" / "blank_pages.json"
    if bp.exists():
        blank = {Path(b["dest_path"]).stem for b in json.loads(bp.read_text())}
    on_blank = sum(1 for it in items if it.get("source_file") in blank)
    corrupt = sum(1 for it in items
                  if "'='" in str(it.get("gold_answer", ""))
                  or re.search(r"\b\d{3}/[A-Z]", str(it.get("gold_answer", ""))))
    print("\nstructural integrity:")
    for label, n in (("missing gold", no_gold), ("missing response_format", no_fmt),
                     ("missing page image", no_img),
                     ("questions on blank pages", on_blank),
                     ("corrupt-gold questions", corrupt)):
        flag = "OK" if n == 0 else "** CHECK **"
        print(f"  {label:30} {n:>5}   {flag}")

    # per-parser accuracy
    rp = run / "eval" / "eval_r1_rescored.json"
    if rp.exists():
        ev = json.loads(rp.read_text())
        stat = {p: [0, 0] for p in PARSERS}
        for page in ev.values():
            for q in page["questions"].values():
                for p in PARSERS:
                    if p not in q.get("parser_answers", {}):
                        continue
                    for k in q["keys"]:
                        stat[p][1] += 1
                        if p in q.get("sub_correct_llm", {}).get(k, []):
                            stat[p][0] += 1
        print("\nper-parser accuracy (LLM-judge):")
        rows = sorted(((p, c / t * 100) for p, (c, t) in stat.items() if t),
                      key=lambda x: -x[1])
        for p, acc in rows:
            print(f"  {LBL[p]:<16} {acc:>6.1f}%")
        if len(rows) >= 2:
            print(f"  delta: top vs bottom = {rows[0][1]-rows[-1][1]:.1f}pp")

    # sample
    print("\nsample atomic questions:")
    for it in items[:6]:
        print(f"  [{it['question_id']}]")
        print(f"    Q: {it['question'][:90]}")
        print(f"    gold: {it['gold_answer']!s:.70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
