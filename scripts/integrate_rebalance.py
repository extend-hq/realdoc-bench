#!/usr/bin/env python3
"""Master integration for the v500 rebalance.

Waits for the harden chain + checkmark generation, then:
  1. source_chart_pages.py  — pull genuine chart pages from v1.2_final
  2. pilot_50_parse.py      — parse the new chart pages (6 parsers)
  3. gen_chart_qa.py        — regenerate chart questions (tightened prompt,
                              now covering the new pages)
  4. append chart_questions.json + checkmark_questions.json to the bank
  5. pilot_50_eval -> rescore -> dashboard
  6. print the refreshed per-type distribution

Run after harden_iter4 / verify / prune have completed.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "runs" / "v500"
QA, EV = RUN / "qa", RUN / "eval"
PY = str(REPO / ".venv" / "bin" / "python")
PARSERS = ["extend_performance_v1_0_0", "extend_performance_v2_0_0_advanced",
           "aws_textract", "azure_di", "reducto", "llamaparse"]


def sh(cmd: list, log: Path) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    with log.open("w") as f:
        return subprocess.run([str(c) for c in cmd], stdout=f,
                              stderr=subprocess.STDOUT, cwd=REPO).returncode


def wait_for(marker_file: Path, needle: str, label: str, tries: int = 240) -> bool:
    print(f"waiting for {label} ...", flush=True)
    for _ in range(tries):
        if marker_file.exists() and needle in marker_file.read_text():
            return True
        time.sleep(30)
    return marker_file.exists() and needle in marker_file.read_text()


def main() -> int:
    t0 = time.time()
    if not wait_for(RUN / "prune_chain.log", "prune chain done", "harden+prune chain"):
        print("FATAL: harden/prune chain never finished"); return 1
    if not wait_for(RUN / "gen_checkmark_qa.log", "validated complex-checkmark",
                    "checkmark generation"):
        print("WARN: checkmark gen not confirmed — continuing")

    # 0. cleanup — drop blank-page + corrupt-gold questions before anything else
    sh([PY, "scripts/cleanup_bank.py", "v500"], RUN / "rebalance_cleanup.log")

    # 1. source chart pages
    if sh([PY, "scripts/source_chart_pages.py"], RUN / "rebalance_source.log") != 0:
        print("WARN: source_chart_pages failed — continuing without new pages")

    # 2. parse the new chart pages (cached docs skip)
    sh([PY, "scripts/pilot_50_parse.py", "--run-dir", RUN, "--workers", "6",
        "--parsers", *PARSERS], RUN / "rebalance_parse.log")

    # 3. regenerate chart questions (tightened prompt, now incl. new pages)
    sh([PY, "scripts/gen_chart_qa.py", "v500"], RUN / "rebalance_chartqa.log")

    # 4. append chart + checkmark questions to the bank
    bank = json.loads((QA / "bank_r1.json").read_text())
    existing = {it["question_id"] for it in bank["items"]}
    added = 0
    for fname in ("chart_questions.json", "checkmark_questions.json"):
        fp = QA / fname
        if not fp.exists():
            continue
        for it in json.loads(fp.read_text()).get("items", []):
            if it.get("question_id") and it["question_id"] not in existing:
                bank["items"].append(it)
                existing.add(it["question_id"])
                added += 1
    shutil.copy(QA / "bank_r1.json", QA / "bank_before_rebalance.json")
    (QA / "bank_r1.json").write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n")
    print(f"appended {added} chart/checkmark questions — bank now "
          f"{len(bank['items'])} questions", flush=True)

    # 4b. atomize — every multi-part question -> N self-contained atomic
    #     single-value questions (section anchor + one field + format).
    if sh([PY, "scripts/atomize_bank.py", "v500"],
          RUN / "rebalance_atomize.log") != 0:
        print("WARN: atomize failed — continuing with multi-part bank")

    # 5. eval -> rescore -> dashboard
    sh([PY, "scripts/pilot_50_eval.py", "--run-dir", RUN, "--max-workers", "12"],
       RUN / "rebalance_eval.log")
    sh([PY, "scripts/pilot_50_rescore.py", "--run-dir", RUN, "--max-workers", "12"],
       RUN / "rebalance_rescore.log")
    sh([PY, "scripts/pilot_50_build_html.py", "--run-dir", RUN, "--mode", "post"],
       RUN / "rebalance_html.log")
    if (RUN / "review_post.html").exists():
        shutil.copy(RUN / "review_post.html", RUN / "dashboard.html")

    # 6. refreshed per-type distribution
    _print_distribution()
    print(f"\nDONE in {(time.time()-t0)/60:.1f}min — dashboard rebuilt", flush=True)
    return 0


_BUCKET = {
    "checkbox_state": "checkbox", "circled_choice": "checkbox",
    "handdrawn_check": "checkbox", "filled_bubble": "checkbox",
    "crossed_out": "checkbox", "ambiguous_mark": "checkbox",
    "chart_value": "chart", "chart_axis": "chart", "chart_legend": "chart",
    "chart_title": "chart", "figure_content": "figure", "figure_caption": "figure",
}
_SUFFIX_BUCKET = {
    "field_value_pairing": "kv", "form_region": "kv", "repeated_labels": "kv",
    "repeated_pairs": "kv", "blank_field": "kv", "scanned_form": "kv",
    "small_text": "text", "multi_field_reasoning": "text", "remarks_binding": "text",
}


def _print_distribution() -> None:
    bank = json.loads((QA / "bank_r1.json").read_text())
    counts: dict[str, int] = defaultdict(int)
    for it in bank["items"]:
        caps = it.get("capabilities") or []
        bucket = next((_BUCKET[c] for c in caps if c in _BUCKET), None)
        if bucket is None:
            m = re.match(r"v500_(?:fig_|mark_)?q\d+_(.+)", it["question_id"])
            suf = m.group(1) if m else ""
            bucket = _SUFFIX_BUCKET.get(suf, "table")
        counts[bucket] += 1
    total = sum(counts.values()) or 1
    print("\n=== refreshed question-type distribution ===", flush=True)
    for b in ("checkbox", "chart", "figure", "kv", "table", "text"):
        print(f"  {b:10} {counts.get(b,0):>5}  ({counts.get(b,0)/total*100:.0f}%)")


if __name__ == "__main__":
    raise SystemExit(main())
