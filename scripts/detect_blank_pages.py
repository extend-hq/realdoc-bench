#!/usr/bin/env python3
"""Detect blank / unfilled form pages in a run.

A page is 'blank' if it shows only the pre-printed form template — field
labels, lines, boxes — with NO filled-in data (no typed values, no
handwriting, no checkmarks/circles). Such pages are degenerate benchmark
material: there is nothing to extract.

Writes runs/<run>/qa/blank_pages.json — the list of blank dest_paths.
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
from pilot_50_gen_qa import gemini_generate, encoded_image, load_env_file  # noqa: E402

BLANK_SYSTEM = (
    "You are checking whether a document page is a BLANK / UNFILLED form.\n\n"
    "blank = the page shows only the pre-printed form template (field labels, "
    "lines, boxes, section headings) with NO filled-in data: no typed entries, "
    "no handwriting, no checked/circled boxes, no signatures.\n"
    "not blank = the form has ANY filled data, OR the page is narrative text / "
    "a table with data / a chart — anything with real content.\n\n"
    'Return exactly: {"is_blank": true|false, "filled_fields_seen": "<short '
    'note: a couple of filled values you see, or \'none\'>"}'
)


def check(image_path: Path) -> dict:
    try:
        res = gemini_generate(
            [{"text": "Is this a blank, unfilled form template?"},
             {"inline_data": {"mime_type": "image/png",
                              "data": encoded_image(image_path)}}],
            system=BLANK_SYSTEM, temperature=0.0)
    except Exception as exc:
        return {"is_blank": False, "error": str(exc)[:120]}
    return res if isinstance(res, dict) else {"is_blank": False}


def main() -> int:
    run = REPO / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "v500")
    load_env_file(REPO / ".env")
    rows = list(csv.DictReader((run / "sample.csv").open()))
    jobs = []
    for r in rows:
        img = run / "images" / f"{Path(r['dest_path']).stem}.png"
        if img.exists():
            jobs.append((r, img))
    print(f"checking {len(jobs)} pages for blankness", flush=True)

    t0 = time.time()
    blank, done = [], 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(check, img): r for r, img in jobs}
        for fut in as_completed(futs):
            r = futs[fut]
            res = fut.result()
            done += 1
            if res.get("is_blank"):
                blank.append({"dest_path": r["dest_path"], "type": r.get("type"),
                              "is_filled": r.get("is_filled"),
                              "note": res.get("filled_fields_seen", "")})
            if done % 50 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] blank so far: {len(blank)} "
                      f"({(time.time()-t0)/60:.1f}min)", flush=True)

    out = run / "qa" / "blank_pages.json"
    out.write_text(json.dumps(blank, indent=2, ensure_ascii=False))
    from collections import Counter
    by_type = Counter(b["type"] for b in blank)
    print(f"\n{len(blank)} blank pages / {len(jobs)} ({len(blank)/len(jobs)*100:.0f}%)")
    for t, n in by_type.most_common():
        print(f"  {t:32} {n}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
