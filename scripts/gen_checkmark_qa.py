#!/usr/bin/env python3
"""Generate complex / non-standard checkmark questions.

Standard [x] checkboxes are well covered. This targets the marks parsers
actually struggle with: options CIRCLED instead of ticked, hand-drawn check /
X marks, filled-in bubbles (scantron style), struck-through / crossed-out
options, and ambiguous / overwritten marks.

For each candidate page (handwriting / checkbox / hand-filled form pages): one
vision call decides whether the page has non-standard selection marks and, if
so, authors up to 3 exact-match questions about the SELECTION STATE. Each gold
is re-verified by an image-baseline call.

Output: runs/<run>/qa/checkmark_questions.json
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

MARK_CATS = {"handwriting", "checkboxes", "mixed"}
MARK_TYPES = {"v1_patient_intake", "v1_vaers", "v1_clinical_crf", "v1_cms1500",
              "v1_clinical_facesheet", "v1_clinical_consent", "v1_fnma_710",
              "v1_acord_125_126", "v1_acord_4", "v1_urla_1003", "v1_urar",
              "v1_fec_form3"}
MARK_CAPS = {"circled_choice", "handdrawn_check", "filled_bubble",
             "crossed_out", "ambiguous_mark", "checkbox_state"}

MARK_SYSTEM = (
    "You author exact-match benchmark questions that test whether a "
    "document-to-markdown parser correctly captured NON-STANDARD selection "
    "marks on a form — the marks parsers most often miss:\n"
    "  - an option CIRCLED (rather than a ticked checkbox)\n"
    "  - a HAND-DRAWN check or X mark (not a printed glyph)\n"
    "  - a FILLED-IN bubble / oval (scantron style)\n"
    "  - a STRUCK-THROUGH / crossed-out option\n"
    "  - an AMBIGUOUS / partial / overwritten mark\n\n"
    "First decide: does this page contain any non-standard selection mark like "
    "the above? If NOT (only plain printed [x] checkboxes, or none), return "
    '{"has_marks": false, "questions": []}.\n\n'
    "If it does, author up to 3 questions asking WHICH option is marked and "
    "HOW. MANDATORY RULES:\n"
    "  1. The gold must be the selection state actually visible — which option "
    "is circled / checked / filled / crossed out. Never guess; never compute.\n"
    "  2. FLAT GOLD: a plain 'key=value; key=value' string, dotted compound "
    "keys for grouped marks. No braces/nesting. 'null' for unmarked.\n"
    "  3. ANCHOR by the printed label of the field / question being marked.\n"
    "  4. response_format must spell out every key and value form.\n\n"
    "Return JSON: {\"has_marks\": <bool>, \"questions\": [{\"question\": ..., "
    "\"response_format\": ..., \"gold_answer\": ..., \"capabilities\": [..]}]}\n"
    f"Use only these capability tags: {sorted(MARK_CAPS)}"
)


def gen_for_page(image_path: Path, source_file: str) -> list[dict]:
    parts = [
        {"text": "Inspect this page for non-standard selection marks "
                 "(circled / hand-drawn / filled / crossed-out) and author "
                 "checkmark-state questions per the rules."},
        {"inline_data": {"mime_type": "image/png", "data": encoded_image(image_path)}},
    ]
    try:
        payload = gemini_generate(parts, system=MARK_SYSTEM, temperature=0.2)
    except Exception as exc:
        print(f"  ! {source_file}: {exc}", flush=True)
        return []
    if not isinstance(payload, dict) or not payload.get("has_marks"):
        return []
    out = []
    for q in payload.get("questions", []) or []:
        if not isinstance(q, dict) or not q.get("gold_answer"):
            continue
        caps = [c for c in (q.get("capabilities") or []) if c in MARK_CAPS]
        q["capabilities"] = caps or ["circled_choice"]
        q["source_file"] = source_file
        q["answer_type"] = "complex_checkmark_exact"
        out.append(q)
    return out


def main() -> int:
    run = REPO / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "v500")
    load_env_file(REPO / ".env")
    rows = list(csv.DictReader((run / "sample.csv").open()))
    cands = {}
    for r in rows:
        if r.get("category") in MARK_CATS or r.get("type") in MARK_TYPES:
            cands[r["dest_path"]] = r
    pages = [(dp, run / "images" / f"{Path(dp).stem}.png") for dp in cands]
    pages = [(dp, img) for dp, img in pages if img.exists()]
    print(f"complex-checkmark candidate pages: {len(pages)}", flush=True)

    t0 = time.time()
    raw: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(gen_for_page, img, Path(dp).stem) for dp, img in pages]
        for i, fut in enumerate(as_completed(futs), 1):
            raw.extend(fut.result())
            if i % 20 == 0 or i == len(pages):
                print(f"  [{i}/{len(pages)}] proposed {len(raw)} questions "
                      f"({(time.time()-t0)/60:.1f}min)", flush=True)
    print(f"proposed {len(raw)} checkmark questions — validating golds...", flush=True)

    img_by_src = {Path(dp).stem: img for dp, img in pages}
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
        if v.get("image_answer"):
            return {**q, "gold_answer": v["image_answer"]}
        return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(validate, q) for q in raw]):
            r = fut.result()
            n += 1
            if r:
                kept.append(r)
            if n % 25 == 0 or n == len(raw):
                print(f"  validated [{n}/{len(raw)}] kept={len(kept)}", flush=True)

    for i, q in enumerate(sorted(kept, key=lambda x: x["source_file"]), 1):
        q["question_id"] = f"v500_mark_q{i:03d}_{q['capabilities'][0]}"
        q["difficulty"] = "hard_structural"

    out = run / "qa" / "checkmark_questions.json"
    out.write_text(json.dumps({"items": kept}, indent=2, ensure_ascii=False))
    print(f"\n{len(kept)} validated complex-checkmark questions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
