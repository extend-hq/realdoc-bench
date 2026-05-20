#!/usr/bin/env python3
"""Detect computed/derived questions — ones whose gold answer requires
arithmetic (sum, total, count, difference, average, percentage) rather than
verbatim extraction of a value printed on the page.

Keyword pre-filter narrows candidates, then a text LLM classifies each.
Writes <run>/qa/computed_qids.json — the list of computed question_ids.
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from pilot_50_gen_qa import gemini_generate, load_env_file   # noqa: E402

KW = re.compile(
    r"\b(total|subtotal|sum|summed|combined|aggregate|calculat|comput|"
    r"difference|average|\bmean\b|percentage|percent|count|how many|"
    r"number of|add up|grand total|net amount)\b", re.I)

CLASSIFY_SYSTEM = (
    "You classify a document-extraction benchmark question. Decide whether "
    "producing its GOLD answer requires ARITHMETIC or DERIVATION (adding, "
    "totaling, counting items, subtracting, averaging, computing a "
    "percentage) — versus simply reading values printed VERBATIM on a "
    "document.\n\n"
    "computed=true  — the gold can only be obtained by calculation (e.g. "
    "'sum of the line items', 'how many rows', 'difference between A and B').\n"
    "computed=false — every gold value is printed on the page and copied "
    "verbatim. A printed 'Total: $5,000' field is computed=false (that total "
    "is printed); counting how many entries exist is computed=true.\n\n"
    'Return exactly: {"computed": true|false, "reason": "<short>"}'
)


def classify(it: dict) -> bool:
    prompt = (
        f"QUESTION: {it.get('question','')}\n"
        f"RESPONSE FORMAT: {it.get('response_format','')}\n"
        f"GOLD ANSWER: {it.get('gold_answer','')}\n\n"
        "Does producing this gold require arithmetic/derivation?"
    )
    try:
        r = gemini_generate([{"text": prompt}], system=CLASSIFY_SYSTEM, temperature=0.0)
        return bool(isinstance(r, dict) and r.get("computed"))
    except Exception:
        return False


def main() -> int:
    run = REPO / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "v500")
    load_env_file(REPO / ".env")
    bank = json.loads((run / "qa" / "bank_r1.json").read_text())
    items = bank["items"]
    cands = [it for it in items
             if KW.search(f"{it.get('question','')} {it.get('gold_answer','')}")]
    print(f"{len(items)} questions · {len(cands)} keyword candidates", flush=True)

    computed: list[str] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(classify, it): it for it in cands}
        for i, fut in enumerate(as_completed(futs), 1):
            if fut.result():
                computed.append(futs[fut]["question_id"])
            if i % 50 == 0 or i == len(cands):
                print(f"  [{i}/{len(cands)}] computed: {len(computed)} "
                      f"({(time.time()-t0)/60:.1f}min)", flush=True)

    out = run / "qa" / "computed_qids.json"
    out.write_text(json.dumps(sorted(computed), indent=2))
    print(f"\n{len(computed)} computed questions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
