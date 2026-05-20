"""Multi-VLM ground-truth verifier.

For every (doc × field) in the v0_tax dataset:
  1. Render the relevant PDF pages as PNG images (150 DPI).
  2. Ask Gemini 3 Pro (visual inspection) to confirm or correct the candidate.
  3. Ask Claude Opus 4.7 (visual inspection) to confirm or correct the candidate.
  4. Aggregate verdicts:
       - both confirm                    → ✅ verified-by-vlms
       - both correct to the same value  → 🔧 corrected-by-vlms (GT updated)
       - they disagree / one unsure      → ⚠ disputed (GT untouched, needs human)
  5. Write ground_truth/<doc>.json (with corrections applied) + a sidecar
     ground_truth/<doc>.verification.json with per-field per-verifier verdicts.

Seed candidates come from the most reliable existing extraction we have
(Extend v2 GA + claude_sonnet_4_6 agent), already cached under
runs/verify_v0tax_v3/<doc>/extend_performance_v2_0_0/claude_sonnet_4_6/agent_output.json
— if that's missing we fall back to gemini_3_flash agent output.

Usage:
  python scripts/verify_gt_with_vlms.py                          # all 6 docs
  python scripts/verify_gt_with_vlms.py --doc tax_harris_2023    # one doc
  python scripts/verify_gt_with_vlms.py --doc tax_harris_2023 --field filing_status_checkbox_grid   # one field
  python scripts/verify_gt_with_vlms.py --resume                 # skip fields already verified

Env: GEMINI_API_KEY (or GOOGLE_API_KEY), ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "datasets" / "extraction" / "v0_tax"
SOURCES = DATA / "sources"
SCHEMAS = DATA / "schemas"
GT_DIR = DATA / "ground_truth"
SEED_RUN = REPO / "runs" / "verify_v0tax_v3"

GEMINI_MODEL = "gemini-3-pro-preview"   # may need to be "gemini-3-pro" depending on availability
OPUS_MODEL   = "claude-opus-4-7"
DEFAULT_DPI  = 150
MAX_PAGES_PER_FIELD = 6   # cap to keep image payloads reasonable

# ── page selection per field ────────────────────────────────────────────────
# Maps regex on field path → predicate that returns which page numbers to render.
# Falls back to "first 4 pages + page containing the matching sub-doc" if no match.

def select_pages(doc_id: str, field_path: str, doc_sub_documents: list[dict]) -> list[int]:
    """Pick 1-MAX_PAGES_PER_FIELD pages most relevant to the field."""
    def pages_of(type_substring: str) -> list[int]:
        out = []
        for s in doc_sub_documents or []:
            if type_substring.lower() in (s.get("type", "") or "").lower():
                out.extend(range(s["start_page"], s["end_page"] + 1))
        return sorted(set(out))[:MAX_PAGES_PER_FIELD]

    # Cover-page-anchored fields (filing status, names, checkboxes, headers)
    if field_path in (
        "tax_year", "primary_taxpayer_full_name", "spouse_full_name",
        "filing_status_checkbox_grid", "presidential_campaign_fund_grid",
        "digital_assets_question_grid", "standard_deduction_checkboxes_grid",
        "age_blindness_checkboxes_grid", "dependents_table",
    ):
        # Form 1040 cover (1-2)
        return pages_of("Form 1040")[:2] or [1, 2]

    if field_path == "form_1040_income_and_tax_lines":
        return pages_of("Form 1040")[:2] or [1, 2]

    if field_path.startswith("schedule_a"):
        return pages_of("Schedule A")[:2] or []

    if field_path.startswith("schedule_b"):
        return pages_of("Schedule B")[:2] or []

    if field_path.startswith("schedule_c"):
        return pages_of("Schedule C")[:2] or []

    if field_path.startswith("schedule_e"):
        return pages_of("Schedule E")[:2] or []

    if field_path.startswith("form_1116"):
        return pages_of("Form 1116")[:MAX_PAGES_PER_FIELD] or []

    if field_path == "state_returns_attached":
        # state returns scattered through packet; show first 6 pages that aren't Form 1040
        scattered: list[int] = []
        for s in doc_sub_documents or []:
            t = (s.get("type") or "").lower()
            if t == "misc_form" or t.startswith("form 540") or t.startswith("form d-40") or t.startswith("form il"):
                scattered.extend(range(s["start_page"], s["end_page"] + 1))
        return sorted(set(scattered))[:MAX_PAGES_PER_FIELD] or []

    if "preparer" in field_path:
        # Preparer block usually on the bottom of Form 1040 page 2
        return pages_of("Form 1040")[:2] or [1, 2]

    if field_path == "all_form_headers_with_first_page":
        # Need broad coverage; sample first page of each unique sub-doc, max MAX_PAGES_PER_FIELD
        seen_types: set[str] = set()
        pages = []
        for s in doc_sub_documents or []:
            t = s.get("type") or ""
            if t not in seen_types:
                seen_types.add(t)
                pages.append(s["start_page"])
        return pages[:MAX_PAGES_PER_FIELD]

    if field_path.startswith("form_1040x"):
        # Amended-return-specific. Look for Form 1040X in sub-docs
        return pages_of("1040X")[:2] or pages_of("amended")[:2] or []

    if field_path.startswith("form_1065"):
        return pages_of("Form 1065")[:2] or [1, 2]
    if field_path.startswith("schedule_k_distrib"):
        return pages_of("Schedule K")[:2] or []
    if field_path == "schedule_l_balance_sheet_rows":
        return pages_of("Schedule L")[:2] or []
    if field_path == "schedule_m1_book_to_tax_lines":
        return pages_of("M-1")[:2] or pages_of("M-3")[:2] or []
    if field_path == "partners_table":
        return pages_of("K-1")[:MAX_PAGES_PER_FIELD] or []
    if field_path == "entity_type_checkbox_grid":
        return pages_of("Schedule B")[:1] or pages_of("Form 1065")[:2] or [1, 2]
    if field_path == "schedule_b_questions_table":
        return pages_of("Schedule B")[:MAX_PAGES_PER_FIELD] or pages_of("Form 1065")[:2] or []

    if field_path.startswith("form_4562"):
        return pages_of("Form 4562")[:MAX_PAGES_PER_FIELD] or [1, 2, 3]
    if field_path.startswith("form_8938"):
        return pages_of("Form 8938")[:MAX_PAGES_PER_FIELD] or []
    if field_path == "supplements_with_first_page":
        # Each unique supplement's first page
        seen: set[str] = set()
        pages = []
        for s in doc_sub_documents or []:
            t = s.get("type") or ""
            if t.startswith("supplement") or t == "other":
                key = (s.get("identifier") or "") + "::" + t
                if key not in seen:
                    seen.add(key)
                    pages.append(s["start_page"])
        return pages[:MAX_PAGES_PER_FIELD]

    # Default — first 3 pages
    return [1, 2, 3]


def render_pages(doc_id: str, page_numbers: list[int], dpi: int = DEFAULT_DPI) -> list[bytes]:
    import pymupdf
    pdf_path = SOURCES / f"{doc_id}.pdf"
    pdf = pymupdf.open(str(pdf_path))
    out: list[bytes] = []
    for n in page_numbers:
        if 1 <= n <= len(pdf):
            pix = pdf[n - 1].get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72))
            out.append(pix.tobytes("png"))
    pdf.close()
    return out


# ── verifier prompt ─────────────────────────────────────────────────────────

VERIFY_SYSTEM = (
    "You are a tax-form GROUND-TRUTH verifier.\n\n"
    "You will receive: (a) page images from a U.S. tax-return PDF, (b) the schema "
    "definition of one extraction field, (c) a CANDIDATE answer produced by an "
    "automated extraction system. Your job is to VISUALLY INSPECT the image(s) "
    "and determine whether the candidate is correct.\n\n"
    "Rules:\n"
    "- Treat the page images as the source of truth. Don't infer from the candidate.\n"
    "- For checkbox grids, look at the actual box glyph: [x]/[X]/checked vs [ ]/empty.\n"
    "- For numeric line items, the printed number is the answer (commas + decimals as-is).\n"
    "- For arrays, every row that should exist must be enumerated; omit duplicates.\n"
    "- For null-able fields, return null only if the field is genuinely absent.\n"
    "- For dates, return ISO format (YYYY-MM-DD).\n\n"
    "Respond with a single JSON object, no markdown fences, no extra text:\n"
    "{\n"
    '  "verdict": "confirm" | "correct" | "unsure",\n'
    '  "corrected_value": <only if verdict=="correct"; can be string/number/bool/array/object/null>,\n'
    '  "confidence": "high" | "medium" | "low",\n'
    '  "explanation": "1-2 short sentences explaining what you see"\n'
    "}"
)


def _user_text(field_path: str, schema_def: dict, candidate) -> str:
    return (
        f"FIELD PATH: {field_path}\n"
        f"SCHEMA TYPE: {schema_def.get('type')}\n"
        f"SCHEMA DESCRIPTION: {schema_def.get('description','')}\n"
        + (f"ENUM VALUES: {schema_def.get('enum')}\n" if schema_def.get('enum') else '')
        + (f"ITEM SHAPE: {json.dumps(schema_def.get('items'), default=str)[:600]}\n" if schema_def.get('items') else '')
        + (f"PROPERTIES: {json.dumps(schema_def.get('properties'), default=str)[:600]}\n" if schema_def.get('properties') else '')
        + f"\nCANDIDATE ANSWER:\n```json\n{json.dumps(candidate, indent=2, default=str)}\n```\n\n"
        f"Verify the candidate against the page image(s). Return the JSON verdict only."
    )


def _parse_verdict_json(txt: str) -> dict:
    """Extract first {…} block from model text + json.loads."""
    if not txt:
        return {"verdict": "unsure", "explanation": "empty response"}
    # Try parsing the whole thing first
    txt = txt.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```\s*$", "", txt)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError as e:
                return {"verdict": "unsure", "explanation": f"json decode failed: {e}", "raw_text": txt[:400]}
    return {"verdict": "unsure", "explanation": "no JSON object in response", "raw_text": txt[:400]}


def verify_with_gemini(field_path: str, schema_def: dict, candidate, pngs: list[bytes]) -> dict:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"verdict": "unsure", "explanation": "no GEMINI_API_KEY"}

    client = genai.Client(api_key=api_key)

    parts: list = [VERIFY_SYSTEM]
    for png in pngs:
        parts.append(types.Part.from_bytes(data=png, mime_type="image/png"))
    parts.append(_user_text(field_path, schema_def, candidate))

    last_err = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=parts,
                config={"response_mime_type": "application/json"},
            )
            return _parse_verdict_json(getattr(resp, "text", "") or "")
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    return {"verdict": "unsure", "explanation": f"gemini error: {last_err}"}


def verify_with_opus(field_path: str, schema_def: dict, candidate, pngs: list[bytes]) -> dict:
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"verdict": "unsure", "explanation": "no ANTHROPIC_API_KEY"}

    client = Anthropic(api_key=api_key)

    content: list = []
    for png in pngs:
        b64 = base64.b64encode(png).decode()
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })
    content.append({"type": "text", "text": _user_text(field_path, schema_def, candidate)})

    last_err = None
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=OPUS_MODEL,
                max_tokens=4000,
                system=VERIFY_SYSTEM,
                messages=[{"role": "user", "content": content}],
            )
            txt = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            return _parse_verdict_json(txt)
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    return {"verdict": "unsure", "explanation": f"opus error: {last_err}"}


# ── status reconciliation ──────────────────────────────────────────────────


def _values_equal(a, b) -> bool:
    """Loose equality for JSON values."""
    if a is None and b is None:
        return True
    try:
        return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    except Exception:
        return False


def reconcile(candidate, gemini_verdict: dict, opus_verdict: dict) -> dict:
    """Returns {status, final_value, applied_correction_bool}."""
    g_v = (gemini_verdict or {}).get("verdict")
    o_v = (opus_verdict or {}).get("verdict")

    # Both confirm
    if g_v == "confirm" and o_v == "confirm":
        return {"status": "verified", "final_value": candidate, "applied_correction": False}

    # Both correct
    if g_v == "correct" and o_v == "correct":
        g_corr = gemini_verdict.get("corrected_value")
        o_corr = opus_verdict.get("corrected_value")
        if _values_equal(g_corr, o_corr):
            return {"status": "corrected", "final_value": g_corr, "applied_correction": True}
        else:
            return {"status": "disputed", "final_value": candidate, "applied_correction": False}

    # One confirms, the other corrects — disputed unless the "corrected" value matches candidate
    if g_v == "confirm" and o_v == "correct":
        if _values_equal(candidate, opus_verdict.get("corrected_value")):
            return {"status": "verified", "final_value": candidate, "applied_correction": False}
        return {"status": "disputed", "final_value": candidate, "applied_correction": False}
    if o_v == "confirm" and g_v == "correct":
        if _values_equal(candidate, gemini_verdict.get("corrected_value")):
            return {"status": "verified", "final_value": candidate, "applied_correction": False}
        return {"status": "disputed", "final_value": candidate, "applied_correction": False}

    # Anything else (unsure / errors / null verdict)
    return {"status": "needs_review", "final_value": candidate, "applied_correction": False}


# ── per-doc orchestration ──────────────────────────────────────────────────


def load_seed_gt(doc_id: str) -> dict:
    """Pick the best available seed extraction (Extend v2 GA × claude preferred)."""
    for parser in ["extend_performance_v2_0_0", "aws_textract", "azure_di"]:
        for agent in ["claude_sonnet_4_6", "gemini_3_flash"]:
            ao = SEED_RUN / doc_id / parser / agent / "agent_output.json"
            if ao.exists():
                try:
                    return json.load(open(ao))
                except Exception:
                    continue
    return {}


def load_sub_documents(doc_id: str) -> list[dict]:
    """Read sub_documents from the existing GT (we have splitting GT from PoliTax)."""
    gt = GT_DIR / f"{doc_id}.json"
    if gt.exists():
        try:
            data = json.load(open(gt))
            return data.get("sub_documents", []) or []
        except Exception:
            return []
    return []


def verify_field(doc_id: str, field_path: str, schema_def: dict, candidate, sub_docs: list[dict],
                 verifiers: tuple[str, ...] = ("gemini", "opus")) -> dict:
    """Run both verifiers on one field, reconcile."""
    pages = select_pages(doc_id, field_path, sub_docs)
    if not pages:
        return {
            "status": "needs_review",
            "final_value": candidate,
            "applied_correction": False,
            "explanation": "no relevant pages identified (field type-specific)",
            "verdicts": {},
        }
    pngs = render_pages(doc_id, pages)
    if not pngs:
        return {
            "status": "needs_review",
            "final_value": candidate,
            "applied_correction": False,
            "explanation": f"could not render pages {pages}",
            "verdicts": {},
        }
    g = verify_with_gemini(field_path, schema_def, candidate, pngs) if "gemini" in verifiers else None
    o = verify_with_opus(field_path, schema_def, candidate, pngs) if "opus" in verifiers else None
    rec = reconcile(candidate, g or {}, o or {})
    rec["pages_inspected"] = pages
    rec["verdicts"] = {"gemini_3_pro": g, "claude_opus_4_7": o}
    return rec


def verify_doc(doc_id: str, *, only_field: str | None = None, resume: bool = False,
               parallelism: int = 4):
    """End-to-end for one doc: seed → verify each field → write GT + sidecar."""
    schema = json.load(open(SCHEMAS / f"{doc_id}.json"))
    seed = load_seed_gt(doc_id)
    sub_docs = load_sub_documents(doc_id)

    sidecar_path = GT_DIR / f"{doc_id}.verification.json"
    existing_sidecar = json.load(open(sidecar_path)) if (resume and sidecar_path.exists()) else {}

    # Start from seed (replace old GT structure)
    gt = dict(seed)
    sidecar: dict = dict(existing_sidecar)

    fields = list(schema.get("properties", {}).items())
    if only_field:
        fields = [(f, p) for f, p in fields if f == only_field]

    print(f"\n=== {doc_id}: verifying {len(fields)} fields ===")
    # Submit in parallel up to `parallelism`
    futures = {}
    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        for field_path, schema_def in fields:
            if resume and field_path in sidecar:
                print(f"  ⏭  {field_path}: skip (already verified)")
                continue
            cand = seed.get(field_path)
            fut = pool.submit(verify_field, doc_id, field_path, schema_def, cand, sub_docs)
            futures[fut] = field_path
        for fut in as_completed(futures):
            field = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"status": "needs_review", "explanation": f"verifier crashed: {e}",
                       "final_value": seed.get(field), "applied_correction": False, "verdicts": {}}
                traceback.print_exc()
            sidecar[field] = rec
            icon = {"verified": "✅", "corrected": "🔧", "disputed": "⚠️", "needs_review": "🟡"}.get(rec["status"], "?")
            print(f"  {icon} {field:<48s} {rec['status']:<14s}  pages={rec.get('pages_inspected')}")
            if rec.get("applied_correction"):
                gt[field] = rec["final_value"]

    # Persist
    GT_DIR.mkdir(parents=True, exist_ok=True)
    (GT_DIR / f"{doc_id}.json").write_text(json.dumps(gt, indent=2, default=str))
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str))

    counts = {"verified": 0, "corrected": 0, "disputed": 0, "needs_review": 0}
    for v in sidecar.values():
        counts[v.get("status", "needs_review")] = counts.get(v.get("status", "needs_review"), 0) + 1
    print(f"  → {counts}")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", help="single doc_id (default: all 6)")
    ap.add_argument("--field", help="single field path (default: all)")
    ap.add_argument("--resume", action="store_true", help="skip fields already in sidecar")
    ap.add_argument("--parallelism", type=int, default=4)
    args = ap.parse_args()

    DOCS = ["tax_harris_2023", "tax_obama_2009", "tax_biden_2018",
            "tax_trump_2019", "tax_trump_lfb_2015", "tax_harris_2020"]
    docs = [args.doc] if args.doc else DOCS

    totals = {"verified": 0, "corrected": 0, "disputed": 0, "needs_review": 0}
    for d in docs:
        c = verify_doc(d, only_field=args.field, resume=args.resume, parallelism=args.parallelism)
        for k, v in c.items():
            totals[k] = totals.get(k, 0) + v

    print(f"\n========================================")
    print(f"OVERALL: {totals}")
    print(f"========================================")


if __name__ == "__main__":
    raise SystemExit(main())
