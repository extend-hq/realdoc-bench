"""Generate hard structural QA questions from page images.

This script encodes the iterative process we used to build the
selected_pages hard structural benchmark:

  1. Analyze each page image with Gemini to enumerate structural features
     that are likely to challenge document parsers (checkbox grids,
     repeated records, field-value blocks, totals, parallel columns,
     multi-section forms, etc.).
  2. Propose candidate questions for each image. Every question must use
     an exact-match response format and be anchored by semantic labels
     visible on the page, not visual position.
  3. Validate each gold answer with a fresh image-only Gemini call. Drop
     questions where the image baseline disagrees with the proposed gold
     (these are usually ambiguous or have a wrong gold).
  4. Persist the surviving questions to a question bank JSON that matches
     the shape used by selected_pages/qa_questions_exact.json.

The script never overwrites existing files. Re-runs reuse cached
analysis/candidate/validation artifacts so individual images can be
re-tried without re-paying for the whole run.

Usage example (initial bank):
    python selected_pages_v2/generate_structural_qa.py \\
        --images selected_pages_v2/images \\
        --out selected_pages_v2/qa_questions_v2.json \\
        --question-set selected_pages_v2_hard_structural \\
        --questions-per-image 4

Replace-easy mode (after running the bank through run_qa_eval.py):
    python selected_pages_v2/generate_structural_qa.py \\
        --replace-easy \\
        --bank selected_pages_v2/qa_questions_v2.json \\
        --results selected_pages_v2/qa_eval/qa_results_gemini.json \\
        --images selected_pages_v2/images \\
        --out selected_pages_v2/qa_questions_v2_round2.json

Replace-easy looks at every question where all parsers (or >= --easy-threshold)
got an exact match, and asks Gemini to author a strictly harder replacement on
the SAME page and SAME structural capability, anchored by printed labels, with
a fresh image-baseline gold check. Hard questions are kept verbatim.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent / "images"
DEFAULT_OUT_PATH = Path(__file__).resolve().parent / "qa_questions_v2.json"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "question_generation"
DEFAULT_EXEMPLARS_PATH = REPO_ROOT / "selected_pages" / "qa_questions_exact.json"
MODEL = os.environ.get("GEMINI_QUESTION_MODEL", os.environ.get("GEMINI_QA_MODEL", "gemini-3-flash-preview"))
MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_QUESTION_MAX_OUTPUT_TOKENS", "16384"))
# Gemini 2.5/3 thinking models eat output tokens for internal reasoning. We need
# deterministic + bounded output for JSON parsing, so disable thinking entirely.
# Override via env if you want to experiment.
GEMINI_THINKING_BUDGET = int(os.environ.get("GEMINI_THINKING_BUDGET", "0"))


CAPABILITY_TAXONOMY = (
    "checkbox_state",
    "multi_column_grid",
    "row_binding",
    "table_structure",
    "column_alignment",
    "repeated_records",
    "parallel_columns",
    "field_value_pairing",
    "form_region",
    # "layout_regions",  ← REMOVED: led to header/footer/page-number questions
    "scanned_form",
    "blank_field",
    "merged_cells",
    "total_row",
    "small_text",
    "hazmat_fields",
    "line_binding",
    "remarks_binding",
    "repeated_pairs",
    "repeated_labels",
    "n_a_vs_n_o",
    "multi_field_reasoning",
    # chart / figure capabilities — content of plots, graphs, diagrams
    "chart_value",      # a labelled data value in a bar/line/pie chart
    "chart_axis",       # axis title / unit / tick label
    "chart_legend",     # legend category / series name
    "chart_title",      # chart or plot title
    "figure_content",   # a label / value inside a diagram or figure
    "figure_caption",   # the printed caption of a figure
    # complex / non-standard selection marks
    "circled_choice",   # an option circled rather than checkbox-ticked
    "handdrawn_check",  # a hand-drawn check / X mark (not a printed glyph)
    "filled_bubble",    # a filled-in bubble / oval (scantron style)
    "crossed_out",      # a struck-through / crossed-out option
    "ambiguous_mark",   # a partial / ambiguous / overwritten mark
)


ANALYZE_SYSTEM = (
    "You are inspecting a single document page image to find structural challenges "
    "that test document-to-markdown parsers. Focus on features that depend on layout, "
    "alignment, or repetition. Do not invent text; rely only on what is visible. "
    "Return a single JSON object."
)


PROPOSE_SYSTEM = (
    "You are authoring exact-match benchmark questions for a document-to-markdown "
    "parser evaluation. Follow these mandatory rules:\n\n"
    "RULE 1 — FLAT PLAIN-STRING GOLD ANSWERS. The gold_answer MUST be a single "
    "plain string in 'key1=value1; key2=value2; ...' format. NEVER return JSON "
    "objects, arrays, nested structures, curly braces, or a colon-grouped "
    "map-of-maps. For grouped / 2-level data (per block, per row, per "
    "officer), FLATTEN it with compound keys joined by a dot — "
    "'GROUP.field=value' — e.g. 'BLOCK 1.INDIVIDUAL=selected; BLOCK 1.GL "
    "CODE=91851; BLOCK 2.INDIVIDUAL=unselected'. Every segment must be exactly "
    "one 'key=value' pair. Use the literal lowercase token 'null' for "
    "blank/missing values.\n\n"
    "RULE 2 — SEMANTIC ANCHORS. Anchor every question by labels printed on the "
    "page (e.g., 'In the signatory block for \"Signature on behalf of packer\"'). "
    "Never use visual position descriptors like 'upper-right' or 'top-left'.\n\n"
    "RULE 3 — STRUCTURAL FOCUS. Questions must test whether the parser preserved "
    "page structure: row-column binding, parallel columns, checkbox grids, totals, "
    "repeated records, field-value pairs. Require multiple paired values together; "
    "avoid single-cell lookups.\n\n"
    "RULE 4 — SURGICAL RESPONSE_FORMAT. The response_format string must spell out "
    "every key name, every delimiter, and every value form (e.g. <yes|no|null>, "
    "<digits>, exact spelling). Do not request units or commas unless explicitly "
    "part of the canonical answer. Match the rigor of the provided exemplars.\n\n"
    "RULE 5 — UNAMBIGUOUS, MACHINE-CHECKABLE GOLD. Gold must be exactly what a "
    "correct parser plus a careful reader would emit. No alternate phrasings, no "
    "added units, no quoting, no commas inside numbers unless required.\n\n"
    "RULE 6 — NO LAYOUT / HEADER / FOOTER QUESTIONS. Questions must test "
    "CONTENT extraction (actual data fields, table cells, checkbox states, "
    "narrative answers, chart values, signatures, figure labels), NOT layout "
    "decoration. FORBIDDEN question types:\n"
    "  • page numbers (header or footer)\n"
    "  • document URLs, report IDs, footnote markers in footer\n"
    "  • document title / report name in header (alone)\n"
    "  • agency name or classification banner\n"
    "  • watermarks, copyright marks, version stamps\n"
    "  • section heading transcription by itself (only ask about the heading "
    "as an ANCHOR for content beneath it)\n"
    "  • font / color / style / formatting questions\n"
    "  • presence-only questions (\"is there a footer?\")\n"
    "These decoration questions don't test parser structural understanding — "
    "any OCR can read header text. We care about whether parsers preserve "
    "row-column binding, checkbox grids, field-value pairs, narrative regions "
    "with their headings, and complex tables.\n\n"
    "Return only the JSON requested. No prose."
)


REFINE_SYSTEM = (
    "You are reviewing one benchmark question that failed image-baseline "
    "validation. Decide whether the question is fixable.\n\n"
    "- If the original gold was wrong, produce a corrected gold matching what the "
    "image clearly shows.\n"
    "- If the question wording was ambiguous, tighten it with a clearer printed-label "
    "anchor.\n"
    "- If the response_format allowed alternate readings, tighten it.\n"
    "- If the question is fundamentally unanswerable from the image, discard.\n\n"
    "ABSOLUTE RULES (same as propose):\n"
    "  • gold_answer must remain a plain 'key=value; key=value' string. Never a "
    "JSON object, array, or nested structure.\n"
    "  • Use the literal token 'null' for blanks.\n"
    "  • Anchor by printed labels, never by visual position.\n\n"
    "Return JSON with this exact shape:\n"
    "  {\"action\": \"refine\" | \"discard\",\n"
    "   \"reason\": \"one-sentence explanation\",\n"
    "   \"refined\": <full updated question object or null>}"
)


VALIDATE_SYSTEM = (
    "You are answering a single benchmark question by reading the page image only. "
    "Do not use outside knowledge. If the requested value is missing, blank, "
    "contradictory, or unclear, return null. Return exactly one JSON object of the "
    "form {\"answer\": <string or null>}. The answer must follow the requested format "
    "exactly: no prose, labels, extra punctuation, or extra keys."
)


REPLACE_SYSTEM = (
    "You are authoring a STRICTLY HARDER replacement for an exact-match benchmark "
    "question that every document-to-markdown parser already got right. The "
    "replacement must target the SAME structural capability on the SAME page, but "
    "must be meaningfully harder for a parser that loses layout (table rows, "
    "parallel columns, checkbox grids, repeated records). Anchor by printed "
    "labels, never by visual position. The gold answer must be unambiguous, "
    "machine-checkable, and grounded in text that is clearly visible on the page. "
    "Never duplicate any existing question on the same image.\n\n"
    "RULE — VERBATIM VALUES ONLY, NO COMPUTATION. Every gold value MUST be text "
    "printed directly on the page, copyable verbatim. NEVER author a question "
    "whose answer is a sum, total, subtotal, difference, count, average, "
    "percentage, or ANY value the reader must calculate or derive. A printed "
    "'Total' field is allowed ONLY if that exact total is printed on the page. "
    "If a value is not literally visible, do not ask for it.\n\n"
    "Return only the JSON object requested."
)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY (e.g. in .env.local) before running."
        )
    return key


def encoded_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def gemini_generate(
    parts: list[dict[str, Any]],
    *,
    system: str,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Call Gemini generateContent with a JSON response and return the parsed body."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": GEMINI_THINKING_BUDGET},
        },
    }
    with httpx.Client(timeout=300) as client:
        response = client.post(url, params={"key": api_key()}, json=body)
        response.raise_for_status()
        raw = response.json()
    text = _extract_text(raw)
    parsed = _parse_json_object(text)
    return parsed


def _extract_text(raw: dict[str, Any]) -> str:
    candidates = raw.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {raw}")
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
    if not text:
        raise RuntimeError(f"Gemini returned no text: {raw}")
    return text


def _parse_json_object(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Locate the first '{' and decode the first balanced JSON object, ignoring
    # any trailing content (Gemini occasionally returns prose or a second JSON
    # object after the first one).
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    decoder = json.JSONDecoder()
    obj, _end = decoder.raw_decode(text[start:])
    return obj


# ---------------------------------------------------------------------------
# Stage 1: analyze structural features in each image
# ---------------------------------------------------------------------------


ANALYZE_INSTRUCTION = (
    "Inspect the page image and produce JSON describing:\n"
    "  document_type: short label for what kind of document this is\n"
    "  features: array of objects, each describing one structural element worth\n"
    "    targeting in a parser benchmark. Each feature must include:\n"
    "      label: a short human-friendly label (e.g. 'Site Hazards grid')\n"
    "      anchor: the exact words from the page that label this element\n"
    "      structure: one of [checkbox_grid, table, repeated_records,\n"
    "                 field_value_block, total_row, parallel_columns,\n"
    "                 form_region, signature_block, list]\n"
    "      what_makes_it_hard: short reason this is structurally challenging\n"
    "  capabilities: array of strings drawn from the list below that this page\n"
    "    can plausibly test:\n"
    f"      {list(CAPABILITY_TAXONOMY)}\n"
    "Return ONLY the JSON object. Do not return prose."
)


def analyze_image(image_path: Path) -> dict[str, Any]:
    parts = [
        {"text": ANALYZE_INSTRUCTION},
        {"inline_data": {"mime_type": "image/png", "data": encoded_image(image_path)}},
    ]
    return gemini_generate(parts, system=ANALYZE_SYSTEM, temperature=0.0)


# ---------------------------------------------------------------------------
# Stage 2: propose candidate questions
# ---------------------------------------------------------------------------


def load_v1_exemplars(path: Path, count: int = 5) -> list[dict[str, Any]]:
    """Pick a small, capability-diverse set of v1 exemplar questions.

    Used as few-shot grounding so the propose stage matches the rigor and
    plain-string answer shape of the v1 benchmark.
    """

    if not path.exists():
        return []
    data = json.loads(path.read_text())
    items = data.get("items", [])
    seen_caps: set[tuple[str, ...]] = set()
    exemplars: list[dict[str, Any]] = []
    for item in items:
        caps = tuple(sorted(item.get("capabilities") or []))
        if not caps or caps in seen_caps:
            continue
        gold = item.get("gold_answer") or ""
        if gold.lstrip().startswith(("{", "[")):
            continue
        seen_caps.add(caps)
        exemplars.append(
            {
                "question": item.get("question"),
                "response_format": item.get("response_format"),
                "gold_answer": gold,
                "capabilities": list(item.get("capabilities") or []),
            }
        )
        if len(exemplars) >= count:
            break
    return exemplars


def derive_capability_budget(analysis: dict[str, Any], max_buckets: int) -> list[str]:
    """Pick up to ``max_buckets`` capability tags this image can plausibly test.

    Prefer tags emitted by the analysis stage; fall back to a sensible default
    ordering for documents where analysis is sparse.
    """

    analyzed = [c for c in (analysis.get("capabilities") or []) if c in CAPABILITY_TAXONOMY]
    if analyzed:
        # De-duplicate while preserving order.
        seen: set[str] = set()
        ordered: list[str] = []
        for c in analyzed:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        if len(ordered) >= max_buckets:
            return ordered[:max_buckets]
        # Top off with broadly applicable structural tags.
        for fallback in ("field_value_pairing", "table_structure", "row_binding", "total_row"):
            if fallback not in seen and len(ordered) < max_buckets:
                ordered.append(fallback)
                seen.add(fallback)
        return ordered[:max_buckets]
    return ["field_value_pairing", "table_structure", "row_binding", "total_row"][:max_buckets]


def propose_instruction(
    *,
    num_questions: int,
    source_file: str,
    prefix: str,
    start_index: int,
    exemplars: list[dict[str, Any]],
    capability_budget: list[str],
) -> str:
    exemplars_block = json.dumps(exemplars, ensure_ascii=False, indent=2)
    budget_block = ", ".join(capability_budget)
    return (
        f"Generate exactly {num_questions} hard structural QA questions for the "
        "page image. Each question must target a DISTINCT capability bucket from "
        f"this budget (one question per bucket): {budget_block}\n"
        "If the image clearly cannot support a given bucket, replace it with a "
        "different stronger structural challenge on this same page; do not skip "
        "and shrink the count.\n\n"
        "Below are exemplar questions from our v1 benchmark. MATCH their rigor "
        "and EXACTLY their plain-string gold shape ('key=value; key=value' or "
        "'value; value; value'). Note that NO exemplar gold is a JSON object:\n"
        f"{exemplars_block}\n\n"
        "Return JSON with this exact shape (no prose):\n"
        "  {\n"
        "    \"items\": [\n"
        "      {\n"
        f"        \"question_id\": \"{prefix}_q###_{{topic}}\" using zero-padded\n"
        f"          3-digit numbering starting at q{start_index:03d},\n"
        f"        \"source_file\": \"{source_file}\",\n"
        "        \"question\": \"...\",\n"
        "        \"answer_type\": \"hard_structural_exact\",\n"
        "        \"response_format\": \"Return exactly: key=<form>; ... Use null for blanks.\",\n"
        "        \"gold_answer\": \"key1=value1; key2=value2; ...\",   ← PLAIN STRING ONLY\n"
        "        \"difficulty\": \"very_hard\",\n"
        "        \"capabilities\": [capability tag from this budget plus 1-2 supporting tags]\n"
        "      }\n"
        "    ]\n"
        "  }\n"
        "Capability taxonomy:\n"
        f"  {list(CAPABILITY_TAXONOMY)}\n"
        "Use only lowercase, underscore-separated capability tags. NEVER return a "
        "JSON object, array, or nested structure inside gold_answer."
    )


def propose_questions(
    image_path: Path,
    analysis: dict[str, Any],
    num_questions: int,
    source_file: str,
    prefix: str,
    start_index: int,
    *,
    exemplars: list[dict[str, Any]],
    capability_budget: list[str],
) -> list[dict[str, Any]]:
    parts = [
        {
            "text": propose_instruction(
                num_questions=num_questions,
                source_file=source_file,
                prefix=prefix,
                start_index=start_index,
                exemplars=exemplars,
                capability_budget=capability_budget,
            )
        },
        {
            "text": "Structural analysis from the page (use as guidance):\n"
            + json.dumps(analysis, ensure_ascii=False, indent=2)
        },
        {"inline_data": {"mime_type": "image/png", "data": encoded_image(image_path)}},
    ]
    payload = gemini_generate(parts, system=PROPOSE_SYSTEM, temperature=0.2)
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise RuntimeError(f"Propose stage did not return items: {payload}")
    return items


# ---------------------------------------------------------------------------
# Stage 3: validate gold answers with a fresh image-only call
# ---------------------------------------------------------------------------


def _try_parse_json(value: Any) -> Any:
    """Return parsed JSON for dicts/lists or JSON-string inputs, else None."""

    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped[0] not in "{[":
            return None
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def normalize_exact(value: Any) -> str:
    """Canonicalize an answer for exact-match scoring.

    For JSON-shaped values (dicts/lists or JSON strings), compare structurally
    via ``json.dumps(sort_keys=True)``. Otherwise mirror the plain-string
    normalization used by selected_pages/run_qa_eval.normalize_exact.
    """

    parsed = _try_parse_json(value)
    if parsed is not None:
        return json.dumps(parsed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    if value is None:
        return "null"
    text = str(value).strip()
    text = text.replace("None", "null")
    text = re.sub(r"(?<=\d)\s*:\s*(?=\d)", ":", text)
    text = re.sub(r"(?<=\d)\s*([ap])m\b", r" \1m", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)[oO](?=F\b)", "0", text)
    text = re.sub(r"(?<=\d)\s+GA\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\s+KGS?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ;", ";").replace("; ", "; ")
    text = text.replace(" =", "=").replace("= ", "=")
    return text.lower()


def validate_question(image_path: Path, question: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "Question:\n"
        f"{question['question']}\n\n"
        "Answer style:\n"
        f"{question['response_format']}"
    )
    parts = [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/png", "data": encoded_image(image_path)}},
    ]
    payload = gemini_generate(parts, system=VALIDATE_SYSTEM, temperature=0.0)
    answer = payload.get("answer") if isinstance(payload, dict) else None
    gold = question.get("gold_answer")
    exact = normalize_exact(answer) == normalize_exact(gold)
    return {
        "image_answer": answer,
        "gold_answer": gold,
        "image_matches_gold": exact,
    }


# ---------------------------------------------------------------------------
# Stage 3b: refine questions that fail image-baseline validation
# ---------------------------------------------------------------------------


def refine_question(
    image_path: Path,
    question: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Ask the model to fix or discard a question that failed validation.

    Returns a dict like ``{"action": "refine"|"discard", "reason": "...",
    "refined": <updated question or None>}``.
    """

    parts = [
        {
            "text": (
                "Original question (failed image-baseline validation):\n"
                f"{json.dumps(question, ensure_ascii=False, indent=2)}\n\n"
                "Image baseline produced this answer instead:\n"
                f"{json.dumps(validation.get('image_answer'), ensure_ascii=False)}\n\n"
                "Proposed gold:\n"
                f"{question.get('gold_answer')}\n\n"
                "Decide whether to refine (fix question/format/gold so the image "
                "baseline reproduces the gold) or discard (unfixable). Keep the "
                "gold as a plain 'key=value; key=value' string."
            )
        },
        {"inline_data": {"mime_type": "image/png", "data": encoded_image(image_path)}},
    ]
    payload = gemini_generate(parts, system=REFINE_SYSTEM, temperature=0.0)
    if not isinstance(payload, dict):
        return {"action": "discard", "reason": "non-dict refine response", "refined": None}
    action = payload.get("action")
    refined = payload.get("refined") if action == "refine" else None
    return {
        "action": "discard" if action != "refine" else "refine",
        "reason": payload.get("reason"),
        "refined": refined if isinstance(refined, dict) else None,
    }


# ---------------------------------------------------------------------------
# Stage 4 (replace-easy mode): regenerate harder versions of easy questions
# ---------------------------------------------------------------------------


def replace_instruction(
    *,
    easy_question: dict[str, Any],
    other_questions: list[dict[str, Any]],
    source_file: str,
    new_question_id: str,
) -> str:
    capabilities = easy_question.get("capabilities") or []
    other_summaries = [
        {
            "question_id": q.get("question_id"),
            "question": q.get("question"),
            "gold_answer": q.get("gold_answer"),
        }
        for q in other_questions
    ]
    return (
        "Every parser passed the question below. Author ONE strictly harder "
        "replacement on the SAME page image that:\n"
        "  - Targets the same structural capability tags as the original.\n"
        "  - Is anchored by labels printed on the page (no 'top-right', 'upper-left').\n"
        "  - Requires returning multiple values together as a FLAT 'key=value; "
        "key=value' string. For grouped data use dotted compound keys "
        "(GROUP.field=value) — NEVER nested braces {}, a colon-grouped "
        "map-of-maps, or a paired list.\n"
        "  - Probes a layout feature parsers tend to lose: row-column binding,\n"
        "    parallel columns, repeated records, checkbox grids, totals, or\n"
        "    multi-section forms. Prefer cells/rows that are visually adjacent\n"
        "    or look similar to other rows.\n"
        "  - Has an unambiguous, exact-match gold answer fully visible on the page.\n"
        "  - Does NOT duplicate, paraphrase, or significantly overlap any question\n"
        "    listed under 'Other existing questions on this page'.\n\n"
        "Original (now too easy) question:\n"
        f"{json.dumps(easy_question, ensure_ascii=False, indent=2)}\n\n"
        "Other existing questions on this page (do not duplicate):\n"
        f"{json.dumps(other_summaries, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with this exact shape:\n"
        "  {\n"
        "    \"item\": {\n"
        f"      \"question_id\": \"{new_question_id}\",\n"
        f"      \"source_file\": \"{source_file}\",\n"
        "      \"question\": \"...\",\n"
        "      \"answer_type\": \"hard_structural_exact\",\n"
        "      \"response_format\": \"...\",\n"
        "      \"gold_answer\": \"...\",\n"
        "      \"difficulty\": \"very_hard\",\n"
        f"      \"capabilities\": {json.dumps(list(capabilities))}\n"
        "    },\n"
        "    \"why_harder\": \"one short sentence explaining the new structural trap\"\n"
        "  }\n"
        "Use only lowercase, underscore-separated capability tags from this taxonomy:\n"
        f"  {list(CAPABILITY_TAXONOMY)}\n"
        "Do not return prose outside the JSON."
    )


def propose_replacement(
    *,
    image_path: Path,
    easy_question: dict[str, Any],
    other_questions: list[dict[str, Any]],
    source_file: str,
    new_question_id: str,
) -> dict[str, Any]:
    parts = [
        {
            "text": replace_instruction(
                easy_question=easy_question,
                other_questions=other_questions,
                source_file=source_file,
                new_question_id=new_question_id,
            )
        },
        {"inline_data": {"mime_type": "image/png", "data": encoded_image(image_path)}},
    ]
    payload = gemini_generate(parts, system=REPLACE_SYSTEM, temperature=0.3)
    item = payload.get("item") if isinstance(payload, dict) else None
    if not isinstance(item, dict):
        raise RuntimeError(f"Replacement stage did not return an item: {payload}")
    item["_why_harder"] = payload.get("why_harder")
    return item


def load_eval_results(path: Path) -> dict[str, dict[str, Any]]:
    """Group exact_match flags by question_id from a run_qa_eval.py result file."""

    data = json.loads(path.read_text())
    results = data.get("results")
    if not isinstance(results, list):
        raise RuntimeError(f"{path} does not contain a 'results' array")

    by_qid: dict[str, dict[str, Any]] = {}
    for entry in results:
        qid = entry.get("question_id")
        if not qid:
            continue
        bucket = by_qid.setdefault(
            qid,
            {
                "source_file": entry.get("source_file"),
                "parsers": [],
                "correct": [],
                "wrong": [],
            },
        )
        parser = entry.get("parser")
        if parser is None:
            continue
        bucket["parsers"].append(parser)
        if entry.get("exact_match"):
            bucket["correct"].append(parser)
        else:
            bucket["wrong"].append(parser)
    return by_qid


def classify_easy_questions(
    bank_items: list[dict[str, Any]],
    eval_by_qid: dict[str, dict[str, Any]],
    *,
    easy_threshold: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Split bank items into (easy, hard, unscored).

    easy_threshold: minimum number of parsers that got the question right for it
    to be considered easy. When None, defaults to "all parsers correct".
    """

    easy: list[dict[str, Any]] = []
    hard: list[dict[str, Any]] = []
    unscored: list[str] = []
    for item in bank_items:
        qid = item.get("question_id")
        bucket = eval_by_qid.get(qid)
        if not bucket:
            unscored.append(qid)
            hard.append(item)
            continue
        num_parsers = len(bucket["parsers"])
        num_correct = len(bucket["correct"])
        threshold = easy_threshold if easy_threshold is not None else num_parsers
        item_with_stats = {**item, "_stats": {"correct": num_correct, "parsers": num_parsers}}
        if num_parsers > 0 and num_correct >= threshold:
            easy.append(item_with_stats)
        else:
            hard.append(item)
    return easy, hard, unscored


def replace_easy_mode(args: argparse.Namespace) -> None:
    if not args.bank or not args.bank.exists():
        raise FileNotFoundError(f"--bank {args.bank} does not exist")
    if not args.results or not args.results.exists():
        raise FileNotFoundError(f"--results {args.results} does not exist")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing {args.out}")

    api_key()

    bank = json.loads(args.bank.read_text())
    items = bank.get("items")
    if not isinstance(items, list):
        raise RuntimeError(f"{args.bank} does not contain an 'items' array")

    eval_by_qid = load_eval_results(args.results)
    easy, hard, unscored = classify_easy_questions(
        items, eval_by_qid, easy_threshold=args.easy_threshold
    )

    print(
        f"Bank: {len(items)} items | easy: {len(easy)} | hard: {len(hard)} | "
        f"unscored: {len(unscored)}",
        flush=True,
    )
    if unscored:
        print(f"  (no eval rows found for: {unscored[:5]}{'...' if len(unscored) > 5 else ''})", flush=True)
    if not easy:
        print("No easy questions to replace; refusing to write a no-op output.", flush=True)
        return

    by_source: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_source.setdefault(item.get("source_file"), []).append(item)

    swap_log: list[dict[str, Any]] = []
    replacements_by_qid: dict[str, dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []
    started = time.perf_counter()

    # Per-easy-question worker — propose a harder replacement + validate its
    # gold against an image-baseline call. Runs in a thread pool; each unit is
    # independent (own cache files), so results are merged single-threaded.
    def process_one(easy_item: dict[str, Any]) -> dict[str, Any]:
        qid = easy_item.get("question_id")
        source_file = easy_item.get("source_file")
        image_path = args.images / f"{source_file}.png"
        if not image_path.exists():
            return {"qid": qid, "replacement": None, "log": None,
                    "dropped": {"question_id": qid, "reason": "image_not_found"}}
        other_questions = [
            q for q in by_source.get(source_file, []) if q.get("question_id") != qid
        ]

        def run_proposal() -> dict[str, Any]:
            stripped_easy = {k: v for k, v in easy_item.items() if not k.startswith("_")}
            return propose_replacement(
                image_path=image_path, easy_question=stripped_easy,
                other_questions=other_questions, source_file=source_file,
                new_question_id=qid,
            )

        try:
            replacement = cached_call(
                args.artifact_dir / "replace_easy" / f"{qid}.json", run_proposal)
            validation = cached_call(
                args.artifact_dir / "replace_easy_validation" / f"{qid}.json",
                lambda: validate_question(image_path, replacement))
        except Exception as exc:  # one bad question must not kill the run
            return {"qid": qid, "replacement": None, "log": None,
                    "dropped": {"question_id": qid, "reason": f"error: {exc}"}}

        capabilities = [t for t in replacement.get("capabilities", [])
                        if t in CAPABILITY_TAXONOMY]
        replacement["capabilities"] = (capabilities or easy_item.get("capabilities")
                                       or ["table_structure"])
        replacement["source_file"] = source_file
        replacement.setdefault("answer_type", "hard_structural_exact")
        replacement.setdefault("difficulty", "very_hard")
        replacement["question_id"] = qid

        log_entry = {
            "question_id": qid, "source_file": source_file,
            "original": {
                "question": easy_item.get("question"),
                "gold_answer": easy_item.get("gold_answer"),
                "correct": easy_item.get("_stats", {}).get("correct"),
                "parsers": easy_item.get("_stats", {}).get("parsers"),
            },
            "replacement": {
                "question": replacement.get("question"),
                "gold_answer": replacement.get("gold_answer"),
                "why_harder": replacement.get("_why_harder"),
            },
            "validation": validation,
        }
        replacement.pop("_why_harder", None)

        if validation.get("image_matches_gold"):
            log_entry["outcome"] = "replaced"
            return {"qid": qid, "replacement": replacement, "log": log_entry, "dropped": None}
        if args.auto_fix_gold and validation.get("image_answer") is not None:
            replacement["gold_answer"] = validation["image_answer"]
            log_entry["outcome"] = "replaced_with_auto_fixed_gold"
            return {"qid": qid, "replacement": replacement, "log": log_entry, "dropped": None}
        log_entry["outcome"] = "dropped"
        return {"qid": qid, "replacement": None, "log": log_entry,
                "dropped": {"question_id": qid, "reason": "validation_failed",
                            "validation": validation}}

    workers = int(os.environ.get("REPLACE_EASY_WORKERS", "8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, e): e for e in easy}
        for idx, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            res = fut.result()
            if res.get("log"):
                swap_log.append(res["log"])
            if res.get("replacement") is not None:
                replacements_by_qid[res["qid"]] = res["replacement"]
            if res.get("dropped"):
                dropped.append(res["dropped"])
            if idx % 25 == 0 or idx == len(easy):
                print(f"  [replace {idx}/{len(easy)}] replaced={len(replacements_by_qid)} "
                      f"dropped={len(dropped)} ({(time.perf_counter()-started)/60:.1f}min)",
                      flush=True)

    refined_items: list[dict[str, Any]] = []
    for item in items:
        qid = item.get("question_id")
        if qid in replacements_by_qid:
            new = replacements_by_qid[qid]
            # Version the retired question: carry prior versions forward and
            # append a snapshot of the question being replaced.
            prior = list(item.get("versions", []))
            snapshot = {k: item.get(k) for k in
                        ("question", "gold_answer", "response_format", "capabilities")}
            snapshot["retired_reason"] = "too_easy_or_computed"
            new["versions"] = prior + [snapshot]
            refined_items.append(new)
        else:
            # If the question was easy and not replaced (e.g. validation failed
            # without --auto-fix-gold), drop it. Otherwise keep the original.
            if any(d.get("question_id") == qid for d in dropped):
                continue
            refined_items.append(item)

    refined_bank = {
        **{k: v for k, v in bank.items() if k != "items"},
        "items": refined_items,
        "replace_easy_metadata": {
            "model": MODEL,
            "input_bank": str(args.bank),
            "input_results": str(args.results),
            "easy_threshold": args.easy_threshold,
            "num_easy_identified": len(easy),
            "num_replaced": sum(1 for log in swap_log if log["outcome"].startswith("replaced")),
            "num_dropped": len(dropped),
            "auto_fix_gold": args.auto_fix_gold,
            "elapsed_sec": time.perf_counter() - started,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(refined_bank, indent=2, ensure_ascii=False) + "\n")

    swap_log_path = args.artifact_dir / "replace_easy_log.json"
    swap_log_path.parent.mkdir(parents=True, exist_ok=True)
    swap_log_path.write_text(json.dumps(swap_log, indent=2, ensure_ascii=False) + "\n")

    print(
        f"Wrote refined bank to {args.out} "
        f"({refined_bank['replace_easy_metadata']['num_replaced']} replaced, "
        f"{refined_bank['replace_easy_metadata']['num_dropped']} dropped). "
        f"Swap log at {swap_log_path}.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


FROM_CACHE_ONLY = False


def cached_call(path: Path, producer) -> Any:
    """Reuse an existing JSON artifact when present; otherwise run producer and persist.

    When ``FROM_CACHE_ONLY`` is set, raise on cache miss instead of calling the
    producer. Callers wrap this in try/except so missing-cache images are
    skipped non-fatally.
    """

    if path.exists():
        return json.loads(path.read_text())
    if FROM_CACHE_ONLY:
        raise FileNotFoundError(f"--from-cache set and no cached artifact at {path}")
    payload = producer()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def list_images(image_dir: Path) -> list[Path]:
    images = sorted(image_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No .png files found in {image_dir}")
    return images


def assign_question_ids(items: list[dict[str, Any]], prefix: str, start_index: int) -> int:
    """Renumber question_id values sequentially across the run so they're unique."""

    next_index = start_index
    for item in items:
        existing = item.get("question_id") or ""
        match = re.match(rf"^{re.escape(prefix)}_q\d+_(.+)$", existing)
        topic = match.group(1) if match else "topic"
        topic = re.sub(r"[^a-z0-9_]+", "_", topic.lower()).strip("_") or "topic"
        item["question_id"] = f"{prefix}_q{next_index:03d}_{topic}"
        next_index += 1
    return next_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory containing .png images.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Question bank JSON to write. Refuses to overwrite.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR, help="Where to cache per-stage artifacts.")
    parser.add_argument("--doc-id", default="selected_pages_v2", help="doc_id field for the resulting question bank.")
    parser.add_argument("--question-set", default="selected_pages_v2_hard_structural", help="question_set field for the resulting bank.")
    parser.add_argument("--prefix", default="hard_v3", help="Prefix used for question_id values (e.g. hard_v3).")
    parser.add_argument("--questions-per-image", type=int, default=4, help="How many questions to propose per image.")
    parser.add_argument("--start-index", type=int, default=1, help="Starting q### index across all images.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of images to process.")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env.local", help="Env file containing GEMINI_API_KEY.")
    parser.add_argument(
        "--auto-fix-gold",
        action="store_true",
        help="When the image baseline disagrees with the proposed gold, replace gold with the baseline answer instead of dropping the question.",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help=(
            "Build the bank only from cached artifacts under --artifact-dir. Images "
            "missing any cached stage are skipped non-fatally. Never calls Gemini."
        ),
    )
    parser.add_argument(
        "--exemplars-path",
        type=Path,
        default=DEFAULT_EXEMPLARS_PATH,
        help="Path to a v1 question bank to use as few-shot exemplars in propose.",
    )
    parser.add_argument(
        "--exemplars-count",
        type=int,
        default=5,
        help="Number of capability-diverse v1 exemplars to inject.",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help=(
            "Skip the refine stage. Questions that fail image-baseline validation "
            "are dropped immediately (default: try one refinement pass)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of images to process concurrently (default: 4).",
    )
    parser.add_argument(
        "--replace-easy",
        action="store_true",
        help=(
            "Replace-easy mode: read an existing question bank plus its eval "
            "results and regenerate harder questions where all (or --easy-threshold) "
            "parsers got it right. Requires --bank and --results."
        ),
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=None,
        help="Path to an existing question bank JSON (input to --replace-easy mode).",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help=(
            "Path to a qa_results_gemini.json produced by run_qa_eval.py for the bank "
            "(input to --replace-easy mode)."
        ),
    )
    parser.add_argument(
        "--easy-threshold",
        type=int,
        default=None,
        help=(
            "Minimum number of parsers that must have gotten a question right for it "
            "to be replaced. Defaults to 'all parsers correct'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)

    global FROM_CACHE_ONLY
    FROM_CACHE_ONLY = args.from_cache

    if args.replace_easy:
        replace_easy_mode(args)
        return

    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing {args.out}")

    if not args.from_cache:
        api_key()  # fail fast if no key

    images = list_images(args.images)
    if args.limit is not None:
        images = images[: args.limit]

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    started = time.perf_counter()

    image_errors: list[dict[str, Any]] = []
    exemplars = load_v1_exemplars(args.exemplars_path, count=args.exemplars_count)
    if exemplars:
        print(f"Loaded {len(exemplars)} v1 exemplar(s) from {args.exemplars_path}", flush=True)
    else:
        print(f"No exemplars loaded from {args.exemplars_path}; proposing without few-shot.", flush=True)

    print_lock = threading.Lock()

    def log(msg: str) -> None:
        with print_lock:
            print(msg, flush=True)

    def process_image(image_idx: int, image_path: Path) -> dict[str, Any]:
        """Run analyze → propose → validate (+ refine) for one image.

        Returns ``{"accepted": [...], "rejected": [...], "error": dict|None,
        "skipped_via_cache": bool}``. Question IDs are pre-allocated from
        ``image_idx`` so concurrent execution still yields a deterministic
        bank ordering.
        """

        source_file = image_path.stem
        image_start_index = args.start_index + (image_idx - 1) * args.questions_per_image
        accepted_local: list[dict[str, Any]] = []
        rejected_local: list[dict[str, Any]] = []

        try:
            analysis_path = args.artifact_dir / "analysis" / f"{source_file}.json"
            analysis = cached_call(analysis_path, lambda: analyze_image(image_path))

            capability_budget = derive_capability_budget(analysis, args.questions_per_image)

            candidates_path = args.artifact_dir / "candidates" / f"{source_file}.json"
            candidates = cached_call(
                candidates_path,
                lambda: propose_questions(
                    image_path,
                    analysis,
                    args.questions_per_image,
                    source_file,
                    args.prefix,
                    image_start_index,
                    exemplars=exemplars,
                    capability_budget=capability_budget,
                ),
            )
            if not isinstance(candidates, list):
                raise RuntimeError(f"Candidates artifact for {source_file} is not a list")
        except Exception as exc:
            log(f"[{image_idx:>2}/{len(images)}] {source_file}  SKIP propose ({exc})")
            return {
                "accepted": [],
                "rejected": [],
                "error": {"source_file": source_file, "stage": "propose", "error": str(exc)},
            }

        assign_question_ids(candidates, args.prefix, image_start_index)

        validation_path = args.artifact_dir / "validation" / f"{source_file}.json"

        def run_validation() -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for item in candidates:
                try:
                    out.append(validate_question(image_path, item))
                except Exception as exc:
                    out.append(
                        {
                            "image_answer": None,
                            "gold_answer": item.get("gold_answer"),
                            "image_matches_gold": False,
                            "error": str(exc),
                        }
                    )
            return out

        try:
            validations = cached_call(validation_path, run_validation)
        except Exception as exc:
            log(f"[{image_idx:>2}/{len(images)}] {source_file}  SKIP validate ({exc})")
            return {
                "accepted": [],
                "rejected": [],
                "error": {"source_file": source_file, "stage": "validate", "error": str(exc)},
            }

        for item, validation in zip(candidates, validations):
            item["source_file"] = source_file
            item.setdefault("answer_type", "hard_structural_exact")
            item.setdefault("difficulty", "very_hard")
            capabilities = [tag for tag in item.get("capabilities", []) if tag in CAPABILITY_TAXONOMY]
            item["capabilities"] = capabilities or ["table_structure"]

            image_answer = validation.get("image_answer")
            recomputed_match = (
                image_answer is not None
                and normalize_exact(image_answer) == normalize_exact(item.get("gold_answer"))
            )
            validation["image_matches_gold"] = recomputed_match

            entry = {**item, "validation": validation}

            if recomputed_match:
                accepted_local.append(item)
                continue

            if args.no_refine:
                if args.auto_fix_gold and image_answer is not None:
                    item["gold_answer"] = (
                        json.dumps(image_answer, ensure_ascii=False)
                        if isinstance(image_answer, (dict, list))
                        else image_answer
                    )
                    entry["auto_fix_applied"] = True
                    accepted_local.append(item)
                else:
                    rejected_local.append(entry)
                continue

            refine_cache = args.artifact_dir / "refine" / f"{item['question_id']}.json"
            try:
                refine_payload = cached_call(
                    refine_cache,
                    lambda: refine_question(image_path, item, validation),
                )
            except Exception as exc:
                refine_payload = {"action": "discard", "reason": f"refine failed: {exc}", "refined": None}

            refined = refine_payload.get("refined") if refine_payload.get("action") == "refine" else None
            if not refined:
                entry["refine"] = refine_payload
                rejected_local.append(entry)
                continue

            refined["question_id"] = item["question_id"]
            refined["source_file"] = source_file
            refined.setdefault("answer_type", "hard_structural_exact")
            refined.setdefault("difficulty", "very_hard")
            refined_caps = [t for t in refined.get("capabilities", []) if t in CAPABILITY_TAXONOMY]
            refined["capabilities"] = refined_caps or item["capabilities"]

            revalidate_cache = args.artifact_dir / "revalidate" / f"{item['question_id']}.json"
            try:
                revalidation = cached_call(
                    revalidate_cache,
                    lambda: validate_question(image_path, refined),
                )
            except Exception as exc:
                revalidation = {
                    "image_answer": None,
                    "gold_answer": refined.get("gold_answer"),
                    "image_matches_gold": False,
                    "error": str(exc),
                }
            revalid_answer = revalidation.get("image_answer")
            revalid_match = (
                revalid_answer is not None
                and normalize_exact(revalid_answer) == normalize_exact(refined.get("gold_answer"))
            )
            revalidation["image_matches_gold"] = revalid_match

            entry["refine"] = refine_payload
            entry["revalidation"] = revalidation

            if revalid_match:
                refined["refined_from"] = {
                    "original_question": item.get("question"),
                    "original_gold": item.get("gold_answer"),
                    "reason": refine_payload.get("reason"),
                }
                accepted_local.append(refined)
            else:
                rejected_local.append(entry)

        log(
            f"[{image_idx:>2}/{len(images)}] {source_file}  "
            f"accepted={len(accepted_local)} rejected={len(rejected_local)}"
        )
        return {"accepted": accepted_local, "rejected": rejected_local, "error": None}

    # Run images concurrently. Cache short-circuits images that already have
    # full artifacts on disk; the executor still walks them but each finishes
    # in milliseconds.
    futures_by_idx: dict[int, concurrent.futures.Future[dict[str, Any]]] = {}
    workers = max(1, args.max_workers)
    log(f"Running {len(images)} images with {workers} worker(s)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for image_idx, image_path in enumerate(images, start=1):
            futures_by_idx[image_idx] = pool.submit(process_image, image_idx, image_path)

        # Aggregate in image_idx order so bank ordering is deterministic.
        for image_idx in sorted(futures_by_idx):
            result = futures_by_idx[image_idx].result()
            accepted.extend(result["accepted"])
            rejected.extend(result["rejected"])
            if result["error"]:
                image_errors.append(result["error"])

    accepted_after_refine = sum(1 for item in accepted if "refined_from" in item)
    bank = {
        "doc_id": args.doc_id,
        "question_set": args.question_set,
        "items": accepted,
        "generation_metadata": {
            "model": MODEL,
            "questions_per_image": args.questions_per_image,
            "image_count": len(images),
            "candidate_count": sum(1 for _ in accepted) + sum(1 for _ in rejected),
            "accepted": len(accepted),
            "accepted_after_refine": accepted_after_refine,
            "rejected": len(rejected),
            "skipped_images": len(image_errors),
            "exemplar_count": len(exemplars),
            "refine_enabled": not args.no_refine,
            "elapsed_sec": time.perf_counter() - started,
            "auto_fix_gold": args.auto_fix_gold,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n")

    rejected_path = args.artifact_dir / "rejected.json"
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.write_text(json.dumps(rejected, indent=2, ensure_ascii=False) + "\n")

    if image_errors:
        errors_path = args.artifact_dir / "image_errors.json"
        errors_path.write_text(json.dumps(image_errors, indent=2, ensure_ascii=False) + "\n")
        print(f"  ({len(image_errors)} images skipped; see {errors_path})", flush=True)

    print(
        f"Wrote {args.out} with {len(accepted)} accepted questions "
        f"({len(rejected)} rejected, logged at {rejected_path})",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
