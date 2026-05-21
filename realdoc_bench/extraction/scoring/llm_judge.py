"""LLM-as-judge scoring layer.

Gemini adjudicates whether a parser's answer is semantically equivalent to
the gold answer. Used as the third scoring layer — only for the cells the
cheaper strict + fuzzy layers disagree on. Deterministic (temperature 0).

Auth: ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

JUDGE_MODEL = "gemini-3-flash-preview"

JUDGE_PROMPT = """You are grading a structured-data extraction benchmark.

Determine if the PARSER's answer is semantically equivalent to the GOLD answer.

The gold format is plain-string:
  "key1=value1; key2=value2; ..."  (paired values)
  "value1; value2; value3"          (ordered values, keys implied)

A MATCH means:
- Same set of keys (or same number of ordered items in same positions)
- Each value semantically equals the gold value, ignoring formatting:
  * Currency symbols ($1,234.56 ≈ 1234.56)
  * Decimal precision when gold is whole (5 ≈ 5.0)
  * Date format (01/15/2024 ≈ Jan 15, 2024 ≈ 2024-01-15)
  * Case (Apple Inc. ≈ apple inc.)
  * Trailing punctuation, extra whitespace
  * Synonymous values: yes/Yes/Y/checked/marked/✓/X; no/No/N/unchecked; null/None/N/A/blank
  * Minor abbreviations (Inc. vs Inc, U.S. vs US)

NOT a match:
- Any key missing or extra
- Different number value (12 vs 13)
- Different proper noun (Apple vs Amazon)
- Different date by more than format (Jan 15 vs Jan 16)
- Answer is "null" but gold has a non-null value (or vice versa)
- Answer omits a value the gold has

Return ONLY JSON: {"match": <true|false>, "reasoning": "<one short sentence>"}"""


def llm_judge(
    answer: Any,
    gold: Any,
    question: str,
    *,
    model: str = JUDGE_MODEL,
    api_key: str | None = None,
) -> dict:
    """Ask Gemini whether ``answer`` is semantically equivalent to ``gold``.

    Returns ``{"match": bool, "reasoning": str}``; on any failure also sets
    ``"error": True`` so callers can tell a real verdict from a fallback.
    """
    api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"match": False, "reasoning": "no GEMINI_API_KEY", "error": True}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": JUDGE_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text":
            f"Question: {question}\n\nGOLD: {gold}\nPARSER ANSWER: {answer}"
        }]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 256,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    try:
        with httpx.Client(timeout=60) as client:
            r = client.post(url, params={"key": api_key}, json=body)
            if r.status_code != 200:
                return {"match": False, "reasoning": f"http {r.status_code}", "error": True}
            data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {"match": False, "reasoning": "no JSON in response", "error": True}
        obj = json.loads(m.group(0))
        return {"match": bool(obj.get("match")), "reasoning": obj.get("reasoning", "")}
    except Exception as e:  # noqa: BLE001 — any failure → graceful fallback
        return {"match": False, "reasoning": f"{type(e).__name__}: {e}", "error": True}
