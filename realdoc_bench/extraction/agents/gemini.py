"""Gemini 3 Flash extractor agent.

Uses the Gemini ``responseSchema`` capability to constrain output to the
dialect-stripped JSON Schema. The agent never sees the source PDF.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from realdoc_bench.extraction.agents.base import (
    AgentResult,
    ExtractionAgent,
    register_agent,
)
from realdoc_bench.extraction.schema.dialect import to_plain_jsonschema
from realdoc_bench.extraction.schema.validate import validate
from realdoc_bench.shared.pricing.meter import agent_cost

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_SYSTEM = (_PROMPTS_DIR / "extractor_system.md").read_text()
_USER_TEMPLATE = (_PROMPTS_DIR / "extractor_user.md").read_text()


def _render_user(markdown: str, schema: dict[str, Any]) -> str:
    return _USER_TEMPLATE.replace("{{ markdown }}", markdown).replace(
        "{{ schema_json }}", json.dumps(schema, indent=2)
    )


@register_agent("gemini_3_flash", version="1.0.0")
class GeminiExtractor(ExtractionAgent):
    model = "gemini-3-flash-preview"

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        return self._client

    def extract(
        self,
        markdown: str,
        schema: dict[str, Any],
        doc_id: str,
    ) -> AgentResult:
        del doc_id
        plain_schema = to_plain_jsonschema(schema)
        client = self._ensure_client()

        retries = 0
        last_error: str | None = None
        latency_total = 0.0
        usage_total = {"input_tokens": 0, "output_tokens": 0}
        for attempt in range(2):
            user = _render_user(markdown, plain_schema)
            if last_error is not None:
                user += f"\n\nPrevious attempt failed validation:\n{last_error}\n\nFix and retry."
            t0 = time.perf_counter()
            try:
                resp = client.models.generate_content(
                    model=self.model,
                    contents=[{"role": "user", "parts": [{"text": user}]}],
                    config={
                        "system_instruction": _SYSTEM,
                        "response_mime_type": "application/json",
                        "response_schema": plain_schema,
                    },
                )
            except Exception as e:
                latency_total += time.perf_counter() - t0
                return AgentResult(
                    value=None,
                    status="FAILED",
                    latency_sec=latency_total,
                    usage=usage_total,
                    retries=attempt,
                    failure_reason=f"{type(e).__name__}: {e}",
                )
            latency_total += time.perf_counter() - t0

            usage = getattr(resp, "usage_metadata", None)
            if usage is not None:
                usage_total["input_tokens"] += getattr(usage, "prompt_token_count", 0) or 0
                usage_total["output_tokens"] += getattr(usage, "candidates_token_count", 0) or 0

            text = getattr(resp, "text", None)
            if not text:
                last_error = "Empty response from Gemini."
                retries = attempt + 1
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON: {e}"
                retries = attempt + 1
                continue
            if not isinstance(value, dict):
                last_error = f"Output was {type(value).__name__}, expected object."
                retries = attempt + 1
                continue
            errors = validate(value, plain_schema)
            if errors:
                last_error = "; ".join(errors[:5])
                retries = attempt + 1
                continue
            return AgentResult(
                value=value,
                status="PROCESSED",
                latency_sec=latency_total,
                usage=usage_total,
                cost_estimate_usd=agent_cost(
                    "gemini_3_flash",
                    input_tokens=usage_total["input_tokens"],
                    output_tokens=usage_total["output_tokens"],
                ),
                retries=attempt,
            )
        return AgentResult(
            value=None,
            status="SCHEMA_VIOLATION",
            latency_sec=latency_total,
            usage=usage_total,
            cost_estimate_usd=agent_cost(
                "gemini_3_flash",
                input_tokens=usage_total["input_tokens"],
                output_tokens=usage_total["output_tokens"],
            ),
            retries=retries,
            failure_reason=last_error,
        )
