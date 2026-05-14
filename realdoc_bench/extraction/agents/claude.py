"""Claude Sonnet 4.6 extractor agent (tool-use mode).

The agent is given the dialect-stripped JSON Schema as a tool's
``input_schema``. Claude is forced to invoke the tool, so its response is
guaranteed to be a JSON object validated against the schema (modulo Claude's
own validation, which we re-check with ``jsonschema`` afterwards).
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


@register_agent("claude_sonnet_4_7", version="1.0.0")
class ClaudeExtractor(ExtractionAgent):
    model = "claude-sonnet-4-7"
    max_tokens = 4096

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return self._client

    def extract(
        self,
        markdown: str,
        schema: dict[str, Any],
        doc_id: str,
    ) -> AgentResult:
        del doc_id  # not used in the prompt today
        plain_schema = to_plain_jsonschema(schema)
        tool = {
            "name": "submit_extraction",
            "description": "Submit the extracted fields as a JSON object matching the schema.",
            "input_schema": plain_schema,
        }
        client = self._ensure_client()

        retries = 0
        last_error: str | None = None
        usage_total = {"input_tokens": 0, "output_tokens": 0}
        latency_total = 0.0
        for attempt in range(2):  # one retry on schema violation
            user = _render_user(markdown, plain_schema)
            if last_error is not None:
                user += f"\n\nPrevious attempt failed validation:\n{last_error}\n\nFix and retry."
            t0 = time.perf_counter()
            try:
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=_SYSTEM,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": "submit_extraction"},
                    messages=[{"role": "user", "content": user}],
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

            usage = getattr(resp, "usage", None)
            if usage is not None:
                usage_total["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
                usage_total["output_tokens"] += getattr(usage, "output_tokens", 0) or 0

            tool_block = next(
                (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
                None,
            )
            if tool_block is None:
                last_error = "Claude did not call the submit_extraction tool."
                retries = attempt + 1
                continue
            value = tool_block.input
            if not isinstance(value, dict):
                last_error = f"Tool input was {type(value).__name__}, expected object."
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
                    "claude_sonnet_4_7",
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
                "claude_sonnet_4_7",
                input_tokens=usage_total["input_tokens"],
                output_tokens=usage_total["output_tokens"],
            ),
            retries=retries,
            failure_reason=last_error,
        )
