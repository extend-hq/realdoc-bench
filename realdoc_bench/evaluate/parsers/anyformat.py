"""anyformat parse adapter — uploads a PDF to a one-node parse workflow, gets markdown back.

The public `/v3/parse/` operation always runs the platform's Fast tier, so this adapter creates
its own workflow instead and names the tier it wants. Page count comes from the blocks the parse
returns, and cost from the published credits-per-page of that tier.

Auth: ``ANYFORMAT_API_KEY``. Base URL: ``ANYFORMAT_BASE_URL`` (defaults to the public API).
Pricing: ``anyformat_standard`` in catalog.yaml.
"""

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from realdoc_bench.evaluate.parsers.base import (
    ParseProvider,
    ParseResult,
    register_parser,
)
from realdoc_bench.shared.io.cache import sha256_text
from realdoc_bench.shared.pricing.meter import parse_cost

_DEFAULT_BASE_URL = "https://api.anyformat.ai"
_POLL_INTERVAL_SECONDS = 4.0
_POLL_INTERVAL_CAP_SECONDS = 30.0
_POLL_TRANSIENT_LIMIT = 3
_RATE_LIMIT_ATTEMPTS = 6
_RATE_LIMIT_FALLBACK_SECONDS = 30.0
_REQUEST_TIMEOUT_SECONDS = 120.0
_JOB_TIMEOUT_SECONDS = 900.0
_TERMINAL_FAILURE_STATUSES = frozenset({"error", "cancelled"})

# The API marks each block with an anchor and each table cell with a citation id, for
# grounding. Neither is content.
_ANCHOR_RE = re.compile(r'<a\s+id="p\d+_b\d+"\s*>\s*</a>\n?')
_DATA_ATTR_RE = re.compile(r'\s+data-[a-z][a-z0-9-]*="[^"]*"')


def clean_markdown(markdown: str) -> str:
    return _DATA_ATTR_RE.sub("", _ANCHOR_RE.sub("", markdown or ""))


def _retry_after(response: Any) -> float:
    """The wait the server asks for: its header first, then the seconds named in the body."""
    header = response.headers.get("Retry-After") if hasattr(response, "headers") else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = re.search(r"(\d+)\s*s", str(getattr(response, "text", "")))
    return float(match.group(1)) if match else _RATE_LIMIT_FALLBACK_SECONDS


def page_count(parse: dict[str, Any]) -> int | None:
    """The document's page count, read off the pages its blocks name."""
    pages = [int(b.get("page") or 0) for b in (parse.get("blocks") or []) if isinstance(b, dict)]
    return max(pages) if pages and max(pages) > 0 else None


@register_parser("anyformat_standard", version="1.0.0")
class AnyformatParser(ParseProvider):
    """anyformat's Standard tier: layout detection, then a page-level vision pass per page."""

    mode: str = "standard"

    def __init__(self) -> None:
        self._client: Any | None = None
        self._workflow_id: str | None = None
        # The runner shares one provider across its workers; unguarded, every thread would
        # create its own workflow.
        self._lock = threading.Lock()
        self._client_lock = threading.Lock()

    def config_hash(self) -> str:
        return sha256_text(f"{self.name}|{self.version}|mode={self.mode}")[:7]

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            import httpx

            api_key = os.environ.get("ANYFORMAT_API_KEY")
            if not api_key:
                raise RuntimeError("ANYFORMAT_API_KEY not set")
            base_url = (os.environ.get("ANYFORMAT_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
            self._client = httpx.Client(
                base_url=base_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            return self._client

    def _request(self, method: str, url: str, what: str, **kwargs: Any) -> Any:
        """One request, waiting out the 429 the server answers a long run with."""
        client = self._ensure_client()
        for attempt in range(_RATE_LIMIT_ATTEMPTS):
            response = getattr(client, method)(url, **kwargs)
            if response.status_code != 429:
                return response
            if attempt == _RATE_LIMIT_ATTEMPTS - 1:
                break
            time.sleep(_retry_after(response))
        raise RuntimeError(f"anyformat {what} rate limited after {_RATE_LIMIT_ATTEMPTS} attempts")

    def _json(self, response: Any, what: str) -> dict[str, Any]:
        if response.status_code >= 400:
            raise RuntimeError(f"anyformat {what} failed: {response.status_code} {response.text[:300]}")
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError(f"anyformat {what} returned {type(body).__name__}, expected an object")
        return body

    def _ensure_workflow(self) -> str:
        """The workflow this adapter parses through, created once and reused after that."""
        with self._lock:
            if self._workflow_id:
                return self._workflow_id
            self._workflow_id = self._create_or_adopt()
            return self._workflow_id

    def _create_or_adopt(self) -> str:
        name = f"realdocbench-{self.mode}"
        response = self._request(
            "post",
            "/v3/workflows/",
            "workflow create",
            json={
                "name": name,
                "description": "realdoc-bench harness: one-node parse workflow",
                "nodes": [{"id": "parse_1", "type": "parse", "mode": self.mode, "cache": False}],
                "edges": [],
            },
        )
        if response.status_code == 409:
            workflow_id = self._verified_workflow(self._find_workflow(name))
        else:
            workflow_id = self._json(response, "workflow create").get("id")
            if not isinstance(workflow_id, str) or not workflow_id:
                raise RuntimeError("anyformat workflow create returned no id")
        return workflow_id

    def _find_workflow(self, name: str) -> str:
        """The workflow of this name, on any page of the listing. The server ignores case when
        it reports the conflict, so the comparison does too."""
        wanted = name.casefold()
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            params: dict[str, Any] = {"limit": 100}
            if cursor:
                params["cursor"] = cursor
            listed = self._json(
                self._request("get", "/v3/workflows/", "workflow list", params=params), "workflow list"
            )
            for workflow in listed.get("items") or []:
                if isinstance(workflow, dict) and str(workflow.get("name", "")).casefold() == wanted:
                    workflow_id = workflow.get("id")
                    if isinstance(workflow_id, str) and workflow_id:
                        return workflow_id
            nxt = listed.get("next_cursor")
            cursor = nxt if isinstance(nxt, str) and nxt else None
            if not cursor or cursor in seen:
                break
            seen.add(cursor)
        raise RuntimeError(f"anyformat workflow {name!r} conflicts but is not in the listing")

    def _verified_workflow(self, workflow_id: str) -> str:
        """The id back, once its graph says it parses what this adapter claims to measure."""
        detail = self._json(
            self._request("get", f"/v3/workflows/{workflow_id}/", "workflow read"), "workflow read"
        )
        nodes = [n for n in (detail.get("nodes") or []) if isinstance(n, dict) and n.get("type") == "parse"]
        if len(nodes) != 1 or nodes[0].get("mode") != self.mode or nodes[0].get("cache"):
            raise RuntimeError(
                f"anyformat workflow {workflow_id} does not parse as {self.mode} with caching off: {nodes}"
            )
        return workflow_id

    def _submit(self, workflow_id: str, pdf_path: Path) -> str:
        # A retry re-sends the file, so it is a new request: replaying one makes the API sign
        # an upload the object store refuses.
        key = uuid.uuid4().hex
        with open(pdf_path, "rb") as handle:
            response = self._request(
                "post",
                f"/v3/workflows/{workflow_id}/upload/run/",
                "upload/run",
                files=[("files", (pdf_path.name, handle, "application/pdf"))],
                data={"on_conflict": "rename"},
                headers={"Idempotency-Key": key},
            )
        run_id = self._json(response, "upload/run").get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("anyformat upload/run returned no run_id")
        return run_id

    def _poll(self, run_id: str) -> dict[str, Any]:
        """The finished run. A failure to ask is not a failure of the run, so a few are waited
        out rather than taken for one."""
        deadline = time.monotonic() + _JOB_TIMEOUT_SECONDS
        wait = _POLL_INTERVAL_SECONDS
        transient = 0
        while time.monotonic() < deadline:
            time.sleep(wait)
            wait = min(wait * 1.5, _POLL_INTERVAL_CAP_SECONDS)
            try:
                run = self._json(self._request("get", f"/v3/runs/{run_id}/", "run poll"), "run poll")
            except RuntimeError:
                transient += 1
                if transient > _POLL_TRANSIENT_LIMIT:
                    raise
                continue
            transient = 0
            status = run.get("status")
            if status == "processed":
                return run
            if status in _TERMINAL_FAILURE_STATUSES:
                raise RuntimeError(f"anyformat run {run_id} ended with status={status}")
        raise RuntimeError(f"anyformat run {run_id} did not finish within {_JOB_TIMEOUT_SECONDS:.0f}s")

    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
        del cache_dir
        started = time.perf_counter()
        run = self._poll(self._submit(self._ensure_workflow(), pdf_path))
        latency = time.perf_counter() - started

        results = run.get("results")
        parse = results.get("parse") if isinstance(results, dict) else None
        parse = parse if isinstance(parse, dict) else {}
        pages = page_count(parse)
        markdown = clean_markdown(str(parse.get("markdown") or ""))
        if not markdown.strip():
            raise RuntimeError(f"anyformat run returned no markdown for {pdf_path.name}")

        return ParseResult(
            markdown=markdown,
            page_count=pages,
            latency_sec=latency,
            cost_estimate_usd=parse_cost(self.name, pages=pages),
            pages_processed=pages,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
        )
