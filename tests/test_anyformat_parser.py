from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from realdoc_bench.evaluate.parsers.anyformat import AnyformatParser

STANDARD_NODE = {"id": "parse_1", "type": "parse", "mode": "standard", "cache": False}
MARKDOWN = '<a id="p1_b0"></a>\n# Title\n\n<td data-cell-id="r0c0">7</td>'


class _Response:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self) -> Any:
        return self._body


class _Client:
    def __init__(
        self,
        *,
        workflow_conflict: bool = False,
        listed_name: str = "realdocbench-standard",
        pages: int = 2,
        nodes: dict[str, Any] | None = None,
        markdown: str = MARKDOWN,
        run_status: str = "processed",
        second_page: bool = False,
        rate_limited: int = 0,
    ) -> None:
        self.workflow_conflict = workflow_conflict
        self.listed_name = listed_name
        self.pages = pages
        self.nodes = STANDARD_NODE if nodes is None else nodes
        self.markdown = markdown
        self.run_status = run_status
        self.second_page = second_page
        self.created_workflows: list[dict[str, Any]] = []
        self.rate_limited = rate_limited
        self.waited_out = 0
        self.creating = 0
        self.most_creating_at_once = 0
        self.uploads: list[dict[str, Any]] = []
        self.listings: list[dict[str, Any] | None] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        if self.rate_limited:
            self.rate_limited -= 1
            self.waited_out += 1
            return _Response(429, {"detail": "Rate limit exceeded. Try again in 7s."})
        if url == "/v3/workflows/":
            if self.workflow_conflict:
                return _Response(409, {"error_code": "WORKFLOW_NAME_CONFLICT"})
            self.creating += 1
            self.most_creating_at_once = max(self.most_creating_at_once, self.creating)
            # Long enough that any thread not held out by the lock arrives while this one is
            # still inside; `time.sleep` is stubbed in these tests, so wait on an event.
            threading.Event().wait(0.05)
            self.creating -= 1
            self.created_workflows.append(kwargs["json"])
            return _Response(201, {"id": "wf-1"})
        if url.endswith("/upload/run/"):
            files = kwargs["files"]
            self.uploads.append(
                {"name": files[0][1][0], "bytes": files[0][1][1].read(), "headers": kwargs.get("headers", {})}
            )
            return _Response(202, {"run_id": "run-1"})
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url: str, **kwargs: Any) -> _Response:
        if url == "/v3/workflows/":
            params = kwargs.get("params")
            self.listings.append(params)
            if self.second_page and not (params or {}).get("cursor"):
                return _Response(200, {"items": [{"id": "other", "name": "unrelated"}], "next_cursor": "c2"})
            return _Response(
                200, {"items": [{"id": "wf-existing", "name": self.listed_name}], "next_cursor": None}
            )
        if url.startswith("/v3/workflows/"):
            return _Response(
                200, {"id": "wf-existing", "name": self.listed_name, "nodes": [self.nodes], "edges": []}
            )
        if url.startswith("/v3/runs/"):
            return _Response(
                200,
                {
                    "status": self.run_status,
                    "results": {
                        "parse": {
                            "markdown": self.markdown,
                            "blocks": [{"page": n + 1} for n in range(self.pages)],
                        }
                    },
                },
            )
        raise AssertionError(f"unexpected GET {url}")


@pytest.fixture
def parser(monkeypatch: pytest.MonkeyPatch) -> AnyformatParser:
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    return AnyformatParser()


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    document = tmp_path / "doc.pdf"
    document.write_bytes(b"%PDF-1.4\nbody")
    return document


def test_the_parse_asks_for_the_standard_tier(parser: AnyformatParser, pdf: Path) -> None:
    client = _Client()
    parser._client = client

    parser.parse(pdf)

    assert client.created_workflows[0]["nodes"][0]["mode"] == "standard"
    assert client.created_workflows[0]["nodes"][0]["cache"] is False


def test_the_document_is_uploaded_with_its_bytes(parser: AnyformatParser, pdf: Path) -> None:
    client = _Client()
    parser._client = client

    parser.parse(pdf)

    assert client.uploads[0]["bytes"] == pdf.read_bytes()
    assert client.uploads[0]["name"] == "doc.pdf"


def test_each_upload_carries_its_own_idempotency_key(parser: AnyformatParser, pdf: Path) -> None:
    first, second = _Client(), _Client()
    parser._client = first
    parser.parse(pdf)
    AnyformatParser.__init__(parser)
    parser._client = second
    parser.parse(pdf)

    assert first.uploads[0]["headers"]["Idempotency-Key"] != second.uploads[0]["headers"]["Idempotency-Key"]


def test_a_second_run_reuses_the_workflow_the_first_one_created(parser: AnyformatParser, pdf: Path) -> None:
    parser._client = _Client(workflow_conflict=True)

    result = parser.parse(pdf)

    assert parser._workflow_id == "wf-existing"
    assert result.markdown


def test_the_workflow_is_found_on_a_later_page_of_the_listing(parser: AnyformatParser, pdf: Path) -> None:
    client = _Client(workflow_conflict=True, second_page=True)
    parser._client = client

    parser.parse(pdf)

    assert parser._workflow_id == "wf-existing"
    assert client.listings[1]["cursor"] == "c2"


def test_the_workflow_name_is_matched_the_way_the_server_matched_it(
    parser: AnyformatParser, pdf: Path
) -> None:
    parser._client = _Client(workflow_conflict=True, listed_name="RealDocBench-Standard")

    parser.parse(pdf)

    assert parser._workflow_id == "wf-existing"


def test_a_workflow_that_parses_another_tier_is_refused(parser: AnyformatParser, pdf: Path) -> None:
    parser._client = _Client(workflow_conflict=True, nodes={**STANDARD_NODE, "mode": "lite"})

    with pytest.raises(RuntimeError, match="does not parse as standard"):
        parser.parse(pdf)


def test_a_workflow_serving_from_cache_is_refused(parser: AnyformatParser, pdf: Path) -> None:
    parser._client = _Client(workflow_conflict=True, nodes={**STANDARD_NODE, "cache": True})

    with pytest.raises(RuntimeError, match="caching off"):
        parser.parse(pdf)


def test_every_thread_shares_the_one_workflow(parser: AnyformatParser, pdf: Path) -> None:
    client = _Client()
    parser._client = client
    ready = threading.Barrier(4)

    def _parse() -> None:
        ready.wait()
        parser.parse(pdf)

    threads = [threading.Thread(target=_parse) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(client.created_workflows) == 1
    assert client.most_creating_at_once == 1


def test_a_run_that_ends_in_error_says_so(parser: AnyformatParser, pdf: Path) -> None:
    parser._client = _Client(run_status="error")

    with pytest.raises(RuntimeError, match="status=error"):
        parser.parse(pdf)


def test_a_refused_request_is_an_error(parser: AnyformatParser, pdf: Path) -> None:
    client = _Client()
    client.post = lambda *a, **k: _Response(400, {"error": "bad"})  # type: ignore[method-assign]
    parser._client = client

    with pytest.raises(RuntimeError, match="400"):
        parser.parse(pdf)


def test_a_parse_that_returns_no_markdown_is_a_failure(parser: AnyformatParser, pdf: Path) -> None:
    parser._client = _Client(markdown="   ")

    with pytest.raises(RuntimeError, match="no markdown"):
        parser.parse(pdf)


def test_anchors_and_cell_ids_never_reach_the_scorer(parser: AnyformatParser, pdf: Path) -> None:
    parser._client = _Client()

    markdown = parser.parse(pdf).markdown

    assert "<a id=" not in markdown
    assert "data-cell-id" not in markdown
    assert "# Title" in markdown


def test_the_page_count_and_its_cost_come_from_the_parsed_blocks(parser: AnyformatParser, pdf: Path) -> None:
    parser._client = _Client()

    result = parser.parse(pdf)

    assert result.page_count == 2
    assert result.cost_estimate_usd == pytest.approx(0.058)


def test_a_rate_limited_request_is_waited_out_not_failed(parser: AnyformatParser, pdf: Path) -> None:
    client = _Client(rate_limited=2)
    parser._client = client

    result = parser.parse(pdf)

    assert client.waited_out == 2
    assert result.markdown
