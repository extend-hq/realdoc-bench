from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from realdoc_bench.evaluate.parsers.mistral_ocr import MistralOCR4Parser


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "model": "mistral-ocr-latest",
            "pages": [
                {"markdown": "# Page 1\n\nHello"},
                {"markdown": "Page 2"},
            ],
            "usage_info": {"pages_processed": 2},
        }


class _FakeClient:
    last_url: str = ""
    last_headers: dict[str, str] = {}
    last_payload: dict[str, Any] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> _FakeResponse:
        _FakeClient.last_url = url
        _FakeClient.last_headers = headers
        _FakeClient.last_payload = json
        return _FakeResponse()


def test_mistral_ocr_4_parser_posts_pdf_data_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("MISTRAL_OCR_BASE_URL", "https://mistral.test")
    monkeypatch.setattr("realdoc_bench.evaluate.parsers.mistral_ocr.httpx.Client", _FakeClient)

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    result = MistralOCR4Parser().parse(pdf)

    assert result.markdown == "# Page 1\n\nHello\n\n---\n\nPage 2"
    assert result.page_count == 2
    assert result.cost_estimate_usd == 0.008
    assert _FakeClient.last_url == "https://mistral.test/v1/ocr"
    assert _FakeClient.last_headers["Authorization"] == "Bearer test-key"
    assert _FakeClient.last_payload["model"] == "mistral-ocr-latest"
    doc = _FakeClient.last_payload["document"]
    assert doc["type"] == "document_url"
    assert doc["document_url"] == (
        "data:application/pdf;base64,"
        + base64.b64encode(pdf.read_bytes()).decode("ascii")
    )
