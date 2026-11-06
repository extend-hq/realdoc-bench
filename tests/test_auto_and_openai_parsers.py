from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from realdoc_bench.evaluate.parsers.cloud_vlm import OpenAIGpt61SolParser
from realdoc_bench.evaluate.parsers.extend import ExtendAutoV1


class _FakeExtend:
    last_config: dict[str, Any] = {}

    def __init__(self, *, token: str, base_url: str) -> None:
        self.files = SimpleNamespace(upload=lambda file: SimpleNamespace(id="file_123"))
        self.parse_runs = SimpleNamespace(create_and_poll=self._create_and_poll)

    @staticmethod
    def _create_and_poll(*, file: dict[str, Any], config: dict[str, Any]) -> Any:
        _FakeExtend.last_config = config
        return SimpleNamespace(
            output=SimpleNamespace(chunks=[SimpleNamespace(content="# Page 1"), SimpleNamespace(content="Page 2")]),
            metrics=SimpleNamespace(page_count=2),
            usage=SimpleNamespace(credits=2.5),
        )


def test_extend_auto_parser_meters_cost_from_reported_credits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EXTEND_API_KEY", "test-key")
    monkeypatch.setattr("extend_ai.Extend", _FakeExtend)

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    result = ExtendAutoV1().parse(pdf)

    assert _FakeExtend.last_config["engine"] == "parse_auto"
    assert _FakeExtend.last_config["engineVersion"] == "1.0.0"
    assert result.markdown == "# Page 1\n\nPage 2"
    assert result.page_count == 2
    assert result.cost_estimate_usd == pytest.approx(0.025)


class _FakeResponses:
    last_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        _FakeResponses.last_kwargs = kwargs
        return SimpleNamespace(
            output_text="```markdown\n# Title\n```",
            usage=SimpleNamespace(input_tokens=1000, output_tokens=500),
        )


class _FakeOpenAI:
    def __init__(self, *, api_key: str | None) -> None:
        self.responses = _FakeResponses()


def test_openai_parser_sends_high_detail_page_image(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)

    png = tmp_path / "page.png"
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(buf, format="PNG")
    png.write_bytes(buf.getvalue())

    result = OpenAIGpt61SolParser().parse(png)

    kwargs = _FakeResponses.last_kwargs
    assert kwargs["model"] == "gpt-6.1-sol"
    image, text = kwargs["input"][0]["content"]
    assert image["type"] == "input_image"
    assert image["detail"] == "high"
    assert image["image_url"].startswith("data:image/png;base64,")
    assert text["type"] == "input_text"
    assert result.markdown == "## Page 1\n\n# Title"
    assert result.cost_estimate_usd == pytest.approx(1000 * 0.002 / 1000 + 500 * 0.010 / 1000)
