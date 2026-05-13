"""Markdown leaderboard writer."""

from __future__ import annotations

from typing import Any


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 100 else f"{value:.1f}"
    return str(value)


def render_table(
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    sort_by: str | None = None,
    descending: bool = True,
) -> str:
    if sort_by is not None:
        rows = sorted(rows, key=lambda r: (r.get(sort_by) is None, r.get(sort_by)), reverse=descending)
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = "\n".join("| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def render_section(title: str, body: str, *, level: int = 2) -> str:
    return f"{'#' * level} {title}\n\n{body}\n"
