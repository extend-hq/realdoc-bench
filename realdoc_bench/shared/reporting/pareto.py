"""Pareto frontier extraction for quality-vs-cost / quality-vs-latency plots.

A point ``(x, y)`` is **on the frontier** when no other point has lower or equal x
AND higher or equal y, with at least one strict. x is the cost/latency axis
(lower is better); y is the quality axis (higher is better).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParetoPoint:
    label: str
    x: float
    y: float
    on_frontier: bool


def extract(points: list[tuple[str, float, float]]) -> list[ParetoPoint]:
    """Mark which (label, x, y) points lie on the upper-left frontier.

    Lower x dominates at equal/better y. Returns the same labels in the same
    order with on_frontier flagged.
    """
    n = len(points)
    on = [True] * n
    for i, (_, xi, yi) in enumerate(points):
        for j, (_, xj, yj) in enumerate(points):
            if i == j:
                continue
            # j dominates i: j is at-least-as-cheap AND at-least-as-good,
            # with at least one strict.
            if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                on[i] = False
                break
    return [ParetoPoint(label=lbl, x=x, y=y, on_frontier=on[i]) for i, (lbl, x, y) in enumerate(points)]


def frontier(points: list[tuple[str, float, float]]) -> list[ParetoPoint]:
    """Return only the points that lie on the frontier, sorted by x ascending."""
    return sorted(
        (p for p in extract(points) if p.on_frontier),
        key=lambda p: p.x,
    )
