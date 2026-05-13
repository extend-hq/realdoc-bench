from __future__ import annotations

from realdoc_bench.shared.reporting.pareto import extract, frontier


def test_obvious_dominance():
    pts = [("cheap_good", 1.0, 0.9), ("cheap_bad", 1.0, 0.5), ("expensive_good", 5.0, 0.9)]
    out = {p.label: p.on_frontier for p in extract(pts)}
    assert out["cheap_good"] is True
    assert out["cheap_bad"] is False
    # expensive_good is tied on quality with cheap_good and strictly worse cost → dominated.
    assert out["expensive_good"] is False


def test_frontier_includes_all_pareto():
    pts = [("a", 1.0, 0.6), ("b", 2.0, 0.8), ("c", 3.0, 0.9), ("d", 4.0, 0.7)]
    labels_on = {p.label for p in frontier(pts)}
    assert labels_on == {"a", "b", "c"}


def test_single_point_is_on_frontier():
    pts = [("only", 1.0, 0.5)]
    out = extract(pts)
    assert out[0].on_frontier is True
