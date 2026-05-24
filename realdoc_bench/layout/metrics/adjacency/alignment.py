"""Hungarian assignment helper used by the adjacency matcher."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian(
    cost_matrix: list[list[float]],
) -> tuple[list[tuple[int, int]], float]:
    """Simple Hungarian wrapper using scipy if available; Returns (assignments, total_cost)."""

    cm = np.asarray(cost_matrix, dtype=float)
    ri, ci = linear_sum_assignment(cm)
    pairs = list(zip(ri.tolist(), ci.tolist()))
    total_cost = float(cm[ri, ci].sum())
    return pairs, total_cost


def make_cost_matrix(
    n_rows: int,
    n_cols: int,
    cost_fn: Callable[[int, int], float],
) -> list[list[float]]:
    return [[cost_fn(r, c) for c in range(n_cols)] for r in range(n_rows)]
