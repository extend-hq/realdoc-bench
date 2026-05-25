"""Hungarian assignment helper used by the adjacency matcher."""

from __future__ import annotations

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
