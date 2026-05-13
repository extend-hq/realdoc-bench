"""Array bipartite matching for ``array_object`` fields.

Given expected + actual lists of objects, build a pairwise similarity matrix
(``mean(per-field scores)``) and solve the Hungarian assignment. Unmatched
expected entries are misses; unmatched actual entries are extras.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_match(
    expected: list[Any],
    actual: list[Any],
    *,
    score_pair: Callable[[Any, Any], float],
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Match elements with the Hungarian algorithm maximizing total pair score.

    Returns ``(matches, unmatched_expected_idx, unmatched_actual_idx)`` where
    ``matches`` is a list of ``(expected_idx, actual_idx, score)`` tuples.
    """
    n_e, n_a = len(expected), len(actual)
    if n_e == 0 or n_a == 0:
        return [], list(range(n_e)), list(range(n_a))

    scores = np.zeros((n_e, n_a), dtype=np.float64)
    for i, e in enumerate(expected):
        for j, a in enumerate(actual):
            scores[i, j] = float(score_pair(e, a))

    cost = 1.0 - scores
    row_ind, col_ind = linear_sum_assignment(cost)

    matches: list[tuple[int, int, float]] = []
    used_e: set[int] = set()
    used_a: set[int] = set()
    for r, c in zip(row_ind, col_ind):
        if scores[r, c] <= 0.0:
            continue
        matches.append((int(r), int(c), float(scores[r, c])))
        used_e.add(int(r))
        used_a.add(int(c))
    unmatched_e = [i for i in range(n_e) if i not in used_e]
    unmatched_a = [i for i in range(n_a) if i not in used_a]
    return matches, unmatched_e, unmatched_a
