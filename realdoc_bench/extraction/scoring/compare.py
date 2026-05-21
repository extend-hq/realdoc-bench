"""Schema-free pairwise string comparison — the shared exact/fuzzy matcher.

The field-level scoring engine (:mod:`engine`) and the benchmark rescorer
both need to compare two single string values. This is the one
implementation they share: normalize → exact, else rapidfuzz
``token_set_ratio`` against the profile's ``fuzzy_threshold`` / ``fuzzy_floor``
(with linear partial credit in between).
"""

from __future__ import annotations

from typing import Any, Literal

from rapidfuzz import fuzz

from realdoc_bench.extraction.scoring.normalizers import normalize_string
from realdoc_bench.extraction.scoring.profiles import DEFAULT_PROFILE, ScoringProfile

MatchKind = Literal["exact", "fuzzy", "mismatch", "null_match"]


def compare_text(
    expected: Any,
    actual: Any,
    *,
    profile: ScoringProfile = DEFAULT_PROFILE,
) -> tuple[float, MatchKind]:
    """Compare two values as strings under ``profile``.

    Returns ``(score, kind)`` where ``score`` is in ``[0, 1]``:
      - both empty            → ``(1.0, "null_match")``
      - one empty             → ``(0.0, "mismatch")``
      - normalized-equal      → ``(1.0, "exact")``
      - ratio ≥ threshold     → ``(1.0, "fuzzy")``
      - ratio < floor         → ``(0.0, "mismatch")``
      - in between            → linear partial credit, ``"fuzzy"``
    """
    exp_n = normalize_string(expected, ignore_punctuation=profile.ignore_punctuation)
    act_n = normalize_string(actual, ignore_punctuation=profile.ignore_punctuation)
    if exp_n is None and act_n is None:
        return 1.0, "null_match"
    if exp_n is None or act_n is None:
        return 0.0, "mismatch"
    if exp_n == act_n:
        return 1.0, "exact"

    ratio = fuzz.token_set_ratio(exp_n, act_n)
    if ratio >= profile.fuzzy_threshold:
        return 1.0, "fuzzy"
    if ratio < profile.fuzzy_floor:
        return 0.0, "mismatch"
    span = max(profile.fuzzy_threshold - profile.fuzzy_floor, 1)
    score = profile.fuzzy_floor_score + (1.0 - profile.fuzzy_floor_score) * (
        (ratio - profile.fuzzy_floor) / span
    )
    return score, "fuzzy"


def is_match(expected: Any, actual: Any, *, profile: ScoringProfile = DEFAULT_PROFILE) -> bool:
    """Binary verdict: full credit only (exact, or fuzzy ≥ threshold)."""
    score, _ = compare_text(expected, actual, profile=profile)
    return score >= 1.0
