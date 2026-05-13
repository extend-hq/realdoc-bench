"""Scoring profiles — knobs the user can pick by name."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringProfile:
    """Per-leaf-type knobs.

    Attributes:
        fuzzy_threshold: rapidfuzz ``token_set_ratio`` (0..100) at or above which a
            string is counted as a fuzzy match. Below ``fuzzy_floor`` is 0; in
            between, linear partial credit from ``fuzzy_floor_score`` to 1.0.
        fuzzy_floor: ratio below which the score collapses to 0.
        fuzzy_floor_score: score at the floor (typically a partial-credit anchor).
        number_rel_tolerance: relative tolerance for ``number`` leaves; overridable
            per-field via ``x-tolerance`` in the schema.
        ignore_punctuation: when True, strings are normalized with punctuation
            stripped — useful for the default profile (an extra comma shouldn't
            tank the score). Strict profile sets this to False.
    """

    fuzzy_threshold: int = 90
    fuzzy_floor: int = 70
    fuzzy_floor_score: float = 0.7
    number_rel_tolerance: float = 0.005
    ignore_punctuation: bool = True


DEFAULT_PROFILE = ScoringProfile()
STRICT_PROFILE = ScoringProfile(
    fuzzy_threshold=100,
    fuzzy_floor=100,
    fuzzy_floor_score=1.0,
    number_rel_tolerance=0.0,
    ignore_punctuation=False,
)
LENIENT_PROFILE = ScoringProfile(
    fuzzy_threshold=80,
    fuzzy_floor=50,
    fuzzy_floor_score=0.5,
    number_rel_tolerance=0.05,
    ignore_punctuation=True,
)

PROFILES: dict[str, ScoringProfile] = {
    "default": DEFAULT_PROFILE,
    "strict": STRICT_PROFILE,
    "lenient": LENIENT_PROFILE,
}
