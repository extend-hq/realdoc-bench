"""Field-level scoring engine.

Walks the schema, applies a type-appropriate comparator per leaf, and returns a
list of :class:`FieldScore` rows that downstream aggregators turn into F1.

Default rules (see :class:`realdoc_bench.extraction.scoring.profiles.ScoringProfile`):

| leaf_type        | comparator                                                 |
|------------------|------------------------------------------------------------|
| string (no enum) | normalize → exact, else rapidfuzz token_set_ratio          |
| date             | dateutil-parse → ISO compare                               |
| enum             | normalize → exact                                          |
| integer          | exact                                                      |
| number           | abs(rel error) ≤ profile.number_rel_tolerance              |
| boolean          | exact (null ≠ false)                                       |
| array_primitive  | bag equality with element-fuzzy → F1 over matched set      |
| array_object     | bipartite match → score-weighted F1                        |
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from rapidfuzz import fuzz

from realdoc_bench.extraction.schema.walk import MISSING, SchemaLeaf, get_at_path, walk_leaves
from realdoc_bench.extraction.scoring.matchers import hungarian_match
from realdoc_bench.extraction.scoring.normalizers import (
    normalize_boolean,
    normalize_date,
    normalize_integer,
    normalize_number,
    normalize_string,
)
from realdoc_bench.extraction.scoring.profiles import DEFAULT_PROFILE, ScoringProfile


MatchKind = Literal["exact", "fuzzy", "tolerance", "miss", "extra", "null_match", "mismatch"]


@dataclass
class FieldScore:
    path: str
    leaf_type: str
    expected: Any
    actual: Any
    score: float          # in [0, 1]
    match: MatchKind
    capabilities: list[str]
    difficulty: str | None
    required: bool


def score(
    expected: dict[str, Any] | None,
    actual: dict[str, Any] | None,
    schema: dict[str, Any],
    *,
    profile: ScoringProfile = DEFAULT_PROFILE,
) -> list[FieldScore]:
    """Score ``actual`` against ``expected`` under the leaf-walk of ``schema``."""
    rows: list[FieldScore] = []
    for leaf in walk_leaves(schema):
        exp = get_at_path(expected, leaf.path) if expected is not None else MISSING
        act = get_at_path(actual, leaf.path) if actual is not None else MISSING
        rows.append(_score_leaf(leaf, exp, act, profile))
    return rows


def _score_leaf(
    leaf: SchemaLeaf,
    expected: Any,
    actual: Any,
    profile: ScoringProfile,
) -> FieldScore:
    if expected is MISSING and actual is MISSING:
        return _row(leaf, None, None, 1.0, "null_match")

    # Null handling: if expected is None and actual is None → null_match.
    if expected is None and (actual is None or actual is MISSING):
        return _row(leaf, expected, actual if actual is not MISSING else None, 1.0, "null_match")

    if actual is MISSING:
        return _row(leaf, expected, None, 0.0, "miss")

    if expected is MISSING:
        return _row(leaf, None, actual, 0.0, "extra")

    if expected is None and actual is not None:
        return _row(leaf, None, actual, 0.0, "extra")
    if expected is not None and actual is None:
        return _row(leaf, expected, None, 0.0, "miss")

    leaf_type = leaf.leaf_type
    schema_node = leaf.schema

    if leaf_type == "string" or leaf_type == "enum":
        return _score_string(leaf, expected, actual, profile, enum=leaf_type == "enum")
    if leaf_type == "date":
        return _score_date(leaf, expected, actual)
    if leaf_type == "integer":
        return _score_integer(leaf, expected, actual)
    if leaf_type == "number":
        tol = float(schema_node.get("x-tolerance", profile.number_rel_tolerance))
        return _score_number(leaf, expected, actual, tol)
    if leaf_type == "boolean":
        return _score_boolean(leaf, expected, actual)
    if leaf_type == "array_primitive":
        return _score_array_primitive(leaf, expected, actual, profile)
    if leaf_type == "array_object":
        return _score_array_object(leaf, expected, actual, profile)

    return _row(leaf, expected, actual, 0.0, "mismatch")


def _row(
    leaf: SchemaLeaf,
    expected: Any,
    actual: Any,
    s: float,
    match: MatchKind,
) -> FieldScore:
    return FieldScore(
        path=leaf.path,
        leaf_type=leaf.leaf_type,
        expected=expected,
        actual=actual,
        score=float(s),
        match=match,
        capabilities=list(leaf.capabilities),
        difficulty=leaf.difficulty,
        required=leaf.required,
    )


def _score_string(
    leaf: SchemaLeaf,
    expected: Any,
    actual: Any,
    profile: ScoringProfile,
    *,
    enum: bool,
) -> FieldScore:
    exp_n = normalize_string(expected, ignore_punctuation=profile.ignore_punctuation)
    act_n = normalize_string(actual, ignore_punctuation=profile.ignore_punctuation)
    if exp_n is None and act_n is None:
        return _row(leaf, expected, actual, 1.0, "null_match")
    if exp_n is None or act_n is None:
        return _row(leaf, expected, actual, 0.0, "mismatch")
    if exp_n == act_n:
        return _row(leaf, expected, actual, 1.0, "exact")
    if enum:
        return _row(leaf, expected, actual, 0.0, "mismatch")

    ratio = fuzz.token_set_ratio(exp_n, act_n)
    if ratio >= profile.fuzzy_threshold:
        return _row(leaf, expected, actual, 1.0, "fuzzy")
    if ratio < profile.fuzzy_floor:
        return _row(leaf, expected, actual, 0.0, "mismatch")
    span = max(profile.fuzzy_threshold - profile.fuzzy_floor, 1)
    s = profile.fuzzy_floor_score + (1.0 - profile.fuzzy_floor_score) * (
        (ratio - profile.fuzzy_floor) / span
    )
    return _row(leaf, expected, actual, s, "fuzzy")


def _score_date(leaf: SchemaLeaf, expected: Any, actual: Any) -> FieldScore:
    exp_n = normalize_date(expected)
    act_n = normalize_date(actual)
    if exp_n is None and act_n is None:
        return _row(leaf, expected, actual, 1.0, "null_match")
    if exp_n is None or act_n is None:
        return _row(leaf, expected, actual, 0.0, "mismatch")
    return _row(leaf, expected, actual, 1.0 if exp_n == act_n else 0.0,
                "exact" if exp_n == act_n else "mismatch")


def _score_integer(leaf: SchemaLeaf, expected: Any, actual: Any) -> FieldScore:
    exp_n = normalize_integer(expected)
    act_n = normalize_integer(actual)
    if exp_n is None or act_n is None:
        return _row(leaf, expected, actual, 0.0, "mismatch")
    return _row(leaf, expected, actual, 1.0 if exp_n == act_n else 0.0,
                "exact" if exp_n == act_n else "mismatch")


def _score_number(leaf: SchemaLeaf, expected: Any, actual: Any, rel_tol: float) -> FieldScore:
    exp_n = normalize_number(expected)
    act_n = normalize_number(actual)
    if exp_n is None or act_n is None:
        return _row(leaf, expected, actual, 0.0, "mismatch")
    if exp_n == 0.0 and act_n == 0.0:
        return _row(leaf, expected, actual, 1.0, "exact")
    denom = max(abs(exp_n), 1e-12)
    if abs(exp_n - act_n) / denom <= rel_tol:
        return _row(leaf, expected, actual, 1.0,
                    "exact" if exp_n == act_n else "tolerance")
    return _row(leaf, expected, actual, 0.0, "mismatch")


def _score_boolean(leaf: SchemaLeaf, expected: Any, actual: Any) -> FieldScore:
    exp_n = normalize_boolean(expected)
    act_n = normalize_boolean(actual)
    if exp_n is None or act_n is None:
        return _row(leaf, expected, actual, 0.0, "mismatch")
    return _row(leaf, expected, actual, 1.0 if exp_n == act_n else 0.0,
                "exact" if exp_n == act_n else "mismatch")


def _score_array_primitive(
    leaf: SchemaLeaf,
    expected: Any,
    actual: Any,
    profile: ScoringProfile,
) -> FieldScore:
    if not isinstance(expected, list) or not isinstance(actual, list):
        return _row(leaf, expected, actual, 0.0, "mismatch")
    if not expected and not actual:
        return _row(leaf, expected, actual, 1.0, "exact")
    items_schema = leaf.schema.get("items") or {}
    inner_leaf_type = items_schema.get("type", "string")
    if isinstance(inner_leaf_type, list):
        inner_leaf_type = next((t for t in inner_leaf_type if t != "null"), "string")

    def _norm(v: Any) -> Any:
        if inner_leaf_type == "string":
            return normalize_string(v, ignore_punctuation=profile.ignore_punctuation)
        if inner_leaf_type == "integer":
            return normalize_integer(v)
        if inner_leaf_type == "number":
            return normalize_number(v)
        if inner_leaf_type == "boolean":
            return normalize_boolean(v)
        return v

    exp_set = [(_norm(v), v) for v in expected]
    act_set = [(_norm(v), v) for v in actual]
    matched_idx: set[int] = set()
    matches = 0
    for ne, _ in exp_set:
        for j, (na, _) in enumerate(act_set):
            if j in matched_idx:
                continue
            if ne is None or na is None:
                continue
            if ne == na:
                matched_idx.add(j); matches += 1; break
            if isinstance(ne, str) and isinstance(na, str):
                if fuzz.token_set_ratio(ne, na) >= profile.fuzzy_threshold:
                    matched_idx.add(j); matches += 1; break
    tp = matches
    fp = max(len(actual) - tp, 0)
    fn = max(len(expected) - tp, 0)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return _row(leaf, expected, actual, f1, "exact" if f1 == 1.0 else ("fuzzy" if f1 > 0 else "mismatch"))


def _score_array_object(
    leaf: SchemaLeaf,
    expected: Any,
    actual: Any,
    profile: ScoringProfile,
) -> FieldScore:
    if not isinstance(expected, list) or not isinstance(actual, list):
        return _row(leaf, expected, actual, 0.0, "mismatch")
    if not expected and not actual:
        return _row(leaf, expected, actual, 1.0, "exact")
    items_schema = leaf.schema.get("items") or {}

    def _pair_score(e: Any, a: Any) -> float:
        if not isinstance(e, dict) or not isinstance(a, dict):
            return 0.0
        scores = score(e, a, items_schema, profile=profile)
        if not scores:
            return 0.0
        return sum(s.score for s in scores) / len(scores)

    matches, unmatched_e, unmatched_a = hungarian_match(expected, actual, score_pair=_pair_score)
    tp_weighted = sum(s for _, _, s in matches)
    fp = len(unmatched_a)
    fn = len(unmatched_e)
    p = tp_weighted / (tp_weighted + fp) if (tp_weighted + fp) else 0.0
    r = tp_weighted / (tp_weighted + fn) if (tp_weighted + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return _row(leaf, expected, actual, f1, "exact" if f1 == 1.0 else ("fuzzy" if f1 > 0 else "mismatch"))


# ── Aggregations ────────────────────────────────────────────────────────────


def macro_f1(scores: Iterable[FieldScore]) -> float:
    s = list(scores)
    if not s:
        return 0.0
    return sum(x.score for x in s) / len(s)


def slice_by(scores: Iterable[FieldScore], key: str) -> dict[str, list[FieldScore]]:
    out: dict[str, list[FieldScore]] = {}
    for s in scores:
        if key == "difficulty":
            k = s.difficulty or "unknown"
            out.setdefault(k, []).append(s)
        elif key == "leaf_type":
            out.setdefault(s.leaf_type, []).append(s)
        elif key == "capability":
            for cap in s.capabilities or ["none"]:
                out.setdefault(cap, []).append(s)
        else:
            raise ValueError(f"unknown slice key: {key}")
    return out
