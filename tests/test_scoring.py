from __future__ import annotations

from realdoc_bench.extraction.scoring.engine import macro_f1, score
from realdoc_bench.extraction.scoring.profiles import (
    DEFAULT_PROFILE,
    LENIENT_PROFILE,
    STRICT_PROFILE,
)


SCHEMA = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "fiscal_year_end": {"type": "string", "format": "date"},
        "is_wksi": {"type": ["boolean", "null"]},
        "share_count": {"type": "integer"},
        "total_equity": {"type": "number"},
        "filer_category": {
            "type": ["string", "null"],
            "enum": ["large", "small", None],
        },
        "ticker_symbols": {"type": "array", "items": {"type": "string"}},
        "subsidiaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "state": {"type": "string"},
                },
            },
        },
    },
    "required": [
        "company_name",
        "fiscal_year_end",
        "share_count",
        "total_equity",
        "ticker_symbols",
    ],
}

EXPECTED = {
    "company_name": "Service Properties Trust",
    "fiscal_year_end": "December 31, 2024",
    "is_wksi": True,
    "share_count": 166_500_000,
    "total_equity": 1_234_567_000.0,
    "filer_category": "large",
    "ticker_symbols": ["SVC"],
    "subsidiaries": [
        {"name": "Sonesta International Hotels", "state": "Massachusetts"},
        {"name": "TravelCenters of America", "state": "Ohio"},
    ],
}


def _by_path(rows):
    return {r.path: r for r in rows}


def test_exact_match_all_pass():
    actual = dict(EXPECTED)
    rows = _by_path(score(EXPECTED, actual, SCHEMA))
    assert all(r.score == 1.0 for r in rows.values())
    assert macro_f1(rows.values()) == 1.0


def test_string_fuzzy_partial_credit():
    actual = {**EXPECTED, "company_name": "Service Properties, Trust"}  # extra comma
    rows = _by_path(score(EXPECTED, actual, SCHEMA))
    assert rows["company_name"].score == 1.0
    assert rows["company_name"].match in {"exact", "fuzzy"}


def test_date_normalization():
    actual = {**EXPECTED, "fiscal_year_end": "2024-12-31"}
    rows = _by_path(score(EXPECTED, actual, SCHEMA))
    assert rows["fiscal_year_end"].score == 1.0
    assert rows["fiscal_year_end"].match == "exact"


def test_number_tolerance_default_50bps():
    actual = {**EXPECTED, "total_equity": 1_234_567_000.0 + 1_000_000}  # 0.08%
    rows = _by_path(score(EXPECTED, actual, SCHEMA))
    assert rows["total_equity"].score == 1.0
    assert rows["total_equity"].match == "tolerance"


def test_number_tolerance_breach():
    actual = {**EXPECTED, "total_equity": 1_400_000_000.0}  # >0.5% off
    rows = _by_path(score(EXPECTED, actual, SCHEMA))
    assert rows["total_equity"].score == 0.0


def test_number_tolerance_x_tolerance_override():
    schema = {
        "type": "object",
        "properties": {
            "total_equity": {"type": "number", "x-tolerance": 0.10},
        },
    }
    expected = {"total_equity": 1_000_000.0}
    actual = {"total_equity": 1_080_000.0}  # 8%
    rows = _by_path(score(expected, actual, schema))
    assert rows["total_equity"].score == 1.0
    assert rows["total_equity"].match == "tolerance"


def test_strict_profile_rejects_fuzzy():
    actual = {**EXPECTED, "company_name": "Service Properties, Trust"}
    rows_strict = _by_path(score(EXPECTED, actual, SCHEMA, profile=STRICT_PROFILE))
    assert rows_strict["company_name"].score < 1.0


def test_enum_strict():
    actual = {**EXPECTED, "filer_category": "non_accelerated_filer"}
    rows = _by_path(score(EXPECTED, actual, SCHEMA))
    assert rows["filer_category"].score == 0.0
    assert rows["filer_category"].match == "mismatch"


def test_array_primitive_extra_element():
    actual = {**EXPECTED, "ticker_symbols": ["SVC", "TA"]}
    rows = _by_path(score(EXPECTED, actual, SCHEMA))
    assert 0.0 < rows["ticker_symbols"].score < 1.0


def test_array_object_bipartite_with_misorder():
    actual = {
        **EXPECTED,
        "subsidiaries": [
            {"name": "TravelCenters of America", "state": "Ohio"},
            {"name": "Sonesta International Hotels", "state": "Massachusetts"},
        ],
    }
    rows = _by_path(score(EXPECTED, actual, SCHEMA))
    assert rows["subsidiaries"].score == 1.0


def test_missing_required_field_is_miss():
    actual = {k: v for k, v in EXPECTED.items() if k != "share_count"}
    rows = _by_path(score(EXPECTED, actual, SCHEMA))
    assert rows["share_count"].match == "miss"
    assert rows["share_count"].score == 0.0


def test_null_match_does_not_count_against_quality():
    expected = {"is_wksi": None, **{k: v for k, v in EXPECTED.items() if k != "is_wksi"}}
    actual = dict(expected)
    rows = _by_path(score(expected, actual, SCHEMA))
    assert rows["is_wksi"].score == 1.0
    assert rows["is_wksi"].match == "null_match"


def test_lenient_profile_loosens_tolerance():
    # ~3% off — well outside the 0.5% default, inside the 5% lenient tolerance.
    actual = {**EXPECTED, "total_equity": 1_270_000_000.0}
    default_rows = _by_path(score(EXPECTED, actual, SCHEMA, profile=DEFAULT_PROFILE))
    lenient_rows = _by_path(score(EXPECTED, actual, SCHEMA, profile=LENIENT_PROFILE))
    assert default_rows["total_equity"].score == 0.0
    assert lenient_rows["total_equity"].score == 1.0
