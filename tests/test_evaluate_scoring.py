"""Scoring tests pinned to bank edge cases.

The bank has Q&A items where typed-equality and fuzzy-equality both matter;
these are the cases that have bitten us before. If `deep_equal` /
`fuzzy_equal` change, this test catches regressions before they hit the eval
cache and silently invalidate every score.
"""

from __future__ import annotations

from realdoc_bench.evaluate import score


def test_deep_equal_currency_to_number():
    assert score.deep_equal(1234, "$1,234")
    assert score.deep_equal(1234.0, "$1,234.00")
    assert score.deep_equal(5.875, "5.875%")
    assert not score.deep_equal(1234, "$1,235")


def test_deep_equal_dates_only_normalize_when_template_says_so():
    # deep_equal itself is string-vs-string here; ISO normalization happens at
    # jsonify time, NOT inside deep_equal. So bare dates only match if the
    # strings are already equivalent under _norm_str collapsing.
    assert score.deep_equal("2024-01-15", "2024-01-15")
    assert not score.deep_equal("01/15/2024", "2024-01-15")


def test_deep_equal_ocr_spacing():
    assert score.deep_equal("0.00", "0 . 0 0")
    assert score.deep_equal("CAT", "C A T")


def test_deep_equal_smart_quotes_and_dashes():
    assert score.deep_equal("can't", "can’t")
    assert score.deep_equal("12-34", "12–34")


def test_deep_equal_parser_markdown_styling():
    assert score.deep_equal("hello world", "**hello** world")
    assert score.deep_equal("a b", "a<br/>b")


def test_deep_equal_bool_strict():
    assert score.deep_equal(True, True)
    assert not score.deep_equal(True, 1)        # bool vs int never equal
    assert not score.deep_equal(False, 0)


def test_deep_equal_1elem_list_dict():
    assert score.deep_equal([{"k": 1}], {"k": 1})
    assert score.deep_equal({"k": 1}, [{"k": 1}])


def test_fuzzy_equal_short_strings_stay_exact():
    # short = "ID-like" so a single-digit divergence MUST stay a miss
    assert not score.fuzzy_equal("4001884081", "4001844081")
    assert score.fuzzy_equal("4001884081", "4001884081")


def test_fuzzy_equal_multiword_phrases_match_close_variants():
    a = "the quick brown fox jumps over the lazy dog"
    b = "the quick brown fox jumped over the lazy dogs"
    assert score.fuzzy_equal(a, b)


def test_fuzzy_equal_falls_through_to_deep_equal_on_non_strings():
    assert score.fuzzy_equal(1234, "$1,234")


def test_string_keys_only_picks_plain_strings():
    tmpl = ('{\n  "name": <string>,\n  "amount": <number>,\n'
            '  "active": <boolean>,\n  "category": <one of: A | B>\n}')
    assert score.string_keys(tmpl) == {"name"}


def test_score_typed_mixes_layers():
    gold = {"name": "Acme Corp", "amount": 100}
    answer = {"name": "Acme Corp.", "amount": 100}  # name uses fuzzy
    str_keys = {"name"}
    fm, all_ok = score.score_typed(answer, gold, str_keys)
    assert fm == {"name": True, "amount": True}
    assert all_ok


def test_build_template_paired_format():
    rf = "Return exactly: amount=<number>; date=<date>"
    tmpl = score.build_template(rf, {"amount": 100, "date": "2024-01-15"})
    assert '"amount"' in tmpl and "<number" in tmpl
    assert '"date"' in tmpl


def test_build_template_falls_back_to_gold_shape():
    rf = "describe the items"
    tmpl = score.build_template(rf, {"items": ["a", "b"]})
    assert "list of strings" in tmpl
