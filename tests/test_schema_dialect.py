from __future__ import annotations

from realdoc_bench.extraction.schema.dialect import to_plain_jsonschema
from realdoc_bench.extraction.schema.walk import walk_leaves


def test_strips_x_keys_recursively():
    schema = {
        "type": "object",
        "x-domain": "financial_services",
        "properties": {
            "amount": {"type": "number", "x-difficulty": "hard", "x-capabilities": ["table"]},
            "child": {
                "type": "object",
                "x-doc-type": "thing",
                "properties": {"city": {"type": "string", "x-difficulty": "easy"}},
            },
        },
    }
    out = to_plain_jsonschema(schema)
    assert "x-domain" not in out
    assert "x-difficulty" not in out["properties"]["amount"]
    assert "x-capabilities" not in out["properties"]["amount"]
    assert "x-doc-type" not in out["properties"]["child"]
    assert "x-difficulty" not in out["properties"]["child"]["properties"]["city"]


def test_currency_rewrites_to_number():
    schema = {"type": "currency"}
    out = to_plain_jsonschema(schema)
    assert out["type"] == "number"


def test_date_rewrites_to_string_with_format():
    schema = {"type": "date"}
    out = to_plain_jsonschema(schema)
    assert out["type"] == "string"
    assert out["format"] == "date"


def test_signature_rewrites_to_string():
    schema = {"type": "signature"}
    out = to_plain_jsonschema(schema)
    assert out["type"] == "string"


def test_nullable_union_preserved():
    schema = {
        "type": "object",
        "properties": {
            "filer_category": {"type": ["string", "null"], "enum": ["a", "b", None]},
        },
    }
    out = to_plain_jsonschema(schema)
    assert out["properties"]["filer_category"]["type"] == ["string", "null"]


def test_walk_emits_one_per_leaf_and_recurses():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "nested": {
                "type": "object",
                "properties": {
                    "b": {"type": "integer", "x-difficulty": "easy"},
                    "c": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }
    leaves = list(walk_leaves(schema))
    paths = {(l.path, l.leaf_type) for l in leaves}
    assert ("a", "string") in paths
    assert ("nested.b", "integer") in paths
    assert ("nested.c", "array_primitive") in paths


def test_walk_difficulty_propagates():
    schema = {"type": "object", "properties": {"x": {"type": "string", "x-difficulty": "hard"}}}
    leaves = list(walk_leaves(schema))
    assert leaves[0].difficulty == "hard"
