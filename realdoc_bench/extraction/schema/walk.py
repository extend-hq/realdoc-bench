"""Walk a JSON-Schema, emitting one entry per leaf-typed field.

A "leaf" is anywhere a comparator decides ground truth equals actual:
- scalar leaves (string, number, integer, boolean, enum, date)
- arrays of primitives (treated as a single leaf — the scoring engine handles
  set-style comparison inside)
- arrays of objects (also a single leaf — bipartite matching inside)

Nested objects are recursed into and don't appear in the output.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SchemaLeaf:
    path: str                       # dotted, e.g. "vendor.address.city"
    leaf_type: str                  # "string" | "number" | "integer" | "boolean" | "enum"
                                    # | "date" | "array_primitive" | "array_object"
    schema: dict[str, Any]          # the leaf schema node
    capabilities: list[str]
    difficulty: str | None
    required: bool


def _leaf_type_of(schema: dict[str, Any]) -> str:
    if "enum" in schema:
        return "enum"
    t = schema.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        t = non_null[0] if non_null else None
    if t == "string" and schema.get("format") == "date":
        return "date"
    if t == "array":
        items = schema.get("items") or {}
        items_t = items.get("type")
        if isinstance(items_t, list):
            non_null = [x for x in items_t if x != "null"]
            items_t = non_null[0] if non_null else None
        if items_t == "object":
            return "array_object"
        return "array_primitive"
    return t or "string"


def walk_leaves(schema: dict[str, Any]) -> Iterator[SchemaLeaf]:
    """Yield one :class:`SchemaLeaf` per scoring-eligible field."""
    yield from _walk(schema, prefix="", required=False, parent_required=set())


def _walk(
    node: dict[str, Any],
    *,
    prefix: str,
    required: bool,
    parent_required: set[str],
) -> Iterator[SchemaLeaf]:
    if not isinstance(node, dict):
        return

    t = node.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        t = non_null[0] if non_null else None

    if t == "object" and isinstance(node.get("properties"), dict):
        req = set(node.get("required") or [])
        for field_name, child in node["properties"].items():
            child_path = f"{prefix}.{field_name}" if prefix else field_name
            yield from _walk(
                child,
                prefix=child_path,
                required=field_name in req,
                parent_required=req,
            )
        return

    leaf_type = _leaf_type_of(node)
    yield SchemaLeaf(
        path=prefix,
        leaf_type=leaf_type,
        schema=node,
        capabilities=list(node.get("x-capabilities") or []),
        difficulty=node.get("x-difficulty"),
        required=required,
    )


def get_at_path(value: Any, path: str) -> Any:
    """Return the value at ``path`` (dotted) in nested dicts, or :data:`_MISSING`."""
    if path == "":
        return value
    cur: Any = value
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return MISSING
    return cur


class _Missing:
    def __repr__(self) -> str:
        return "<MISSING>"


MISSING: Any = _Missing()
