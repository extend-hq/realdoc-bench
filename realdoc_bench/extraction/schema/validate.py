"""Thin wrapper around ``jsonschema`` for agent output validation.

Returns a list of human-readable error messages — empty when the value validates.
"""

from __future__ import annotations

from typing import Any


def validate(value: Any, schema: dict[str, Any]) -> list[str]:
    from jsonschema import Draft7Validator

    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path))
    return [
        f"{'.'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in errors
    ]
