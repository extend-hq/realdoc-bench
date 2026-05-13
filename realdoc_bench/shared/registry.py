"""Generic name-keyed registry used by parsers, agents, and layout processors."""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str, value: T) -> T:
        if name in self._entries:
            raise ValueError(f"{self._kind} {name!r} already registered")
        self._entries[name] = value
        return value

    def decorator(self, name: str) -> Callable[[T], T]:
        def _wrap(value: T) -> T:
            self.register(name, value)
            return value

        return _wrap

    def get(self, name: str) -> T:
        try:
            return self._entries[name]
        except KeyError as e:
            raise KeyError(
                f"unknown {self._kind} {name!r}. available: {sorted(self._entries)}"
            ) from e

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)
