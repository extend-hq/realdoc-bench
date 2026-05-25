"""Content-hashed local cache helpers used by parsers + dataset loaders."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def default_cache_root() -> Path:
    env = os.environ.get("REALDOC_BENCH_CACHE")
    if env:
        return Path(env).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "realdoc-bench"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
