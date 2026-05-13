"""Auto-registers all parse providers via side-effect imports."""

from realdoc_bench.extraction.parsers import pymupdf  # noqa: F401

try:
    from realdoc_bench.extraction.parsers import extend  # noqa: F401
except ImportError:
    pass
