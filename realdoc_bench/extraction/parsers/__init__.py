"""Auto-registers all parse providers via side-effect imports."""

from realdoc_bench.extraction.parsers import pymupdf  # noqa: F401

try:
    from realdoc_bench.extraction.parsers import extend  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.extraction.parsers import dots_mocr  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.extraction.parsers import glm_ocr  # noqa: F401
except ImportError:
    pass
