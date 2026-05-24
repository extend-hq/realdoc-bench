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

try:
    from realdoc_bench.extraction.parsers import cloud_vlm  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.extraction.parsers import reducto  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.extraction.parsers import llamaparse  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.extraction.parsers import azure_di  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.extraction.parsers import aws_textract  # noqa: F401
except ImportError:
    pass
