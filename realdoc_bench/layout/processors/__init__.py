"""Auto-discovers and registers all layout processors via side-effect imports."""

# Importing each module triggers @register_layout_processor decorators.
from realdoc_bench.layout.processors import gt_self  # noqa: F401

# Optional providers — best-effort imports so the package stays usable when
# a given SDK isn't installed.
try:
    from realdoc_bench.layout.processors import extend  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.layout.processors import dots_ocr  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.layout.processors import paddle_ocr_vl  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.layout.processors import paddle_structure  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.layout.processors import aws_textract  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.layout.processors import azure_di  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.layout.processors import reducto  # noqa: F401
except ImportError:
    pass
