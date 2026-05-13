"""Auto-discovers and registers all layout processors via side-effect imports."""

# Importing each module triggers @register_layout_processor decorators.
from realdoc_bench.layout.processors import gt_self  # noqa: F401

# Extend depends on the extend-ai SDK; import is best-effort so the package
# stays usable in environments without that dependency installed.
try:
    from realdoc_bench.layout.processors import extend  # noqa: F401
except ImportError:
    pass
