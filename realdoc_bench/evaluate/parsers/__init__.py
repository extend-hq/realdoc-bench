"""Auto-registers all parse providers via side-effect imports."""

try:
    from realdoc_bench.evaluate.parsers import extend  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.evaluate.parsers import cloud_vlm  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.evaluate.parsers import reducto  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.evaluate.parsers import llamaparse  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.evaluate.parsers import azure_di  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.evaluate.parsers import aws_textract  # noqa: F401
except ImportError:
    pass
