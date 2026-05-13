"""Auto-registers all extraction agents via side-effect imports."""

try:
    from realdoc_bench.extraction.agents import claude  # noqa: F401
except ImportError:
    pass

try:
    from realdoc_bench.extraction.agents import gemini  # noqa: F401
except ImportError:
    pass
