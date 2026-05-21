"""Dataset subset selection.

Every pipeline stage works off ``sample.csv`` in the run directory; this
filters those rows down to a domain / document-type subset so you can run
the benchmark on, say, finance-only or medical-only.
"""

from __future__ import annotations


def filter_rows(
    rows: list[dict],
    domain: list[str] | None = None,
    doc_type: list[str] | None = None,
) -> list[dict]:
    """Keep only the ``sample.csv`` rows matching the filters.

    ``domain`` / ``doc_type`` are case-insensitive lists; ``None`` (or empty)
    means no filter on that axis.
    """
    out = rows
    if domain:
        keep = {d.lower() for d in domain}
        out = [r for r in out if (r.get("domain") or "").lower() in keep]
    if doc_type:
        keep = {t.lower() for t in doc_type}
        out = [r for r in out if (r.get("type") or "").lower() in keep]
    return out
