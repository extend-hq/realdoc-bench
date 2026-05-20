"""Locate where a schema field's expected value appears in the source PDF.

Given a (doc_id, field_path) from a run's score.json, this:

  1. Reads the expected value from the latest score.json
  2. Searches every parser's parse_result.json markdown for that value;
     reports which page(s) it lives on (Extend / pymupdf / dots / etc. all
     emit ``## Page N`` markers)
  3. Optionally opens the PDF at the first hit on macOS (Preview's
     `open --jump` doesn't take page args, but Skim or Adobe accept them
     via -a / specific URL schemes)

Usage:
    python scripts/find_answer.py fin_audit_001 highest_deficiency_revenue_band
    python scripts/find_answer.py fin_10k_001 sonesta_brand_property_count --open
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIRS = [
    "runs/bench_extract_v0",
    "runs/bench_extract_v0_v2fix",
    "runs/bench_extract_v2_ga",
    "runs/bench_extract_claude16k",
    "runs/bench_extract_textract",
    "runs/bench_extract_azure_di",
    "runs/bench_extract_llamaparse",
    "runs/bench_extract_reducto_v2",
    "runs/bench_extract_dots_mocr",
    "runs/bench_extract_claude_sonnet_4_6",
    "runs/bench_extract_gemini_3_flash",
]
PREFERRED_PARSERS = [
    "extend_performance_v2_0_0_beta",
    "extend_performance_v2_0_0",
    "extend_performance_v1_0_0",
    "pymupdf",
    "aws_textract",
    "azure_di",
]


def _expected_from_scores(doc_id: str, field: str) -> tuple[str, list[str]]:
    """Find the expected value + list every parser that scored this field."""
    expected = None
    parsers = []
    for rd in RUN_DIRS:
        for sf in (REPO_ROOT / rd).glob(f"{doc_id}/*/*/score.json"):
            parser = sf.parent.parent.name
            try:
                d = json.load(open(sf))
            except Exception:
                continue
            for s in d.get("scores", []):
                if s["path"] == field:
                    expected = s["expected"]
                    if parser not in parsers:
                        parsers.append(parser)
                    break
    return expected, parsers


def _find_in_markdown(needle: str, md: str) -> list[tuple[int, str]]:
    """Return (page_no, context_snippet) for each occurrence of ``needle``.

    Pages are inferred from ``## Page N`` markers (Extend / dots / pymupdf
    use this); for parsers without page markers we fall back to absolute
    char-offset → estimated page (assuming ~5000 chars/page).
    """
    pages = []
    page_re = re.compile(r"^##\s+Page\s+(\d+)", re.MULTILINE)
    page_marks = [(m.start(), int(m.group(1))) for m in page_re.finditer(md)]
    hits = [m.start() for m in re.finditer(re.escape(str(needle)), md)]
    for h in hits:
        # Find the most recent page marker before this hit
        page = 0
        for start, n in page_marks:
            if start <= h:
                page = n
            else:
                break
        if page == 0 and page_marks:
            page = page_marks[0][1] - 1
        elif page == 0:
            # No page markers — estimate
            page = max(1, h // 5000 + 1)
        # context
        ctx = md[max(0, h - 80) : h + len(str(needle)) + 80].replace("\n", " ")
        pages.append((page, ctx))
    return pages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doc_id")
    ap.add_argument("field")
    ap.add_argument("--open", action="store_true", help="Open the PDF on macOS after printing.")
    ap.add_argument("--needle", help="Override the expected value to search for.")
    args = ap.parse_args()

    expected, _ = _expected_from_scores(args.doc_id, args.field)
    needle = args.needle or (str(expected) if expected is not None else None)
    if not needle or needle in {"None", "[]", "{}"}:
        print(f"No useful needle for {args.doc_id}/{args.field}  (expected={expected!r})")
        print("Pass --needle '<text>' to override.")
        return 1

    print(f"doc:    {args.doc_id}")
    print(f"field:  {args.field}")
    print(f"needle: {needle!r}")
    print()

    # Walk each preferred parser's parse_result.json
    found_any = False
    for parser in PREFERRED_PARSERS:
        for rd in RUN_DIRS:
            pr = REPO_ROOT / rd / args.doc_id / parser / "parse_result.json"
            if not pr.exists():
                continue
            try:
                d = json.load(open(pr))
            except Exception:
                continue
            md = d.get("markdown") or ""
            hits = _find_in_markdown(needle, md)
            if hits:
                pages = sorted({p for p, _ in hits})
                print(f"  ✓ {parser}: {len(hits)} hit(s) on page(s) {pages}")
                ctx = hits[0][1]
                print(f"      first-hit context: …{ctx}…")
                found_any = True
                break  # first matching run-id is enough
    if not found_any:
        print("  (needle not found in any parser's markdown — likely a derived/computed field)")

    pdf = REPO_ROOT / "datasets" / "extraction" / "v0" / "sources" / f"{args.doc_id}.pdf"
    print()
    print(f"  📄 PDF: {pdf}")
    if args.open and pdf.exists():
        # On macOS, `open` doesn't accept page numbers — Preview opens at the start.
        # Skim and Adobe accept page args via URL scheme; fallback to plain open.
        subprocess.run(["open", str(pdf)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
