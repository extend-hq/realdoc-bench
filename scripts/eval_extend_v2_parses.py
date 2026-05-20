#!/usr/bin/env python3
"""Evaluate Extend v2 parse outputs on three rubrics.

Inputs:
  runs/extend_v2_parse/<ts>/<domain>/<type>/<filename>.md
  datasets/extraction/v1.1/<domain>/<type>/sources/<filename>.pdf

Rubrics (1-10):
  A. EXTRACTION QUALITY   — does the markdown accurately reflect the PDF's
                            text, tables, key-value pairs, headings?
  B. FORM FILL QUALITY    — is the source PDF a properly filled form
                            (vs partially-blank, blank with stray text,
                            or non-form content like a guide/narrative)?
  C. HALLUCINATION INVERSE — does the markdown invent content not in the PDF?
                            (10 = clean, 1 = heavily hallucinated)

VLM judge: Claude Sonnet 4.5. Image = page 1 (or sampled pages for long docs);
text = the parsed markdown (truncated to 8000 chars).

Output:
  runs/extend_v2_parse/<ts>/quality_grades.csv
  runs/extend_v2_parse/<ts>/quality_grades.md
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
V1_1 = REPO / "datasets" / "extraction" / "v1.1"


def load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k and not os.environ.get(k):
            os.environ[k] = v
load_dotenv()


_anthropic = None
def get_anthropic():
    global _anthropic
    if _anthropic is None:
        from anthropic import Anthropic
        _anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _anthropic


RUBRIC_PROMPT = """You are evaluating an Extend v2 parse OUTPUT (markdown) against the ORIGINAL PDF page image it was extracted from.

You will be given:
1. The first page (or sampled page) of the source PDF, as an image.
2. The parsed markdown output from Extend v2.

Score each rubric 1-10 (10 = perfect, 1 = total failure):

A. EXTRACTION QUALITY
   - 10 = the markdown captures essentially everything visible on the PDF page: text, all field values, tables (with structure), key-value pairs, headings. No important content missing.
   - 7-9 = captures most content; minor table/structure issues or a few missed field values.
   - 4-6 = captures the main text but loses significant structure: tables flattened, key-value pairs jumbled, several field values missing.
   - 1-3 = badly degraded — large swaths missing, garbled OCR, structure destroyed.

B. FORM FILL QUALITY (of the source PDF, NOT of Extend's output)
   - 10 = the source PDF is a properly-filled form with consistent, plausible field values throughout the visible page.
   - 7-9 = most fields filled; a few left blank but that's plausible for the form type.
   - 4-6 = partially filled — many empty fields, or this looks like a narrative document not a form (10-K, audit report, contract) which is acceptable: rate prose docs as 7-8 unless they look unfinished/draft.
   - 1-3 = essentially blank template with only headers visible, OR an instructional/guide document showing how to fill in the form (not an actual filled form).

C. HALLUCINATION INVERSE
   - 10 = markdown content all traces back to the PDF; nothing invented.
   - 7-9 = mostly faithful; maybe one minor inferred value or paraphrase.
   - 4-6 = some content invented or wrong (e.g. value transcribed incorrectly, fake table row added).
   - 1-3 = heavy hallucination — entire sections fabricated.

Output ONLY a single JSON object, no preamble or fences:
{
  "extraction_quality": <int 1-10>,
  "form_fill_quality": <int 1-10>,
  "hallucination_inverse": <int 1-10>,
  "overall_grade": <int 1-10>,
  "is_form": <bool — true if this is a fillable form, false if narrative prose document>,
  "reasoning": "<2-3 sentence justification citing specific elements>"
}

=== PARSED MARKDOWN (may be truncated) ===
{markdown_excerpt}
"""


def render_page_b64(pdf: Path, page_idx: int = 0, dpi: int = 110) -> str:
    with fitz.open(pdf) as doc:
        page = doc[page_idx if page_idx < doc.page_count else 0]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return base64.b64encode(pix.tobytes("png")).decode()


@dataclass
class QualityRow:
    domain: str
    type: str
    filename: str
    extraction_quality: int
    form_fill_quality: int
    hallucination_inverse: int
    overall_grade: int
    is_form: bool
    reasoning: str
    md_length: int
    parse_ok: bool
    error: str = ""


def grade_one(domain: str, type_name: str, pdf: Path, md_path: Path, model: str) -> QualityRow:
    if not md_path.exists():
        return QualityRow(domain, type_name, pdf.name, 0, 0, 0, 0, False,
                          "", 0, False, error="no markdown — parse failed or skipped")
    markdown = md_path.read_text()
    if not markdown.strip():
        return QualityRow(domain, type_name, pdf.name, 0, 0, 0, 0, False,
                          "", 0, True, error="empty markdown")
    md_excerpt = markdown[:8000]
    try:
        img_b64 = render_page_b64(pdf)
    except Exception as e:
        return QualityRow(domain, type_name, pdf.name, 0, 0, 0, 0, False,
                          "", len(markdown), True, error=f"render fail: {e}")
    try:
        client = get_anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=900,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": RUBRIC_PROMPT.replace("{markdown_excerpt}", md_excerpt)},
                ],
            }],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
        g = json.loads(text)
        return QualityRow(
            domain=domain, type=type_name, filename=pdf.name,
            extraction_quality=int(g.get("extraction_quality", 0)),
            form_fill_quality=int(g.get("form_fill_quality", 0)),
            hallucination_inverse=int(g.get("hallucination_inverse", 0)),
            overall_grade=int(g.get("overall_grade", 0)),
            is_form=bool(g.get("is_form", False)),
            reasoning=str(g.get("reasoning", "")).strip(),
            md_length=len(markdown), parse_ok=True,
        )
    except Exception as e:
        return QualityRow(domain, type_name, pdf.name, 0, 0, 0, 0, False,
                          "", len(markdown), True, error=f"{type(e).__name__}: {str(e)[:200]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="runs/extend_v2_parse/<timestamp>/ directory")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="claude-sonnet-4-5")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"!! run dir not found: {run_dir}")
        return 1

    # Walk run_dir/<domain>/<type>/*.md, match against v1.1 PDFs
    work: list[tuple[str, str, Path, Path]] = []
    for md in sorted(run_dir.rglob("*.md")):
        rel = md.relative_to(run_dir).parts
        if len(rel) < 3:
            continue
        domain, type_name, md_name = rel[0], rel[1], rel[2]
        pdf = V1_1 / domain / type_name / "sources" / md_name.replace(".md", ".pdf")
        if not pdf.exists():
            continue
        work.append((domain, type_name, pdf, md))
    if args.limit:
        work = work[:args.limit]
    print(f"Evaluating {len(work)} parse outputs with {args.model} ({args.workers} workers)")

    t0 = time.time()
    results: list[QualityRow] = []
    def _job(t):
        return grade_one(*t, model=args.model)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_job, t) for t in work]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            tag = (f"err {r.error}" if r.error else
                   f"overall={r.overall_grade} ext={r.extraction_quality} fill={r.form_fill_quality} hal={r.hallucination_inverse} {'form' if r.is_form else 'prose'}")
            print(f"  [{i}/{len(work)}] {r.domain}/{r.type}/{r.filename}  {tag}", flush=True)

    # Sort by overall desc within domain
    results.sort(key=lambda r: (r.domain, r.type, -r.overall_grade, r.filename))

    # CSV
    csv_path = run_dir / "quality_grades.csv"
    fields = ["domain", "type", "filename", "overall_grade",
              "extraction_quality", "form_fill_quality", "hallucination_inverse",
              "is_form", "md_length", "reasoning", "parse_ok", "error"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in fields})

    # Markdown report
    md_path = run_dir / "quality_grades.md"
    with md_path.open("w") as f:
        f.write(f"# Extend v2 parse quality eval — {run_dir.name}\n\n")
        f.write(f"Model: `{args.model}`  ·  Docs: {len(results)}  ·  Wall: {time.time()-t0:.0f}s\n\n")
        # Per-type averages
        f.write("## Per-type averages\n\n")
        f.write("| Domain | Type | n | overall | ext | fill | hallu↑ | form? |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        by_type: dict[tuple[str, str], list[QualityRow]] = {}
        for r in results:
            by_type.setdefault((r.domain, r.type), []).append(r)
        for (d, t), rows in sorted(by_type.items()):
            graded = [r for r in rows if not r.error]
            if not graded:
                f.write(f"| {d} | `{t}` | {len(rows)} | ERR | - | - | - | - |\n")
                continue
            avg = lambda k: sum(getattr(r, k) for r in graded) / len(graded)
            forms = sum(1 for r in graded if r.is_form)
            f.write(f"| {d.replace('_', ' ')} | `{t}` | {len(graded)} | "
                    f"{avg('overall_grade'):.1f} | {avg('extraction_quality'):.1f} | "
                    f"{avg('form_fill_quality'):.1f} | {avg('hallucination_inverse'):.1f} | "
                    f"{forms}/{len(graded)} |\n")

        # Per-domain summary
        f.write("\n## Per-domain summary\n\n")
        f.write("| Domain | n | overall | ext | fill | hallu↑ |\n|---|---:|---:|---:|---:|---:|\n")
        by_dom: dict[str, list[QualityRow]] = {}
        for r in results:
            by_dom.setdefault(r.domain, []).append(r)
        for d, rows in sorted(by_dom.items()):
            graded = [r for r in rows if not r.error]
            if not graded:
                continue
            avg = lambda k: sum(getattr(r, k) for r in graded) / len(graded)
            f.write(f"| {d.replace('_', ' ')} | {len(graded)} | "
                    f"{avg('overall_grade'):.1f} | {avg('extraction_quality'):.1f} | "
                    f"{avg('form_fill_quality'):.1f} | {avg('hallucination_inverse'):.1f} |\n")

        # Per-doc
        f.write("\n## Per-doc grades\n\n")
        f.write("| Overall | ext | fill | hallu↑ | form? | Doc | Reasoning |\n")
        f.write("|---:|---:|---:|---:|---|---|---|\n")
        for r in results:
            if r.error:
                f.write(f"| ERR | - | - | - | - | `{r.domain}/{r.type}/{r.filename}` | {r.error} |\n")
            else:
                reason = r.reasoning.replace("|", "\\|").replace("\n", " ")
                f.write(f"| {r.overall_grade} | {r.extraction_quality} | {r.form_fill_quality} | "
                        f"{r.hallucination_inverse} | {'✓' if r.is_form else '·'} | "
                        f"`{r.domain}/{r.type}/{r.filename}` | {reason} |\n")

    # Console summary
    graded = [r for r in results if not r.error]
    if graded:
        avg_overall = sum(r.overall_grade for r in graded) / len(graded)
        avg_ext = sum(r.extraction_quality for r in graded) / len(graded)
        avg_fill = sum(r.form_fill_quality for r in graded) / len(graded)
        avg_hal = sum(r.hallucination_inverse for r in graded) / len(graded)
        print(f"\nMean across {len(graded)} graded docs:")
        print(f"  overall: {avg_overall:.2f}/10")
        print(f"  extraction_quality: {avg_ext:.2f}/10")
        print(f"  form_fill_quality: {avg_fill:.2f}/10")
        print(f"  hallucination_inverse: {avg_hal:.2f}/10")
    print(f"\nWrote:\n  {csv_path.relative_to(REPO)}\n  {md_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
