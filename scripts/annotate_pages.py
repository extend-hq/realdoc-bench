#!/usr/bin/env python3
"""Page-level annotation across v1.1 source PDFs (including synth fills).

For each page of each PDF in datasets/extraction/v1.1/<dom>/<typ>/sources/:
  - Render to PNG (120 DPI)
  - Send to Claude Sonnet with rubric prompt
  - Get back JSON: {has_tables, has_checkboxes, has_form_fields,
                    has_key_values, n_tables, n_checkboxes, n_form_fields_est,
                    primary_category, richness_score, page_summary}
  - Append to manifest CSV

Output:
  runs/page_annotations/<ts>/manifest.csv
  runs/page_annotations/<ts>/_failures.csv  (any per-page errors)

Idempotent — re-running skips already-annotated (doc, page_idx) tuples.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
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


PROMPT = """Look at this document page image. Classify what's on the page. Return ONLY a JSON object — no preamble, no fences.

{
  "has_tables": <bool — any structured grid with multiple rows/columns of data>,
  "has_checkboxes": <bool — any visible ☐ ☑ □ ■ or printed checkbox glyphs>,
  "has_form_fields": <bool — printed labeled boxes/lines meant to be filled in (e.g. "Name: ____")>,
  "has_key_values": <bool — labeled prose values like "Date: 12/01/2024" or "Status: Active">,
  "has_signatures": <bool — handwritten or printed signature lines>,
  "n_tables": <int — count of distinct tables, 0 if none>,
  "n_checkboxes_visible": <int — total checkbox glyphs you can see, 0-many>,
  "n_form_fields_est": <int — rough count of fillable boxes/lines, 0-many>,
  "primary_category": <one of: "tables" | "checkboxes" | "form" | "key_values" | "prose" | "mixed" | "blank">,
  "is_blank_template": <bool — true if this is a blank or template form (almost no data filled in)>,
  "richness_score": <int 1-10 — 10 = page densely packed with structured data fields/tables/checkboxes; 5 = mixed; 1 = pure prose or near-blank>,
  "page_summary": <string ≤ 20 words — what this page contains>
}

Rules:
- A 10-K financial statement page with one big numbers table → "tables", richness ≥ 8
- A filled form page with checkboxes + values → "form" or "mixed", richness ≥ 8
- A cover page with title + a few labels → "key_values" or "mixed", richness 4-6
- A page of pure paragraph prose → "prose", richness 1-3
- An empty/blank form page → "blank" or "form" with is_blank_template=true
- A table of contents → "key_values", richness 4-5
"""


def render_page_b64(pdf_path: Path, page_idx: int, dpi: int = 120) -> str | None:
    try:
        with fitz.open(pdf_path) as doc:
            if page_idx >= doc.page_count:
                return None
            page = doc[page_idx]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            return base64.b64encode(pix.tobytes("png")).decode()
    except Exception:
        return None


def annotate_page(pdf_path: Path, page_idx: int, model: str) -> dict:
    t0 = time.time()
    b64 = render_page_b64(pdf_path, page_idx)
    if b64 is None:
        return {"ok": False, "error": "render fail", "elapsed": 0}
    try:
        client = get_anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": PROMPT},
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
        ann = json.loads(text)
        return {"ok": True, "ann": ann,
                "tokens_in": resp.usage.input_tokens,
                "tokens_out": resp.usage.output_tokens,
                "elapsed": time.time() - t0}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}",
                "elapsed": time.time() - t0}


def discover_pages(v1_1: Path) -> list[tuple[str, str, str, Path, int, int]]:
    """Yield (domain, type, fname, pdf_path, page_idx, page_count_total) for every page."""
    out = []
    for domain_dir in sorted(d for d in v1_1.iterdir() if d.is_dir() and not d.name.startswith("_")):
        for type_dir in sorted(t for t in domain_dir.iterdir() if t.is_dir()):
            sources = type_dir / "sources"
            if not sources.exists():
                continue
            for pdf in sorted(sources.glob("*.pdf")):
                try:
                    with fitz.open(pdf) as doc:
                        pc = doc.page_count
                except Exception:
                    continue
                for i in range(pc):
                    out.append((domain_dir.name, type_dir.name, pdf.name, pdf, i, pc))
    return out


def load_existing(manifest_path: Path) -> set[tuple[str, str, str, int]]:
    if not manifest_path.exists():
        return set()
    done: set = set()
    with manifest_path.open() as f:
        for row in csv.DictReader(f):
            try:
                done.add((row["domain"], row["type"], row["filename"], int(row["page_idx"])))
            except (KeyError, ValueError):
                pass
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None,
                    help="Resume into this dir; else create runs/page_annotations/<ts>/")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--type", action="append", help="Restrict to type (repeatable)")
    args = ap.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO / "runs" / "page_annotations" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    fail_path = out_dir / "_failures.csv"

    pages = discover_pages(V1_1)
    if args.type:
        pages = [p for p in pages if p[1] in args.type]
    print(f"Found {len(pages)} pages across {len({(p[0],p[1],p[2]) for p in pages})} docs")

    existing = load_existing(manifest_path)
    todo = [p for p in pages if (p[0], p[1], p[2], p[4]) not in existing]
    if args.limit:
        todo = todo[:args.limit]
    print(f"  done: {len(existing)}   todo: {len(todo)}   workers: {args.workers}")
    print(f"  out: {out_dir.relative_to(REPO)}")

    fields = ["domain", "type", "filename", "page_idx", "page_count_total",
              "primary_category", "richness_score", "is_blank_template",
              "has_tables", "n_tables",
              "has_checkboxes", "n_checkboxes_visible",
              "has_form_fields", "n_form_fields_est",
              "has_key_values", "has_signatures",
              "page_summary", "tokens_in", "tokens_out", "latency_sec"]

    # Open in append mode
    new_file = not manifest_path.exists()
    fmanifest = manifest_path.open("a", newline="")
    writer = csv.DictWriter(fmanifest, fieldnames=fields)
    if new_file:
        writer.writeheader()
        fmanifest.flush()

    ffail_new = not fail_path.exists()
    ffail = fail_path.open("a", newline="")
    fwriter = csv.DictWriter(ffail, fieldnames=["domain", "type", "filename", "page_idx", "error"])
    if ffail_new:
        fwriter.writeheader()
        ffail.flush()

    t0 = time.time()
    n_ok = n_fail = 0
    total_in = total_out = 0
    lock_lock = None  # ThreadPoolExecutor; we'll just rely on append serialization

    def _job(domain, typ, fname, pdf, page_idx, page_count_total):
        r = annotate_page(pdf, page_idx, args.model)
        return (domain, typ, fname, page_idx, page_count_total, r)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_job, *p) for p in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            domain, typ, fname, page_idx, pc_total, r = fut.result()
            if r["ok"]:
                n_ok += 1
                total_in += r["tokens_in"]
                total_out += r["tokens_out"]
                a = r["ann"]
                row = {
                    "domain": domain, "type": typ, "filename": fname,
                    "page_idx": page_idx, "page_count_total": pc_total,
                    "primary_category": a.get("primary_category", ""),
                    "richness_score": a.get("richness_score", 0),
                    "is_blank_template": a.get("is_blank_template", False),
                    "has_tables": a.get("has_tables", False),
                    "n_tables": a.get("n_tables", 0),
                    "has_checkboxes": a.get("has_checkboxes", False),
                    "n_checkboxes_visible": a.get("n_checkboxes_visible", 0),
                    "has_form_fields": a.get("has_form_fields", False),
                    "n_form_fields_est": a.get("n_form_fields_est", 0),
                    "has_key_values": a.get("has_key_values", False),
                    "has_signatures": a.get("has_signatures", False),
                    "page_summary": (a.get("page_summary", "") or "")[:300],
                    "tokens_in": r["tokens_in"],
                    "tokens_out": r["tokens_out"],
                    "latency_sec": round(r["elapsed"], 2),
                }
                writer.writerow(row); fmanifest.flush()
                if i % 50 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] ✓ {domain}/{typ}/{fname[:30]} p{page_idx+1}  "
                          f"[{row['primary_category']}] rich={row['richness_score']} "
                          f"tb={row['n_tables']} cb={row['n_checkboxes_visible']}", flush=True)
            else:
                n_fail += 1
                fwriter.writerow({"domain": domain, "type": typ, "filename": fname,
                                  "page_idx": page_idx, "error": r["error"]})
                ffail.flush()
                if n_fail <= 5 or n_fail % 20 == 0:
                    print(f"  [{i}/{len(todo)}] ✗ {domain}/{typ}/{fname[:30]} p{page_idx+1}  {r['error'][:80]}", flush=True)

    fmanifest.close(); ffail.close()
    wall = time.time() - t0
    cost = (total_in / 1e6) * 3.0 + (total_out / 1e6) * 15.0
    print(f"\nDone — {n_ok} ok / {n_fail} fail in {wall:.0f}s")
    print(f"  tokens: {total_in:,} in / {total_out:,} out, est. ${cost:.2f}")
    print(f"  manifest: {manifest_path.relative_to(REPO)}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
