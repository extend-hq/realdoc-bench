#!/usr/bin/env python3
"""Filter v1.1_pagewise/ into v1.1_pagewise_rich/ keeping only rich pages.

Step 1: HARDLINK rich pages (per Extend chunks classification) into
        v1.1_pagewise_rich/<primary_category>/<type>/<doc>__p<NN>.pdf

Step 2: For pages Extend classified as 'prose' or 'blank' (or pages from docs
        with no Extend run), run Sonnet vision. Promote any that Sonnet
        re-classifies as rich.

Output:
  datasets/extraction/v1.1_pagewise_rich/<category>/<type>/<page>.pdf
  datasets/extraction/v1.1_pagewise_rich/manifest.csv
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
V1_1 = REPO / "datasets" / "extraction" / "v1.1"
PW_SRC = REPO / "datasets" / "extraction" / "v1.1_pagewise"
PW_DST = REPO / "datasets" / "extraction" / "v1.1_pagewise_rich"

RICH_CATEGORIES = {"tables", "form", "mixed"}


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


VISION_PROMPT = """Look at this document page image. Return ONLY a JSON object:
{
  "primary_category": "tables" | "checkboxes" | "form" | "key_values" | "prose" | "mixed" | "blank",
  "n_tables": <int>,
  "n_checkboxes": <int — visible checkbox glyphs>,
  "n_form_fields": <int — fillable boxes/lines>,
  "n_kv_pairs": <int — labeled values>,
  "richness_score": <int 1-10>,
  "page_summary": <≤ 15 words>
}"""


def render_b64(pdf: Path, dpi: int = 110) -> str | None:
    try:
        with fitz.open(pdf) as doc:
            page = doc[0]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            return base64.b64encode(pix.tobytes("png")).decode()
    except Exception:
        return None


def vision_classify(pdf: Path, model: str) -> dict:
    b64 = render_b64(pdf)
    if b64 is None:
        return {"ok": False, "error": "render fail"}
    try:
        client = get_anthropic()
        resp = client.messages.create(
            model=model, max_tokens=400,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": VISION_PROMPT},
            ]}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
        return {"ok": True, "ann": json.loads(text),
                "tokens_in": resp.usage.input_tokens,
                "tokens_out": resp.usage.output_tokens}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def safe_copy(src: Path, dst: Path) -> str:
    """Always copy (not hardlink) so the destination is an independent file."""
    import shutil
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return "skip"
        dst.unlink()
    shutil.copy2(src, dst)
    return "copy"


def build_pagewise_pdf_path(domain: str, type_name: str, source_filename: str,
                            page_idx: int, page_count_total: int) -> Path:
    """Match the naming convention from build_v1_1_pagewise.py."""
    stem = source_filename.removesuffix(".pdf")
    pad = max(2, len(str(page_count_total)))
    page_pdf = f"{stem}__p{page_idx+1:0{pad}d}.pdf"
    return PW_SRC / domain / type_name / page_pdf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extend-features", required=True,
                    help="Path to per_page_features.csv from fetch_extend_chunks.py")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--vision-model", default="claude-sonnet-4-5")
    ap.add_argument("--skip-vision", action="store_true",
                    help="Skip Sonnet pass — only use Extend classifications")
    args = ap.parse_args()

    PW_DST.mkdir(parents=True, exist_ok=True)

    # Load Extend per-page features
    extend_rows = list(csv.DictReader(open(args.extend_features)))
    print(f"Extend per-page features: {len(extend_rows)} rows")

    # Build doc-page → (category, source) map
    # Normalize: extend filenames may or may not have .pdf depending on how
    # fetch_extend_chunks recorded them. Strip and re-add for consistency.
    extend_map: dict = {}
    for r in extend_rows:
        fname = r["filename"]
        if not fname.endswith(".pdf"):
            fname = fname + ".pdf"
        key = (r["domain"], r["type"], fname, int(r["page_idx"]))
        extend_map[key] = {
            "primary_category": r["primary_category"],
            "richness_score": int(r["richness_score"]),
            "n_tables": int(r["n_tables"]),
            "n_checkboxes": int(r["n_checkboxes"]),
            "n_kv": int(r["n_kv"]),
            "n_form_fields": 0,  # not in extend output; use n_kv as proxy
            "source": "extend",
        }

    # Get all pagewise PDFs (manifest)
    pw_manifest = list(csv.DictReader(open(PW_SRC / "manifest.csv")))
    print(f"Pagewise PDFs: {len(pw_manifest)}")

    # ── Step 1: link rich pages from Extend classification ────────────────
    print("\nStep 1: linking rich pages classified by Extend…")
    final_rows: list[dict] = []
    needs_vision: list[dict] = []
    n_linked = 0
    cat_counts: dict = defaultdict(int)

    for pw in pw_manifest:
        domain, typ, src_fname = pw["domain"], pw["type"], pw["source_filename"]
        page_idx = int(pw["page_idx"])
        if page_idx < 0:
            continue
        pc_total = int(pw["page_count_total"])
        page_pdf_src = Path(REPO / pw["page_pdf"])
        if not page_pdf_src.exists():
            continue

        key = (domain, typ, src_fname, page_idx)
        ext = extend_map.get(key)
        if ext is None:
            # No Extend data — vision required
            needs_vision.append({
                "domain": domain, "type": typ, "filename": src_fname,
                "page_idx": page_idx, "page_count_total": pc_total,
                "page_pdf": page_pdf_src, "reason": "no_extend_data",
            })
            continue
        cat = ext["primary_category"]
        if cat in RICH_CATEGORIES:
            # Copy into flat directory (category recorded in manifest, not folders)
            dst = PW_DST / page_pdf_src.name
            safe_copy(page_pdf_src, dst)
            n_linked += 1
            cat_counts[cat] += 1
            final_rows.append({
                "category": cat, "type": typ,
                "domain": domain, "source_filename": src_fname,
                "page_idx": page_idx, "page_count_total": pc_total,
                "page_pdf": str(dst.relative_to(REPO)),
                "richness_score": ext["richness_score"],
                "n_tables": ext["n_tables"], "n_checkboxes": ext["n_checkboxes"],
                "n_kv": ext["n_kv"], "n_form_fields": ext["n_form_fields"],
                "source": "extend", "vision_promoted": False,
            })
        else:
            # prose / blank — candidate for vision verification
            needs_vision.append({
                "domain": domain, "type": typ, "filename": src_fname,
                "page_idx": page_idx, "page_count_total": pc_total,
                "page_pdf": page_pdf_src, "reason": f"extend_says_{cat}",
            })

    print(f"  ✓ linked {n_linked} pages from Extend")
    for c, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"      {c:12} {n:>4}")
    print(f"  → {len(needs_vision)} pages need Sonnet vision verification")

    # ── Step 2: vision on prose/blank pages ────────────────────────────────
    n_promoted = 0
    promote_cat: dict = defaultdict(int)
    vision_tokens_in = vision_tokens_out = 0

    if needs_vision and not args.skip_vision:
        print(f"\nStep 2: vision on {len(needs_vision)} pages with {args.vision_model} "
              f"({args.workers} workers)…")
        t0 = time.time()

        def _job(item):
            r = vision_classify(item["page_pdf"], args.vision_model)
            return (item, r)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_job, it) for it in needs_vision]
            for i, fut in enumerate(as_completed(futures), 1):
                item, r = fut.result()
                if not r["ok"]:
                    final_rows.append({
                        "category": "vision_fail", "type": item["type"],
                        "domain": item["domain"], "source_filename": item["filename"],
                        "page_idx": item["page_idx"], "page_count_total": item["page_count_total"],
                        "page_pdf": "", "richness_score": 0,
                        "n_tables": 0, "n_checkboxes": 0, "n_kv": 0, "n_form_fields": 0,
                        "source": "vision_fail", "vision_promoted": False,
                    })
                    continue
                a = r["ann"]
                vision_tokens_in += r.get("tokens_in", 0)
                vision_tokens_out += r.get("tokens_out", 0)
                cat = a.get("primary_category", "prose")
                if cat in RICH_CATEGORIES:
                    dst = PW_DST / item["page_pdf"].name
                    safe_copy(item["page_pdf"], dst)
                    n_promoted += 1
                    promote_cat[cat] += 1
                    final_rows.append({
                        "category": cat, "type": item["type"],
                        "domain": item["domain"], "source_filename": item["filename"],
                        "page_idx": item["page_idx"], "page_count_total": item["page_count_total"],
                        "page_pdf": str(dst.relative_to(REPO)),
                        "richness_score": int(a.get("richness_score", 0)),
                        "n_tables": int(a.get("n_tables", 0)),
                        "n_checkboxes": int(a.get("n_checkboxes", 0)),
                        "n_kv": int(a.get("n_kv_pairs", 0)),
                        "n_form_fields": int(a.get("n_form_fields", 0)),
                        "source": "vision", "vision_promoted": True,
                    })
                if i % 100 == 0 or i == len(needs_vision):
                    print(f"  [{i}/{len(needs_vision)}] visioned, {n_promoted} promoted", flush=True)

        wall = time.time() - t0
        cost = (vision_tokens_in / 1e6) * 3.0 + (vision_tokens_out / 1e6) * 15.0
        print(f"\n  ✓ visioned {len(needs_vision)} pages in {wall:.0f}s, est. ${cost:.2f}")
        print(f"    promoted {n_promoted} into rich set:")
        for c, n in sorted(promote_cat.items(), key=lambda x: -x[1]):
            print(f"      {c:12} +{n}")

    # ── Manifest + summary ─────────────────────────────────────────────────
    manifest = PW_DST / "manifest.csv"
    fields = ["category", "type", "domain", "source_filename", "page_idx",
              "page_count_total", "page_pdf", "richness_score",
              "n_tables", "n_checkboxes", "n_kv", "n_form_fields",
              "source", "vision_promoted"]
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(final_rows, key=lambda x: (x["category"], x["type"], x["source_filename"], x["page_idx"])):
            w.writerow(r)

    final_cat_counts: dict = defaultdict(int)
    for r in final_rows:
        final_cat_counts[r["category"]] += 1

    summary = PW_DST / "summary.md"
    rich_total = sum(n for c, n in final_cat_counts.items() if c in RICH_CATEGORIES)
    with summary.open("w") as f:
        f.write(f"# v1.1_pagewise_rich — filter summary\n\n")
        f.write(f"**{rich_total} rich pages** filtered from {len(pw_manifest)} pagewise PDFs.\n\n")
        f.write("## Source breakdown\n\n")
        f.write(f"- Extend classifications: {n_linked} pages\n")
        f.write(f"- Vision promotions (caught what Extend missed): {n_promoted} pages\n\n")
        f.write("## Per-category counts (final)\n\n")
        f.write("| Category | Pages |\n|---|---:|\n")
        for c, n in sorted(final_cat_counts.items(), key=lambda x: -x[1]):
            f.write(f"| `{c}` | {n} |\n")

    print(f"\n=== FINAL ===")
    print(f"  rich pages in v1.1_pagewise_rich/: {rich_total} (from Extend: {n_linked}, vision promoted: {n_promoted})")
    for c, n in sorted(final_cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {c:12} {n:>5}")
    print(f"\nWrote: {manifest.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
