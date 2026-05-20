#!/usr/bin/env python3
"""Page annotation using Extend mds as prefilter, vision only where needed.

Strategy:
  1. Read each doc's existing Extend md from runs/parse_sweep/<ts>/extend_*.
  2. Count features: markdown tables, [ ]/[x] checkboxes, key-value patterns,
     figure refs (![), form-field underscores.
  3. CLASSIFY DOCS:
     - "structure_rich": extend md shows ≥1 of (tables, checkboxes, kv, figures)
     - "structure_lite": low signal — needs vision verification
  4. For structure-lite docs, run Sonnet vision per page to catch what Extend
     missed (raster/scanned pages, checkboxes Extend couldn't render, etc.).
  5. Emit:
     - runs/page_annotations_v3/<ts>/per_doc.csv     (Extend-based features)
     - runs/page_annotations_v3/<ts>/per_page.csv    (visioned pages only)
     - runs/page_annotations_v3/<ts>/summary.md

Cost: Extend prefilter free; vision only on small-doc-no-extend-structure docs.
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


# ─────────────────────────────────────────────────────────────────────────────
# Feature regex (run on Extend mds, which preserve [ ]/[x] and pipe tables)
# ─────────────────────────────────────────────────────────────────────────────

TABLE_BLOCK_RE = re.compile(r"((?:^[ \t]*\|[^\n]*\n){2,})", re.MULTILINE)
MD_CHECKBOX_RE = re.compile(r"\[[ xX✓]\]")
UNICODE_CHECKBOX_RE = re.compile(r"[☐☑□■⬜⬛✓✗✔]")
FIGURE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
KV_RE = re.compile(r"(?:\*\*[^*]{2,60}\*\*\s*:|^[A-Z][A-Za-z0-9 /()&'.-]{2,50}:\s+[A-Za-z0-9$/])", re.MULTILINE)
FORM_FIELD_RE = re.compile(r"_{4,}|\.{6,}")
SIGNATURE_RE = re.compile(r"signature[^\n]*?(?:_{3,}|\Z|\n)", re.IGNORECASE)


def feature_counts(md: str) -> dict:
    if not md:
        return {"n_tables": 0, "n_checkboxes": 0, "n_kv": 0, "n_figures": 0,
                "n_form_fields": 0, "has_signature": False, "md_chars": 0}
    return {
        "n_tables": len(TABLE_BLOCK_RE.findall(md)),
        "n_checkboxes": len(MD_CHECKBOX_RE.findall(md)) + len(UNICODE_CHECKBOX_RE.findall(md)),
        "n_kv": len(KV_RE.findall(md)),
        "n_figures": len(FIGURE_RE.findall(md)),
        "n_form_fields": len(FORM_FIELD_RE.findall(md)),
        "has_signature": bool(SIGNATURE_RE.search(md)),
        "md_chars": len(md),
    }


def classify_doc(features: dict, page_count: int) -> str:
    """Return one of: 'rich', 'has_some', 'lite'."""
    total_signal = (features["n_tables"] + features["n_checkboxes"]
                    + features["n_kv"] // 3 + features["n_figures"])
    if total_signal >= 10 or features["n_checkboxes"] >= 5 or features["n_tables"] >= 5:
        return "rich"
    if total_signal >= 1:
        return "has_some"
    return "lite"


# ─────────────────────────────────────────────────────────────────────────────
# Vision (Sonnet) for pages in structure-lite docs
# ─────────────────────────────────────────────────────────────────────────────

VISION_PROMPT = """Look at this document page. Return ONLY a JSON object:
{
  "n_tables": <int>,
  "n_checkboxes": <int — visible checkbox glyphs>,
  "n_form_fields": <int — fillable boxes/lines>,
  "n_kv_pairs": <int — labeled values like "Name: John">,
  "n_figures": <int — non-table images/diagrams>,
  "has_signature": <bool>,
  "is_blank_template": <bool>,
  "primary_category": "tables" | "checkboxes" | "form" | "key_values" | "prose" | "mixed" | "blank",
  "richness_score": <int 1-10>,
  "page_summary": <≤ 15 words>
}"""


def render_b64(pdf_path: Path, page_idx: int, dpi: int = 110) -> str | None:
    try:
        with fitz.open(pdf_path) as doc:
            page = doc[page_idx]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            return base64.b64encode(pix.tobytes("png")).decode()
    except Exception:
        return None


def vision_classify(pdf_path: Path, page_idx: int, model: str) -> dict:
    b64 = render_b64(pdf_path, page_idx)
    if b64 is None:
        return {"ok": False, "error": "render fail"}
    try:
        client = get_anthropic()
        resp = client.messages.create(
            model=model, max_tokens=500,
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


# ─────────────────────────────────────────────────────────────────────────────
# Discovery + main
# ─────────────────────────────────────────────────────────────────────────────

def discover_docs(v1_1: Path) -> list[tuple[str, str, str, Path, int]]:
    out = []
    for domain_dir in sorted(d for d in v1_1.iterdir() if d.is_dir() and not d.name.startswith("_")):
        for type_dir in sorted(t for t in domain_dir.iterdir() if t.is_dir()):
            sources = type_dir / "sources"
            if not sources.exists():
                continue
            for pdf in sorted(sources.glob("*.pdf")):
                try:
                    with fitz.open(pdf) as d:
                        pc = d.page_count
                except Exception:
                    continue
                out.append((domain_dir.name, type_dir.name, pdf.name, pdf, pc))
    return out


def find_extend_md(sweep_dir: Path, domain: str, type_name: str, fname: str) -> Path | None:
    parser_root = sweep_dir / "extend_performance_v2_0_0"
    real = parser_root.resolve()
    md = real / domain / type_name / fname.replace(".pdf", ".md")
    return md if md.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--vision-model", default="claude-sonnet-4-5")
    ap.add_argument("--vision-all-pages-in-lite", action="store_true", default=True,
                    help="Run vision on EVERY page of structure-lite docs (default).")
    ap.add_argument("--vision-sample-rich", type=int, default=0,
                    help="Also vision N random pages per rich doc as spot-check (default 0)")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO / "runs" / "page_annotations_v3" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = discover_docs(V1_1)
    print(f"{len(docs)} docs, {sum(d[4] for d in docs)} pages total")

    # ── Step 1: per-doc features from Extend mds ────────────────────────────
    print("\nStep 1: extracting features from existing Extend mds…")
    per_doc_rows: list[dict] = []
    rich_docs: list[tuple[str, str, str, Path, int]] = []
    lite_docs: list[tuple[str, str, str, Path, int]] = []

    for domain, typ, fname, pdf, pc in docs:
        emd = find_extend_md(sweep_dir, domain, typ, fname)
        if emd is None:
            f = feature_counts("")
            cls = "no_extend_md"
            lite_docs.append((domain, typ, fname, pdf, pc))
        else:
            md = emd.read_text()
            f = feature_counts(md)
            cls = classify_doc(f, pc)
            if cls in ("rich", "has_some"):
                rich_docs.append((domain, typ, fname, pdf, pc))
            else:
                lite_docs.append((domain, typ, fname, pdf, pc))

        per_doc_rows.append({
            "domain": domain, "type": typ, "filename": fname,
            "page_count": pc,
            "extend_classification": cls,
            "n_tables_extend": f["n_tables"],
            "n_checkboxes_extend": f["n_checkboxes"],
            "n_kv_extend": f["n_kv"],
            "n_figures_extend": f["n_figures"],
            "n_form_fields_extend": f["n_form_fields"],
            "has_signature_extend": f["has_signature"],
            "extend_md_chars": f["md_chars"],
            "vision_pages_visited": 0,
            "vision_pages_with_structure": 0,
        })

    n_rich_pages = sum(d[4] for d in rich_docs)
    n_lite_pages = sum(d[4] for d in lite_docs)
    print(f"  → {len(rich_docs)} docs (with structure per Extend, {n_rich_pages} pages)")
    print(f"  → {len(lite_docs)} docs (lite — vision-verify each page, {n_lite_pages} pages)")
    cost_est = (n_lite_pages * 1500 / 1e6 * 3.0) + (n_lite_pages * 150 / 1e6 * 15.0)
    print(f"  vision cost est: ~${cost_est:.2f}")

    # ── Step 2: vision on lite docs (every page) ───────────────────────────
    per_page_rows: list[dict] = []
    if lite_docs:
        print(f"\nStep 2: vision on {n_lite_pages} pages across {len(lite_docs)} lite docs"
              f" ({args.vision_model}, {args.workers} workers)…")
        work: list[tuple] = []
        for d in lite_docs:
            for pi in range(d[4]):
                work.append((d[0], d[1], d[2], d[3], pi, d[4]))

        t0 = time.time()
        tokens_in = tokens_out = 0
        per_doc_vision_counts: dict = defaultdict(lambda: [0, 0])  # [visited, with_structure]

        def _job(domain, typ, fname, pdf, page_idx, pc):
            r = vision_classify(pdf, page_idx, args.vision_model)
            return (domain, typ, fname, page_idx, pc, r)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_job, *w) for w in work]
            for i, fut in enumerate(as_completed(futures), 1):
                domain, typ, fname, page_idx, pc, r = fut.result()
                per_doc_vision_counts[(domain, typ, fname)][0] += 1
                if r["ok"]:
                    a = r["ann"]
                    tokens_in += r["tokens_in"]
                    tokens_out += r["tokens_out"]
                    has_struct = (a.get("n_tables", 0) > 0 or a.get("n_checkboxes", 0) > 0
                                  or a.get("n_form_fields", 0) > 0 or a.get("n_kv_pairs", 0) > 0)
                    if has_struct:
                        per_doc_vision_counts[(domain, typ, fname)][1] += 1
                    per_page_rows.append({
                        "domain": domain, "type": typ, "filename": fname,
                        "page_idx": page_idx, "page_count_total": pc,
                        "primary_category": a.get("primary_category", ""),
                        "richness_score": a.get("richness_score", 0),
                        "is_blank_template": a.get("is_blank_template", False),
                        "n_tables": a.get("n_tables", 0),
                        "n_checkboxes": a.get("n_checkboxes", 0),
                        "n_form_fields": a.get("n_form_fields", 0),
                        "n_kv_pairs": a.get("n_kv_pairs", 0),
                        "n_figures": a.get("n_figures", 0),
                        "has_signature": a.get("has_signature", False),
                        "source": "vision",
                        "page_summary": (a.get("page_summary", "") or "")[:200],
                    })
                else:
                    per_page_rows.append({
                        "domain": domain, "type": typ, "filename": fname,
                        "page_idx": page_idx, "page_count_total": pc,
                        "primary_category": "error", "richness_score": 0,
                        "is_blank_template": False,
                        "n_tables": 0, "n_checkboxes": 0, "n_form_fields": 0,
                        "n_kv_pairs": 0, "n_figures": 0, "has_signature": False,
                        "source": "vision_fail",
                        "page_summary": r.get("error", "")[:150],
                    })
                if i % 30 == 0 or i == len(work):
                    print(f"  [{i}/{len(work)}] visioned", flush=True)
        wall = time.time() - t0
        cost = (tokens_in / 1e6) * 3.0 + (tokens_out / 1e6) * 15.0
        print(f"  vision done: {wall:.0f}s, tokens {tokens_in:,}/{tokens_out:,}, est. ${cost:.2f}")

        # Update per_doc_rows with vision counts
        for row in per_doc_rows:
            k = (row["domain"], row["type"], row["filename"])
            if k in per_doc_vision_counts:
                row["vision_pages_visited"] = per_doc_vision_counts[k][0]
                row["vision_pages_with_structure"] = per_doc_vision_counts[k][1]

    # ── Write outputs ───────────────────────────────────────────────────────
    pd_path = out_dir / "per_doc.csv"
    pd_fields = ["domain", "type", "filename", "page_count", "extend_classification",
                  "n_tables_extend", "n_checkboxes_extend", "n_kv_extend",
                  "n_figures_extend", "n_form_fields_extend", "has_signature_extend",
                  "extend_md_chars",
                  "vision_pages_visited", "vision_pages_with_structure"]
    with pd_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pd_fields)
        w.writeheader()
        for r in sorted(per_doc_rows, key=lambda x: (x["domain"], x["type"], x["filename"])):
            w.writerow(r)

    pp_path = out_dir / "per_page.csv"
    pp_fields = ["domain", "type", "filename", "page_idx", "page_count_total",
                  "primary_category", "richness_score", "is_blank_template",
                  "n_tables", "n_checkboxes", "n_form_fields", "n_kv_pairs",
                  "n_figures", "has_signature", "source", "page_summary"]
    with pp_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pp_fields)
        w.writeheader()
        for r in sorted(per_page_rows, key=lambda x: (x["domain"], x["type"], x["filename"], x["page_idx"])):
            w.writerow(r)

    # ── Summary ─────────────────────────────────────────────────────────────
    summary = out_dir / "summary.md"
    n_total_pages = sum(r["page_count"] for r in per_doc_rows)
    n_pages_extend_structured = sum(r["page_count"] for r in per_doc_rows
                                     if r["extend_classification"] in ("rich", "has_some"))
    n_pages_vision_struct = sum(1 for r in per_page_rows
                                  if r["source"] == "vision" and
                                  (r["n_tables"] > 0 or r["n_checkboxes"] > 0 or
                                   r["n_form_fields"] > 0 or r["n_kv_pairs"] > 0))
    n_pages_vision_total = sum(1 for r in per_page_rows if r["source"] == "vision")

    with summary.open("w") as f:
        f.write(f"# Page annotation — {out_dir.name}\n\n")
        f.write(f"**{len(per_doc_rows)} docs, {n_total_pages} pages total**\n\n")
        f.write("## Extend prefilter\n\n")
        cls_counts = defaultdict(int)
        cls_pages = defaultdict(int)
        for r in per_doc_rows:
            cls_counts[r["extend_classification"]] += 1
            cls_pages[r["extend_classification"]] += r["page_count"]
        f.write("| Classification | Docs | Pages |\n|---|---:|---:|\n")
        for c, n in sorted(cls_counts.items(), key=lambda x: -x[1]):
            f.write(f"| `{c}` | {n} | {cls_pages[c]:,} |\n")
        f.write("\n")
        f.write("## Pages with structure\n\n")
        f.write(f"- Via Extend (doc-level inferred): **{n_pages_extend_structured:,} pages** "
                f"({n_pages_extend_structured/n_total_pages*100:.0f}%) in rich/has_some docs\n")
        f.write(f"- Via vision on lite docs: **{n_pages_vision_struct} pages with structure** "
                f"out of {n_pages_vision_total} visioned ({n_pages_vision_struct/max(n_pages_vision_total,1)*100:.0f}% hit rate)\n\n")

        # Per-type breakdown
        f.write("## Per-type table & checkbox totals (from Extend)\n\n")
        f.write("| Domain | Type | Docs | n_tables | n_checkboxes | n_kv | n_figures |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        per_type: dict = defaultdict(lambda: {"docs": 0, "n_tables": 0, "n_cb": 0, "n_kv": 0, "n_fig": 0})
        for r in per_doc_rows:
            k = (r["domain"], r["type"])
            per_type[k]["docs"] += 1
            per_type[k]["n_tables"] += r["n_tables_extend"]
            per_type[k]["n_cb"] += r["n_checkboxes_extend"]
            per_type[k]["n_kv"] += r["n_kv_extend"]
            per_type[k]["n_fig"] += r["n_figures_extend"]
        for (d, t), v in sorted(per_type.items(), key=lambda kv: (-kv[1]["n_tables"] - kv[1]["n_cb"], kv[0])):
            f.write(f"| {d.replace('_', ' ')} | `{t}` | {v['docs']} | {v['n_tables']:,} | "
                    f"{v['n_cb']:,} | {v['n_kv']:,} | {v['n_fig']:,} |\n")

    print(f"\nWrote:\n  {pd_path.relative_to(REPO)}\n  {pp_path.relative_to(REPO)}\n  {summary.relative_to(REPO)}")
    print(f"\n=== SUMMARY ===")
    print(f"  {n_pages_extend_structured:,} pages in structure-rich docs (Extend prefilter)")
    print(f"  {n_pages_vision_struct} of {n_pages_vision_total} visioned pages had structure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
