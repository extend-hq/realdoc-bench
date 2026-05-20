#!/usr/bin/env python3
"""Page-level feature annotation using existing parser output as prefilter.

Strategy (in order of cost):
  1. CHEAP — Per-page text via pymupdf (existing pymupdf parser md, split by
     '## Page N'). Regex-detect tables, checkboxes, key-values, signatures.
  2. PREFILTER — Cross-check counts against Extend md (which preserves
     checkbox state as [ ] / [x] more reliably than pymupdf).
  3. FALLBACK — For pages where pymupdf returned ≤80 chars (likely raster /
     image-only synth fills), render image and call Claude vision (Haiku
     by default for cheapness).

Output: runs/page_annotations/<ts>/manifest.csv with per-page features.
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
# Regex feature detection
# ─────────────────────────────────────────────────────────────────────────────

# Markdown table block: 2+ consecutive lines starting/ending with |
TABLE_BLOCK_RE = re.compile(r"((?:^[ \t]*\|[^\n]*\n){2,})", re.MULTILINE)
# Markdown checkbox: [ ] [x] [X] [✓]
MD_CHECKBOX_RE = re.compile(r"\[[ xX✓]\]")
# Unicode checkbox glyphs
UNICODE_CHECKBOX_RE = re.compile(r"[☐☑□■⬜⬛✓✗✔]")
# Key-value pair on a line: "Label: value"
KV_RE = re.compile(r"^[A-Z][A-Za-z0-9 /()&'-]{2,50}:\s+[A-Za-z0-9$/]", re.MULTILINE)
# Form-field underline pattern: "Label: _____" or "Name __________"
FORM_FIELD_RE = re.compile(r"_{4,}|\.{6,}")
# Signature line: "Signature" + underscores OR "_______ Date" patterns
SIGNATURE_RE = re.compile(r"(signature|signed by|/s/)[^\n]*?(?:_{3,}|\Z|\n)", re.IGNORECASE)
# Yes/No checkbox pairs (common form pattern)
YES_NO_RE = re.compile(r"\[\s?[xX]?\s?\]\s*(yes|no)", re.IGNORECASE)


def detect_features_from_text(text: str) -> dict:
    """Cheap regex-based feature counts for a single page's text."""
    if not text:
        return {
            "n_tables": 0, "n_checkboxes": 0, "n_form_fields": 0,
            "n_kv_pairs": 0, "has_signature": False, "text_len": 0,
        }
    n_tables = len(TABLE_BLOCK_RE.findall(text))
    n_md_cb = len(MD_CHECKBOX_RE.findall(text))
    n_uni_cb = len(UNICODE_CHECKBOX_RE.findall(text))
    n_form_fields = len(FORM_FIELD_RE.findall(text))
    n_kv = len(KV_RE.findall(text))
    has_sig = bool(SIGNATURE_RE.search(text))
    return {
        "n_tables": n_tables,
        "n_checkboxes": n_md_cb + n_uni_cb,
        "n_form_fields": n_form_fields,
        "n_kv_pairs": n_kv,
        "has_signature": has_sig,
        "text_len": len(text),
    }


def detect_features_pymupdf(pdf_path: Path, page_idx: int) -> dict:
    """Better detection via pymupdf — uses page.find_tables() for true table
    detection (geometric) plus widget detection for AcroForm fields."""
    try:
        with fitz.open(pdf_path) as doc:
            if page_idx >= doc.page_count:
                return {}
            page = doc[page_idx]
            text = page.get_text("text") or ""
            n_tables = 0
            try:
                tabs = page.find_tables()
                # tabs.tables in modern pymupdf, or len(tabs) in older
                if hasattr(tabs, "tables"):
                    n_tables = len(tabs.tables)
                else:
                    n_tables = len(tabs) if tabs else 0
            except Exception:
                n_tables = 0
            # AcroForm widgets on this page
            n_widgets = 0
            try:
                widgets = list(page.widgets() or [])
                n_widgets = len(widgets)
            except Exception:
                pass
        # Combine regex + geometric/widget signal
        text_feats = detect_features_from_text(text)
        return {
            "n_tables": max(text_feats["n_tables"], n_tables),
            "n_checkboxes": text_feats["n_checkboxes"],
            "n_form_fields": max(text_feats["n_form_fields"], n_widgets),
            "n_kv_pairs": text_feats["n_kv_pairs"],
            "has_signature": text_feats["has_signature"],
            "text_len": text_feats["text_len"],
            "n_widgets": n_widgets,
        }
    except Exception:
        return {"n_tables": 0, "n_checkboxes": 0, "n_form_fields": 0,
                "n_kv_pairs": 0, "has_signature": False, "text_len": 0,
                "n_widgets": 0}


def classify_page(features: dict) -> tuple[str, int, bool]:
    """Return (primary_category, richness_score 1-10, is_blank_template)."""
    nt = features["n_tables"]
    ncb = features["n_checkboxes"]
    nff = features["n_form_fields"]
    nkv = features["n_kv_pairs"]
    text_len = features["text_len"]

    # Heuristic richness: more structure = higher score
    structure = nt * 3 + ncb * 1 + nff * 0.5 + nkv * 0.4
    # Text density factor: very short pages can't be rich
    if text_len < 100:
        richness = 1
        cat = "blank"
        blank = True
        return cat, richness, blank

    # Primary category: pick the dominant signal
    is_blank = (ncb > 0 and nff > 10 and nkv == 0)  # form template with no fills
    if nt >= 2 and structure < ncb:
        cat = "tables"
    elif ncb >= 3:
        cat = "checkboxes" if ncb > nt * 2 else "form"
    elif nff >= 6 or (ncb > 0 and nff > 0):
        cat = "form"
    elif nt >= 1 and (nkv > 0 or ncb > 0):
        cat = "mixed"
    elif nkv >= 5:
        cat = "key_values"
    elif nt >= 1:
        cat = "tables"
    elif nkv >= 1 or ncb >= 1 or nff >= 2:
        cat = "key_values"
    else:
        cat = "prose"

    # Richness 1-10
    if structure >= 25:
        richness = 10
    elif structure >= 15:
        richness = 8
    elif structure >= 8:
        richness = 6
    elif structure >= 3:
        richness = 4
    elif structure >= 1:
        richness = 3
    else:
        richness = 1 if cat == "prose" else 2

    return cat, richness, is_blank


# ─────────────────────────────────────────────────────────────────────────────
# Locate pymupdf & extend mds (in our sweep)
# ─────────────────────────────────────────────────────────────────────────────

def find_parser_md(sweep_dir: Path, parser: str, domain: str, type_name: str,
                   fname: str) -> Path | None:
    parser_root = sweep_dir / parser
    real = parser_root.resolve()
    md = real / domain / type_name / fname.replace(".pdf", ".md")
    return md if md.exists() else None


def split_pymupdf_pages(md_text: str) -> list[str]:
    """Split pymupdf md by '## Page N' headers. Returns list of per-page texts."""
    parts = re.split(r"^##\s*Page\s+\d+\s*$", md_text, flags=re.MULTILINE)
    # First part is empty (before first header)
    return [p.strip() for p in parts[1:]] if parts else []


# ─────────────────────────────────────────────────────────────────────────────
# Vision fallback
# ─────────────────────────────────────────────────────────────────────────────

VISION_PROMPT = """Look at this document page image. Return ONLY a JSON object:
{
  "n_tables": <int>,
  "n_checkboxes": <int — visible checkbox glyphs>,
  "n_form_fields": <int — fillable boxes/lines>,
  "n_kv_pairs": <int — labeled values>,
  "has_signature": <bool>,
  "primary_category": <"tables" | "checkboxes" | "form" | "key_values" | "prose" | "mixed" | "blank">,
  "is_blank_template": <bool>,
  "richness_score": <int 1-10>,
  "page_summary": <string ≤ 15 words>
}"""


def render_page_b64(pdf_path: Path, page_idx: int, dpi: int = 110) -> str | None:
    try:
        with fitz.open(pdf_path) as doc:
            page = doc[page_idx]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            return base64.b64encode(pix.tobytes("png")).decode()
    except Exception:
        return None


def annotate_with_vision(pdf_path: Path, page_idx: int, model: str) -> dict:
    b64 = render_page_b64(pdf_path, page_idx)
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
        ann = json.loads(text)
        return {"ok": True, "ann": ann,
                "tokens_in": resp.usage.input_tokens,
                "tokens_out": resp.usage.output_tokens}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def discover_docs(v1_1: Path) -> list[tuple[str, str, str, Path, int]]:
    """Yield (domain, type, fname, pdf, page_count)."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True,
                    help="runs/parse_sweep/<ts>/ to read pymupdf + extend mds from")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--vision-model", default="claude-haiku-4-5",
                    help="Model for raster fallback (cheap by default)")
    ap.add_argument("--skip-vision", action="store_true",
                    help="Skip vision fallback for raster pages; record as 'blank'")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO / "runs" / "page_annotations" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = discover_docs(V1_1)
    total_pages = sum(d[4] for d in docs)
    print(f"{len(docs)} docs, {total_pages} pages total")

    # Pre-compute extend doc-level stats for prefilter (free signal)
    print("\nReading Extend mds for doc-level prefilter…")
    extend_doc_features: dict = {}
    for domain, typ, fname, pdf, pc in docs:
        emd = find_parser_md(sweep_dir, "extend_performance_v2_0_0", domain, typ, fname)
        if emd is None:
            extend_doc_features[(domain, typ, fname)] = None
            continue
        text = emd.read_text()
        ef = detect_features_from_text(text)
        # average per page (only used as sanity check)
        ef["per_page_table_avg"] = ef["n_tables"] / max(pc, 1)
        ef["per_page_cb_avg"] = ef["n_checkboxes"] / max(pc, 1)
        extend_doc_features[(domain, typ, fname)] = ef

    # Build work: per-page text from pymupdf
    print("\nClassifying pages…")
    rows: list[dict] = []
    n_vision_calls = 0
    vision_pages: list[tuple] = []  # (domain, typ, fname, pdf, page_idx)
    vision_tokens_in = vision_tokens_out = 0

    for domain, typ, fname, pdf, pc in docs:
        pmd = find_parser_md(sweep_dir, "pymupdf", domain, typ, fname)
        page_texts: list[str] = []
        if pmd is not None:
            page_texts = split_pymupdf_pages(pmd.read_text())
            # Ensure we have exactly pc entries; pad with empty if needed
            while len(page_texts) < pc:
                page_texts.append("")
            page_texts = page_texts[:pc]
        else:
            page_texts = [""] * pc
        ext_features = extend_doc_features.get((domain, typ, fname))
        for i, page_text in enumerate(page_texts):
            feats = detect_features_from_text(page_text)
            # If text is sparse AND extend says this doc has structure → vision needed
            needs_vision = (
                feats["text_len"] < 80 and
                ext_features is not None and
                (ext_features["n_tables"] > 0 or ext_features["n_checkboxes"] > 0 or
                 ext_features["n_form_fields"] > 0)
            )
            if needs_vision:
                vision_pages.append((domain, typ, fname, pdf, i, pc))
            else:
                cat, rich, blank = classify_page(feats)
                rows.append({
                    "domain": domain, "type": typ, "filename": fname,
                    "page_idx": i, "page_count_total": pc,
                    "primary_category": cat, "richness_score": rich,
                    "is_blank_template": blank,
                    "n_tables": feats["n_tables"],
                    "n_checkboxes": feats["n_checkboxes"],
                    "n_form_fields": feats["n_form_fields"],
                    "n_kv_pairs": feats["n_kv_pairs"],
                    "has_signature": feats["has_signature"],
                    "text_len": feats["text_len"],
                    "source": "pymupdf+regex",
                    "page_summary": "",
                })

    print(f"  cheap-classified: {len(rows)} pages")
    print(f"  need vision fallback: {len(vision_pages)} pages (raster + structure-likely)")

    # Vision fallback
    if vision_pages and not args.skip_vision:
        print(f"\nRunning vision fallback ({args.vision_model}, {args.workers} workers)…")
        t0 = time.time()
        def _job(domain, typ, fname, pdf, page_idx, pc):
            r = annotate_with_vision(pdf, page_idx, args.vision_model)
            return (domain, typ, fname, page_idx, pc, r)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_job, *vp) for vp in vision_pages]
            for j, fut in enumerate(as_completed(futures), 1):
                domain, typ, fname, page_idx, pc, r = fut.result()
                if r["ok"]:
                    a = r["ann"]
                    vision_tokens_in += r.get("tokens_in", 0)
                    vision_tokens_out += r.get("tokens_out", 0)
                    rows.append({
                        "domain": domain, "type": typ, "filename": fname,
                        "page_idx": page_idx, "page_count_total": pc,
                        "primary_category": a.get("primary_category", ""),
                        "richness_score": a.get("richness_score", 0),
                        "is_blank_template": a.get("is_blank_template", False),
                        "n_tables": a.get("n_tables", 0),
                        "n_checkboxes": a.get("n_checkboxes", 0),
                        "n_form_fields": a.get("n_form_fields", 0),
                        "n_kv_pairs": a.get("n_kv_pairs", 0),
                        "has_signature": a.get("has_signature", False),
                        "text_len": 0,
                        "source": "vision",
                        "page_summary": (a.get("page_summary", "") or "")[:200],
                    })
                else:
                    rows.append({
                        "domain": domain, "type": typ, "filename": fname,
                        "page_idx": page_idx, "page_count_total": pc,
                        "primary_category": "unknown", "richness_score": 0,
                        "is_blank_template": False,
                        "n_tables": 0, "n_checkboxes": 0, "n_form_fields": 0,
                        "n_kv_pairs": 0, "has_signature": False,
                        "text_len": 0, "source": "vision_fail",
                        "page_summary": r.get("error", "")[:200],
                    })
                if j % 50 == 0 or j == len(vision_pages):
                    print(f"  [{j}/{len(vision_pages)}] vision done", flush=True)
        wall = time.time() - t0
        # Haiku 4.5 pricing: $0.80/M input, $4/M output
        cost = (vision_tokens_in / 1e6) * 0.80 + (vision_tokens_out / 1e6) * 4.0
        print(f"  vision: {len(vision_pages)} pages in {wall:.0f}s, "
              f"tokens {vision_tokens_in:,}/{vision_tokens_out:,}, est. ${cost:.2f}")

    # Write manifest
    manifest_path = out_dir / "manifest.csv"
    fields = ["domain", "type", "filename", "page_idx", "page_count_total",
              "primary_category", "richness_score", "is_blank_template",
              "n_tables", "n_checkboxes", "n_form_fields", "n_kv_pairs",
              "has_signature", "text_len", "source", "page_summary"]
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["domain"], x["type"], x["filename"], x["page_idx"])):
            w.writerow(r)

    # Summary
    print(f"\nWrote {manifest_path.relative_to(REPO)}  ({len(rows)} page rows)")
    print()
    print("=== Per-category page counts ===")
    cat_counts: dict = defaultdict(int)
    for r in rows:
        cat_counts[r["primary_category"]] += 1
    for cat, n in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:15} {n:>5} pages")
    print()
    print("=== Pages with any structure (tables/checkboxes/form/key-vals) ===")
    rich_n = sum(1 for r in rows if r["primary_category"] in {"tables", "checkboxes", "form", "key_values", "mixed"})
    print(f"  {rich_n} of {len(rows)} pages ({rich_n / max(len(rows), 1) * 100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
