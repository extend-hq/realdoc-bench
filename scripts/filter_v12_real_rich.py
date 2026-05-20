#!/usr/bin/env python3
"""Classify every real page in v1.2 via Gemini 2.5 Flash and drop prose/blank.

Keep: tables, form, checkboxes, charts, figures, handwriting, mixed
Drop: prose, blank

Writes:
  datasets/extraction/v1.2/_real_classifications.csv  (every page + category)
  Deletes prose/blank PDFs from v1.2/real/<dom>/ and removes from _manifest.csv.

Cost: ~$4 for 8k pages at Gemini 2.5 Flash pricing.
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
from pathlib import Path

import fitz
import requests

REPO = Path(__file__).resolve().parent.parent
V12 = REPO / "datasets" / "extraction" / "v1.2"

CLASSIFY_PROMPT = """Classify this single document page into ONE category based on its DOMINANT visual content.

Categories (pick the one that best matches):

- handwriting     ← ANY visible handwritten text, signatures, filled-by-hand fields, scribbled notes, hand-drawn marks (court FILED stamps, hand-written case numbers, etc.). Highest priority — if you see real handwriting, return this.
- form            ← dense form layout (labeled blanks/boxes for filling out, ≥10 form fields with labels), filled or unfilled. E.g. tax returns, applications, intake forms.
- checkboxes      ← page is dominated by checkbox arrays (≥5 checkboxes visible) — questionnaires, eligibility forms.
- tables          ← dense data tables (≥3 columns × ≥10 rows), e.g. financial statements, schedules, ledgers, line-item lists.
- charts          ← charts/graphs/plots dominate the page (bar/line/pie/scatter/timeline). Must be a real visualization, not just a layout box.
- figures         ← diagrams, technical drawings, maps, photographs, anatomy illustrations, flowcharts (non-chart visuals).
- mixed           ← page combines 2+ of the above rich categories in roughly equal weight (e.g. half tables + half form).
- prose           ← page is dominantly narrative TEXT in paragraphs — like SEC 10-K narrative, audit report discussion, ESG report prose, op-ed style text. NO dense tables, NO forms, NO charts, NO handwriting.
- blank           ← page is mostly empty white space (cover pages, divider pages, sometimes single-line headers).

Return ONLY the category word. Just one word."""


def load_env(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"(?:export\s+)?([A-Z_]+)\s*=\s*(.*)", line)
        if m:
            k, v = m.group(1), m.group(2).strip().strip('"').strip("'")
            os.environ.setdefault(k, v)


def render_page_b64(pdf_path: Path, page_idx: int = 0, dpi_scale: float = 0.9) -> str:
    """Render first page (or specified) at low res for classification."""
    with fitz.open(pdf_path) as d:
        pix = d[page_idx].get_pixmap(matrix=fitz.Matrix(dpi_scale, dpi_scale))
        return base64.b64encode(pix.tobytes("png")).decode()


def classify_page(pdf_path: Path, gemini_key: str) -> str:
    try:
        b64 = render_page_b64(pdf_path)
    except Exception as e:
        return f"err_render:{type(e).__name__}"
    body = {
        "contents": [{
            "parts": [
                {"text": CLASSIFY_PROMPT},
                {"inline_data": {"mime_type": "image/png", "data": b64}},
            ]
        }],
        # Enough tokens to reliably emit ONE category word. Gemini 2.5 needs
        # extra headroom for the thinking-token preamble.
        "generationConfig": {"temperature": 0, "maxOutputTokens": 256,
                             "thinkingConfig": {"thinkingBudget": 0}},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
    try:
        r = requests.post(url, json=body, timeout=45)
        if r.status_code != 200:
            return f"http_{r.status_code}"
        data = r.json()
        cands = data.get("candidates") or []
        if not cands:
            return "no_cand"
        cand = cands[0]
        # SAFETY/RECITATION/etc may yield empty content
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        # Find first part with usable text
        text = ""
        for p in parts:
            if isinstance(p, dict) and p.get("text") and not p.get("thought"):
                text = p["text"]
                break
        if not text:
            return f"empty_{cand.get('finishReason','?')}"
        ans = re.sub(r"[^a-z_]", " ", text.strip().lower()).split()
        return ans[0] if ans else "?"
    except Exception as e:
        return f"err_{type(e).__name__}"


RICH = {"tables", "form", "checkboxes", "charts", "figures", "handwriting", "mixed"}
DROP = {"prose", "blank"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true",
                    help="Classify but don't delete files / mutate manifest")
    ap.add_argument("--reclassify", action="store_true",
                    help="Re-classify all (ignore cached _real_classifications.csv)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only classify first N pages (testing)")
    args = ap.parse_args()

    load_env()
    gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("ERROR: GOOGLE_API_KEY not set", file=sys.stderr)
        return 1

    # Real pages from v1.2 manifest
    m = V12 / "_manifest.csv"
    rows = [r for r in csv.DictReader(m.open()) if r["kind"] == "real"]
    print(f"Total real pages in v1.2: {len(rows)}")

    # Load cache if present
    cache_path = V12 / "_real_classifications.csv"
    cache = {}
    if cache_path.exists() and not args.reclassify:
        for r in csv.DictReader(cache_path.open()):
            cache[r["dest_path"]] = r["category"]
        print(f"Cached classifications: {len(cache)}")

    # Plan: classify uncached only
    todo = [r for r in rows if r["dest_path"] not in cache]
    if args.limit:
        todo = todo[:args.limit]
    print(f"To classify: {len(todo)}")

    # Classify in parallel
    classifications = dict(cache)
    if todo:
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(classify_page, V12 / r["dest_path"], gemini_key): r
                for r in todo
            }
            for fut in as_completed(futures):
                r = futures[fut]
                cat = fut.result()
                classifications[r["dest_path"]] = cat
                done += 1
                if done % 100 == 0 or done == len(todo):
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 0.01)
                    eta = (len(todo) - done) / max(rate, 0.01)
                    print(f"  [{done}/{len(todo)}] {rate:.1f}/s, ETA {eta/60:.1f}min", flush=True)

        # Save cache
        with cache_path.open("w", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(["dest_path", "category"])
            for p, c in sorted(classifications.items()):
                w.writerow([p, c])
        print(f"Saved cache: {cache_path.relative_to(REPO)}")

    # Summary
    from collections import Counter
    cnt = Counter(classifications.values())
    print()
    print(f"=== category distribution ===")
    for cat, n in sorted(cnt.items(), key=lambda x: -x[1]):
        marker = "DROP" if cat in DROP else ("KEEP" if cat in RICH else "ERR/UNK")
        print(f"  {cat:25} {n:>5}  [{marker}]")

    # If not dry-run: delete DROP pages
    n_dropped = 0
    if not args.dry_run:
        to_drop = [r for r in rows if classifications.get(r["dest_path"]) in DROP]
        print(f"\nDropping {len(to_drop)} prose/blank pages...")
        for r in to_drop:
            p = V12 / r["dest_path"]
            if p.exists():
                p.unlink()
                n_dropped += 1
        # Prune manifest
        keep_rows = [
            r for r in csv.DictReader(m.open())
            if r["kind"] != "real" or classifications.get(r["dest_path"]) not in DROP
        ]
        # Re-read with original fields
        all_rows = list(csv.DictReader(m.open()))
        fields = list(all_rows[0].keys())
        new_rows = [
            r for r in all_rows
            if r["kind"] != "real" or classifications.get(r["dest_path"]) not in DROP
        ]
        with m.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
            w.writeheader()
            for r in new_rows:
                w.writerow(r)
        print(f"  deleted {n_dropped} PDFs")
        print(f"  manifest: {len(all_rows)} → {len(new_rows)} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
