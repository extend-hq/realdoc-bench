#!/usr/bin/env python3
"""Polish pass: run gpt-image-2 over existing synth PDFs to fill the empty
AcroForm fields that the Phase-A widget fill missed.

Input: datasets/extraction/v1_<type>/synthetic/*.pdf  (already rasterized)
       datasets/extraction/v1_<type>/_synthetic_manifest.csv  (persona index)
       datasets/extraction/_personas/<type>.json  (persona values)

Output: overwrites the synth PDF in place (after polish).

For each synth PDF page:
  - render at 144 dpi
  - send to gpt-image-2 with the persona's fields and a "fill remaining blanks" prompt
  - replace the page with the returned PNG

CLI:
  --type v1_acord_125_126   restrict to one type
  --doc <filename>          restrict to one specific synth PDF
  --limit N                 cap number of synth PDFs to polish
  --max-pages N             cap pages per PDF (cost guard)
  --dry-run                 print plan without API calls
  --smoke                   polish just one PDF as a demo
  --image-model <name>      override (default gpt-image-2)
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
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets" / "extraction"
PERSONAS_DIR = DATASETS / "_personas"
IMAGE_GEN_MODEL_DEFAULT = "gpt-image-2"


def load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k and v and not os.environ.get(k):
            os.environ[k] = v

load_dotenv()


_oai = None
def get_openai_client():
    global _oai
    if _oai is None:
        from openai import OpenAI
        _oai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _oai


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_polish_targets(doc_types: list[str] | None, doc_name: str | None) -> list[tuple[Path, dict]]:
    """Returns list of (synth_pdf_path, persona_dict)."""
    targets = []
    for folder in sorted(DATASETS.glob("v1_*")):
        if doc_types and folder.name not in doc_types:
            continue
        synth_dir = folder / "synthetic"
        if not synth_dir.exists():
            continue
        # Load personas for this type
        personas_path = PERSONAS_DIR / f"{folder.name}.json"
        if not personas_path.exists():
            continue
        try:
            personas = json.loads(personas_path.read_text())
        except Exception:
            continue
        # Walk synth PDFs (use manifest if present, else parse filename)
        manifest = folder / "_synthetic_manifest.csv"
        if manifest.exists():
            with manifest.open() as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                if doc_name and r["filename"] != doc_name:
                    continue
                pdf = synth_dir / r["filename"]
                if not pdf.exists():
                    continue
                try:
                    pi = int(r.get("persona_idx", 0))
                except (TypeError, ValueError):
                    pi = 0
                persona = personas[pi] if pi < len(personas) else personas[0]
                targets.append((pdf, persona))
        else:
            for pdf in sorted(synth_dir.glob("*.pdf")):
                if doc_name and pdf.name != doc_name:
                    continue
                m = re.match(r"^(.*?)__p(\d+)$", pdf.stem)
                if not m:
                    continue
                pi = int(m.group(2))
                persona = personas[pi] if pi < len(personas) else personas[0]
                targets.append((pdf, persona))
    return targets


# ---------------------------------------------------------------------------
# gpt-image-2 polish
# ---------------------------------------------------------------------------

POLISH_PROMPT = """TASK: This image shows a form that has been PARTIALLY filled in (some fields already have typed values). Your job is to COMPLETE the form by filling in the remaining blank fields, using the persona values below.

ABSOLUTE RULES:
1. DO NOT overwrite, modify, or erase any values already typed in the form.
2. DO NOT change any printed label, heading, table border, line, logo, footer, page number, or watermark.
3. DO NOT alter the form layout, dimensions, or column structure.
4. ONLY add ink where a blank fillable area (underline, empty box, empty cell, empty checkbox) currently has no value.
5. Match the typing style of the already-filled values when adding new text.
6. For signature/initial fields: handwritten cursive blue ink.
7. If a blank field has no relevant value below, leave it BLANK. It is normal for real-world forms to have many unfilled fields.

CHECKBOX RULES — DEFAULT IS UNCHECKED. READ CAREFULLY:
- The DEFAULT state of every checkbox is UNCHECKED. Do not check a box unless you have a SPECIFIC reason from the persona below.
- Existing checkmarks (already drawn in the input) must be preserved exactly. Do not erase or add to them.
- For YES/NO pairs near a question: pick exactly ONE based on persona — usually NO is the safe default for risk/loss questions (lawsuits, claims, fraud, etc.).
- For mutually exclusive radio groups (Single/Married/Divorced; Male/Female; Owner/Tenant; Primary/Secondary): check EXACTLY ONE, leave others empty.
- For multi-select lists (Lines of Business, Sections Attached, Race, Coverage Types, Hazards): check AT MOST 2 boxes total, and ONLY if the persona's industry/situation explicitly matches them. If unsure, leave ALL UNCHECKED.
- For long checkbox lists (>8 options): typical real-world fill is 0-2 boxes. If you can't pick with high confidence, check 0.
- HARD CEILINGS PER GROUP:
    * checkbox group with 2 options (Yes/No) → check at most 1
    * checkbox group with 3-6 options → check at most 1
    * checkbox group with 7-15 options → check at most 2
    * checkbox group with >15 options → check at most 2
- DO NOT check every checkbox in a row "to be safe." An over-checked form is unrealistic and worse than an under-checked one.
- Use hand-drawn X or ✓ marks (not solid fills, not large blobs).

TEXT FIELD RULES — anti-hallucination:
- Do not paste the KEY name into the field. Only paste the VALUE.
- If a value looks like a Python boolean ("True"/"False") or a code/identifier from the persona keys, do not write it into a text field that expects a human-readable label.
- If a value is too long for a small field, truncate sensibly, do not bleed into adjacent fields.
- Do not repeat the same long value across multiple unrelated cells.

Persona: {persona_label}

Values to fill in (KEY → VALUE):
{values_md}

The output must be visually indistinguishable from the input except for the newly-added values in previously-empty fields. Realistic, not over-checked.
"""


def render_page_png(page: fitz.Page, dpi: int = 144) -> bytes:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def persona_values_markdown(fields: dict, max_entries: int = 60) -> str:
    lines = []
    for k, v in list(fields.items())[:max_entries]:
        if not isinstance(v, (str, int, float, bool)):
            continue
        nice = k.replace("_", " ").title()
        lines.append(f"- {nice}: {v}")
    return "\n".join(lines)


def gpt_image_polish(page_png: bytes, persona: dict, page_size_pt: tuple[float, float],
                     model: str) -> bytes | None:
    client = get_openai_client()
    label = persona.get("persona_label", "synthetic persona")
    fields = persona.get("fields", persona)
    prompt = POLISH_PROMPT.format(persona_label=label,
                                  values_md=persona_values_markdown(fields))
    w, h = page_size_pt
    size = "1024x1536" if h >= w else "1536x1024"
    try:
        resp = client.images.edit(
            model=model,
            image=("page.png", io.BytesIO(page_png), "image/png"),
            prompt=prompt,
            size=size,
            n=1,
        )
        return base64.b64decode(resp.data[0].b64_json)
    except Exception as e:
        print(f"   [image-gen err] {e}", flush=True)
        return None


def replace_page_with_image(doc: fitz.Document, page_idx: int, img_png: bytes):
    old = doc[page_idx]
    rect = old.rect
    doc.insert_page(page_idx, width=rect.width, height=rect.height)
    new_page = doc[page_idx]
    new_page.insert_image(rect, stream=img_png, keep_proportion=False)
    doc.delete_page(page_idx + 1)


def polish_one(pdf_path: Path, persona: dict, model: str, max_pages: int | None) -> int:
    """Polish one synth PDF in place. Returns number of pages polished."""
    doc = fitz.open(pdf_path)
    n_pages = doc.page_count
    if max_pages is not None:
        n_pages = min(n_pages, max_pages)
    polished = 0
    for pidx in range(n_pages):
        page = doc[pidx]
        png = render_page_png(page, dpi=144)
        filled = gpt_image_polish(png, persona,
                                  (page.rect.width, page.rect.height), model)
        if filled is None:
            print(f"   page {pidx+1}: SKIP (gen failed)", flush=True)
            continue
        replace_page_with_image(doc, pidx, filled)
        polished += 1
        print(f"   page {pidx+1}: polished", flush=True)
    # Save back in place (atomic via temp)
    tmp_path = pdf_path.with_suffix(".pdf.polishing")
    doc.save(tmp_path, garbage=4, deflate=True)
    doc.close()
    tmp_path.replace(pdf_path)
    return polished


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", action="append", dest="types",
                    help="restrict to one type (repeatable); e.g. --type v1_cms_1500 --type v1_acord_125_126")
    ap.add_argument("--doc")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel image-edit workers (default 8)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--image-model", default=os.environ.get("IMAGE_GEN_MODEL", IMAGE_GEN_MODEL_DEFAULT))
    args = ap.parse_args()

    targets = find_polish_targets(args.types, args.doc)
    if not targets:
        print("no polish targets")
        return 1

    if args.smoke:
        targets = targets[:1]
    elif args.limit:
        targets = targets[: args.limit]

    # Cost estimate
    total_pages = 0
    for pdf, _ in targets:
        try:
            with fitz.open(pdf) as d:
                p = d.page_count
                if args.max_pages: p = min(p, args.max_pages)
                total_pages += p
        except Exception:
            pass
    print(f"Plan: polish {len(targets)} synth PDFs, ~{total_pages} total pages")
    print(f"  Estimated cost: ~${total_pages * 0.20:.0f}, wall ~{total_pages * 140 / (3600 * args.workers):.1f}h "
          f"with --workers={args.workers}")
    print(f"  Model: {args.image_model}")
    if args.dry_run:
        for pdf, persona in targets[:10]:
            print(f"  {pdf.relative_to(REPO)}  (persona: {persona.get('persona_label','?')})")
        if len(targets) > 10:
            print(f"  ... +{len(targets)-10} more")
        return 0

    from concurrent.futures import ThreadPoolExecutor, as_completed
    t0 = time.time()
    n_done = 0
    n_fail = 0
    print(f"\nLaunching {args.workers} parallel workers across {len(targets)} PDFs...", flush=True)

    def _job(pdf_path: Path, persona: dict) -> tuple[Path, str, int | None, str | None]:
        try:
            pages = polish_one(pdf_path, persona, args.image_model, args.max_pages)
            return (pdf_path, persona.get("persona_label", "?"), pages, None)
        except Exception as e:
            return (pdf_path, persona.get("persona_label", "?"), None, str(e))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_job, pdf, persona) for pdf, persona in targets]
        for i, fut in enumerate(as_completed(futures), 1):
            pdf_path, label, pages, err = fut.result()
            rel = pdf_path.relative_to(REPO)
            if err:
                n_fail += 1
                print(f"  [{i}/{len(targets)}] FAIL {rel} ({label}): {err}", flush=True)
            else:
                n_done += 1
                elapsed = time.time() - t0
                eta = (elapsed / max(n_done + n_fail, 1)) * (len(targets) - n_done - n_fail)
                print(f"  [{i}/{len(targets)}] ✓ {rel} ({pages}pp, {label})"
                      f"  elapsed {elapsed:.0f}s eta {eta:.0f}s", flush=True)

    print(f"\nDone — polished {n_done}/{len(targets)} (fail {n_fail}) in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
