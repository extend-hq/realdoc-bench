#!/usr/bin/env python3
"""Build a self-contained HTML viewer for datasets/extraction/v1.1/.

Output:
  datasets/extraction/v1.1/_viewer/
    index.html        — single page; grid grouped by domain → type
    thumbs/           — page-1 PNG thumbnail per PDF (cached, idempotent)

Open the HTML directly in a browser (file://) or via `python -m http.server`
from the v1.1 dir.
"""

from __future__ import annotations

import argparse
import html
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
V1_1 = REPO / "datasets" / "extraction" / "v1.1"
VIEWER = V1_1 / "_viewer"
THUMBS = VIEWER / "thumbs"

THUMB_DPI = 60   # ~600px wide; ~25KB per thumb


def thumb_filename(domain: str, type_dir: str, kind: str, pdf_stem: str) -> str:
    """Stable, fs-safe name for a thumbnail file."""
    # Keep parens/dashes but make spaces underscores.
    safe = lambda s: s.replace(" ", "_")
    return f"{safe(domain)}__{safe(type_dir)}__{kind}__{pdf_stem}.png"


def render_thumb(pdf: Path, out_png: Path) -> tuple[Path, bool, int]:
    """Render page 1 of pdf as PNG to out_png. Returns (out_png, ok, page_count)."""
    if out_png.exists() and out_png.stat().st_size > 1000:
        # Cached — but we still need page_count for the card. Cheap re-open.
        try:
            with fitz.open(pdf) as doc:
                return (out_png, True, doc.page_count)
        except Exception:
            pass
    try:
        with fitz.open(pdf) as doc:
            page = doc[0]
            mat = fitz.Matrix(THUMB_DPI / 72, THUMB_DPI / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out_png.parent.mkdir(parents=True, exist_ok=True)
            pix.save(out_png)
            return (out_png, True, doc.page_count)
    except Exception as e:
        print(f"  ! thumb fail {pdf.name}: {e}", flush=True)
        return (out_png, False, 0)


def discover(v1_1: Path, include_synthetic: bool = False) -> list[tuple[str, str, str, Path]]:
    """Yield (domain, type_dir, kind, pdf_path).

    Sources only by default — synthetic fills are out-of-scope for v1.1 viewing.
    """
    out = []
    kinds = ("sources",) + (("synthetic",) if include_synthetic else ())
    for domain_dir in sorted(d for d in v1_1.iterdir() if d.is_dir() and not d.name.startswith("_")):
        for type_dir in sorted(t for t in domain_dir.iterdir() if t.is_dir()):
            for kind in kinds:
                sub = type_dir / kind
                if not sub.exists():
                    continue
                for pdf in sorted(sub.glob("*.pdf")):
                    out.append((domain_dir.name, type_dir.name, kind, pdf))
    return out


HTML_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>realdoc-bench v1.1 viewer</title>
<style>
  :root {
    --bg:#0f1419; --card:#171c23; --card-syn:#1c1a28; --border:#2a323d;
    --fg:#d6dde6; --muted:#7d8a99; --accent:#7da2ff; --tag-src:#3d5a8a; --tag-syn:#7b5c9e;
  }
  html, body { background:var(--bg); color:var(--fg); margin:0;
               font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  header { padding:24px 32px; border-bottom:1px solid var(--border); position:sticky; top:0;
           background:var(--bg); z-index:10; }
  header h1 { margin:0 0 4px 0; font-size:20px; }
  header .meta { color:var(--muted); font-size:13px; }
  nav.toc { padding:12px 32px; background:#0b0f14; border-bottom:1px solid var(--border);
            display:flex; flex-wrap:wrap; gap:16px; font-size:13px; }
  nav.toc a { color:var(--accent); text-decoration:none; }
  nav.toc a:hover { text-decoration:underline; }
  main { padding:0 32px 48px; }
  h2.domain { margin:36px 0 8px; padding-top:24px; border-top:1px solid var(--border);
              font-size:18px; }
  h2.domain .count { color:var(--muted); font-weight:normal; font-size:13px; margin-left:8px; }
  h3.type { margin:24px 0 12px; font-size:15px; color:#a5b1c0; font-weight:600; }
  h3.type .count { color:var(--muted); font-weight:normal; font-size:12px; margin-left:8px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(180px, 1fr));
          gap:12px; margin-bottom:8px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:6px;
          padding:8px; transition:border-color .15s, transform .15s; }
  .card.synthetic { background:var(--card-syn); }
  .card:hover { border-color:var(--accent); transform:translateY(-1px); }
  .card a { display:block; color:inherit; text-decoration:none; }
  .thumb-wrap { aspect-ratio:8.5/11; background:#000; border-radius:4px; overflow:hidden;
                margin-bottom:6px; display:flex; align-items:center; justify-content:center; }
  .thumb-wrap img { width:100%; height:100%; object-fit:contain; }
  .thumb-wrap .placeholder { color:var(--muted); font-size:12px; }
  .filename { font-size:11px; font-family:ui-monospace,SFMono-Regular,monospace;
              color:#b5bec9; word-break:break-all; line-height:1.3;
              overflow:hidden; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; }
  .meta-row { display:flex; justify-content:space-between; align-items:center;
              margin-top:4px; font-size:10px; color:var(--muted); }
  .tag { display:inline-block; padding:1px 6px; border-radius:3px; font-weight:600;
         font-size:10px; text-transform:uppercase; letter-spacing:.04em; }
  .tag-src { background:var(--tag-src); color:#cfe0ff; }
  .tag-syn { background:var(--tag-syn); color:#e8d5ff; }
  .stats { font-variant-numeric:tabular-nums; }
  details summary { cursor:pointer; user-select:none; padding:6px 0; color:var(--muted);
                    font-size:12px; }
  details summary:hover { color:var(--fg); }
  details[open] summary { color:var(--fg); }
</style>
</head>
<body>
"""


def build_html(cards_by_domain: dict, totals: dict) -> str:
    parts = [HTML_HEAD]
    # Header
    parts.append('<header>')
    parts.append(f'<h1>realdoc-bench v1.1</h1>')
    parts.append(f'<div class="meta">'
                 f'{totals["src"]} source PDFs · {totals["syn"]} synthetic/filled · '
                 f'{totals["types"]} types · {totals["domains"]} domains</div>')
    parts.append('</header>')
    # TOC
    parts.append('<nav class="toc">')
    parts.append('<strong style="color:var(--muted);">jump:</strong>')
    for domain in cards_by_domain:
        anchor = domain.lower().replace(" ", "-").replace("_", "-").replace("(", "").replace(")", "").replace(".", "")
        parts.append(f'<a href="#{anchor}">{html.escape(domain.replace("_", " "))}</a>')
    parts.append('</nav>')
    # Main
    parts.append('<main>')
    for domain, types in cards_by_domain.items():
        anchor = domain.lower().replace(" ", "-").replace("_", "-").replace("(", "").replace(")", "").replace(".", "")
        d_src = sum(len([c for c in cards if c["kind"] == "sources"]) for cards in types.values())
        d_syn = sum(len([c for c in cards if c["kind"] == "synthetic"]) for cards in types.values())
        parts.append(f'<h2 class="domain" id="{anchor}">{html.escape(domain.replace("_", " "))}'
                     f'<span class="count">{d_src} sources · {d_syn} synthetic · {len(types)} types</span></h2>')
        for type_name, cards in types.items():
            t_src = len([c for c in cards if c["kind"] == "sources"])
            t_syn = len([c for c in cards if c["kind"] == "synthetic"])
            parts.append(f'<h3 class="type">{html.escape(type_name)}'
                         f'<span class="count">{t_src} sources · {t_syn} synthetic</span></h3>')
            # Sources first, then synthetic in a collapsed details block
            sources_cards = [c for c in cards if c["kind"] == "sources"]
            synth_cards = [c for c in cards if c["kind"] == "synthetic"]
            if sources_cards:
                parts.append('<div class="grid">')
                for c in sources_cards:
                    parts.append(_card_html(c))
                parts.append('</div>')
            if synth_cards:
                if sources_cards:
                    parts.append(f'<details><summary>+ {len(synth_cards)} synthetic fills</summary>')
                parts.append('<div class="grid">')
                for c in synth_cards:
                    parts.append(_card_html(c))
                parts.append('</div>')
                if sources_cards:
                    parts.append('</details>')
    parts.append('</main>')
    parts.append('</body></html>')
    return "\n".join(parts)


def _card_html(c: dict) -> str:
    tag_class = "tag-src" if c["kind"] == "sources" else "tag-syn"
    tag_label = "real" if c["kind"] == "sources" else "synth"
    cls = "card" + (" synthetic" if c["kind"] == "synthetic" else "")
    thumb_html = (f'<img src="{html.escape(c["thumb_rel"])}" loading="lazy" alt="">'
                  if c["thumb_ok"] else '<span class="placeholder">no thumb</span>')
    return (
        f'<div class="{cls}">'
        f'<a href="{html.escape(c["pdf_rel"])}" target="_blank">'
        f'<div class="thumb-wrap">{thumb_html}</div>'
        f'<div class="filename">{html.escape(c["filename"])}</div>'
        f'<div class="meta-row">'
        f'<span class="tag {tag_class}">{tag_label}</span>'
        f'<span class="stats">{c["pages"]}pp · {c["size_kb"]}KB</span>'
        f'</div></a></div>'
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-regen-thumbs", action="store_true",
                    help="Use cached thumbnails only; don't render new ones")
    ap.add_argument("--include-synthetic", action="store_true",
                    help="Also include AI-filled synthetic PDFs (default: sources only)")
    args = ap.parse_args()

    THUMBS.mkdir(parents=True, exist_ok=True)
    pdfs = discover(V1_1, include_synthetic=args.include_synthetic)
    print(f"Found {len(pdfs)} PDFs in v1.1/")

    # Render thumbnails (parallel)
    t0 = time.time()
    thumb_results: dict[Path, tuple[Path, bool, int]] = {}
    work = []
    for domain, type_dir, kind, pdf in pdfs:
        thumb_name = thumb_filename(domain, type_dir, kind, pdf.stem)
        thumb_path = THUMBS / thumb_name
        if args.no_regen_thumbs and not thumb_path.exists():
            thumb_results[pdf] = (thumb_path, False, 0)
            continue
        work.append((pdf, thumb_path))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(render_thumb, p, t): p for p, t in work}
        for i, fut in enumerate(as_completed(futures), 1):
            pdf = futures[fut]
            try:
                thumb_results[pdf] = fut.result()
            except Exception as e:
                thumb_results[pdf] = (THUMBS / "x", False, 0)
            if i % 50 == 0 or i == len(work):
                print(f"  thumbs {i}/{len(work)}", flush=True)
    print(f"  thumbs done in {time.time()-t0:.0f}s")

    # Build cards grouped by domain → type
    cards_by_domain: dict[str, dict[str, list[dict]]] = {}
    for domain, type_dir, kind, pdf in pdfs:
        thumb_path, thumb_ok, pages = thumb_results.get(pdf, (THUMBS / "x", False, 0))
        # Paths relative to index.html (inside _viewer/)
        pdf_rel = f"../{domain}/{type_dir}/{kind}/{pdf.name}"
        thumb_rel = f"thumbs/{thumb_path.name}"
        size_kb = pdf.stat().st_size // 1024
        cards_by_domain.setdefault(domain, {}).setdefault(type_dir, []).append({
            "filename": pdf.name, "kind": kind, "pages": pages,
            "size_kb": size_kb, "thumb_ok": thumb_ok,
            "thumb_rel": thumb_rel, "pdf_rel": pdf_rel,
        })

    # Totals
    totals = {
        "src": sum(1 for d, t, k, _ in pdfs if k == "sources"),
        "syn": sum(1 for d, t, k, _ in pdfs if k == "synthetic"),
        "types": len({(d, t) for d, t, _, _ in pdfs}),
        "domains": len({d for d, _, _, _ in pdfs}),
    }

    html_text = build_html(cards_by_domain, totals)
    index = VIEWER / "index.html"
    index.write_text(html_text)
    print(f"\nWrote {index.relative_to(REPO)}")
    print(f"  open with:  open '{index}'")
    print(f"  or serve:   (cd {V1_1} && python -m http.server 8080)  →  http://localhost:8080/_viewer/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
