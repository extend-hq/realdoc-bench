"""Generate a self-contained HTML report that visually compares each parser's
answers to every schema field, side-by-side, with source-PDF page thumbnails.

For each doc in extraction/v0:
  - One section per doc, with a table of (field × parser → answer)
  - Color-coded by score (green=exact, yellow=fuzzy, red=mismatch, grey=null_match)
  - Each parser cell shows the actual value + score; hovering reveals the
    surrounding context from that parser's markdown
  - A small PDF page thumbnail appears next to every field, jumping to the
    first page where the expected value occurs in pymupdf's parse
  - Filter chips at the top (by doc, by win/loss, by difficulty)

Run:
  python scripts/build_comparison_report.py
  open runs/comparison.html
"""

from __future__ import annotations

import base64
import html
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "datasets" / "extraction" / "v0"
OUT = REPO / "runs" / "comparison.html"
THUMBS = REPO / "runs" / ".page_thumbs"

RUN_DIRS = [
    "runs/bench_extract_v0",
    "runs/bench_extract_v0_v2fix",
    "runs/bench_extract_v2_ga",
    "runs/bench_extract_dots_mocr",
    "runs/bench_extract_claude_sonnet_4_6",
    "runs/bench_extract_gemini_3_flash",
    "runs/bench_extract_reducto_v2",
    "runs/bench_extract_llamaparse",
    "runs/bench_extract_azure_di",
    "runs/bench_extract_textract",
    "runs/bench_extract_claude16k",
]
PARSER_ORDER = [
    # Extend v1 and v2-beta dropped from the visual comparison per user request.
    # Comparing only the GA-grade Performance, plus both Light variants.
    "extend_performance_v2_0_0",
    "extend_light_v1_0_0",
    "extend_light_v0_0_2_beta",
    "aws_textract",
    "azure_di",
    "llamaparse",
    "reducto",
    "gemini_3_flash",
    "claude_sonnet_4_6",
    "dots_mocr",
    "pymupdf",
]
PARSER_LABELS = {
    "extend_performance_v1_0_0": "Extend v1",
    "extend_performance_v2_0_0": "Extend v2 GA",
    "extend_performance_v2_0_0_beta": "Extend v2-beta",
    "extend_light_v1_0_0": "Extend Light v1",
    "extend_light_v0_0_2_beta": "Extend Light v0.0.2",
    "aws_textract": "AWS Textract",
    "azure_di": "Azure DI",
    "llamaparse": "LlamaParse",
    "reducto": "Reducto",
    "gemini_3_flash": "Gemini-prs",
    "claude_sonnet_4_6": "Claude-prs",
    "dots_mocr": "dots.mocr",
    "pymupdf": "pymupdf",
}


def load_data() -> tuple[dict, dict]:
    """Returns:
      cells: (doc, parser, agent, path) -> field-score row
      parses: (doc, parser) -> markdown string (smallest-fingerprint pick)
    """
    cells = {}
    parses = {}
    for rd in RUN_DIRS:
        for sf in (REPO / rd).glob("*/*/*/score.json"):
            a, p, d = sf.parent.name, sf.parent.parent.name, sf.parent.parent.parent.name
            try:
                data = json.load(open(sf))
            except Exception:
                continue
            for s in data.get("scores", []):
                cells[(d, p, a, s["path"])] = s
        for pr in (REPO / rd).glob("*/*/parse_result.json"):
            parser, doc = pr.parent.name, pr.parent.parent.name
            if (doc, parser) not in parses:
                try:
                    parses[(doc, parser)] = (json.load(open(pr)).get("markdown") or "")
                except Exception:
                    pass
    return cells, parses


def render_thumb(doc_id: str, page_no: int, dpi: int = 72) -> str | None:
    """Render a PDF page to a base64 PNG (cached on disk)."""
    THUMBS.mkdir(parents=True, exist_ok=True)
    cached = THUMBS / f"{doc_id}_p{page_no}.png"
    if not cached.exists():
        try:
            import pymupdf
            d = pymupdf.open(str(DATA / "sources" / f"{doc_id}.pdf"))
            if page_no < 1 or page_no > len(d):
                d.close()
                return None
            pix = d[page_no - 1].get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72))
            pix.save(str(cached))
            d.close()
        except Exception:
            return None
    return f"data:image/png;base64,{base64.b64encode(cached.read_bytes()).decode()}"


_PAGE_RE = re.compile(r"^##\s+Page\s+(\d+)", re.MULTILINE)


def find_page_in_md(needle: str, md: str) -> tuple[int | None, str | None]:
    """Locate ``needle`` in ``md``; return (page_no, surrounding-context-snippet)."""
    if not needle or not md:
        return None, None
    m = re.search(re.escape(str(needle)), md)
    if not m:
        return None, None
    h = m.start()
    page = None
    for pm in _PAGE_RE.finditer(md):
        if pm.start() <= h:
            page = int(pm.group(1))
        else:
            break
    ctx = md[max(0, h - 100) : h + len(str(needle)) + 100]
    return page, ctx


def score_class(match: str, score: float) -> str:
    if match in {"exact", "null_match"} and score == 1.0:
        return "ok"
    if score >= 0.8:
        return "fuzzy"
    if score > 0.0:
        return "partial"
    return "miss"


def trunc(s: str | None, n: int = 80) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


CSS = """
:root { --ok:#d1f5d3; --fuzzy:#fff5b8; --partial:#ffd6a0; --miss:#ffd5d5; --null:#e8e8e8; }
* { box-sizing: border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,system-ui,sans-serif; margin:0; padding:24px; background:#fafafa; color:#222; }
h1 { font-size:24px; margin:0 0 4px 0; }
h2 { font-size:18px; margin:32px 0 8px; padding:6px 12px; background:#222; color:white; border-radius:6px; }
.meta { color:#888; font-size:13px; }
.controls { position:sticky; top:0; background:#fafafa; padding:12px 0; border-bottom:1px solid #ddd; z-index:50; margin-bottom:16px; }
.controls input, .controls select { padding:6px 10px; font-size:14px; margin-right:8px; border:1px solid #ccc; border-radius:4px; }
table { width:100%; border-collapse:collapse; font-size:12px; table-layout:fixed; }
th, td { border:1px solid #ddd; padding:6px 8px; vertical-align:top; word-wrap:break-word; }
th { background:#222; color:white; font-weight:600; position:sticky; top:60px; }
.field-cell { width:240px; font-family:ui-monospace,monospace; font-size:11px; }
.field-cell .difficulty { display:inline-block; padding:1px 6px; border-radius:3px; font-size:10px; font-family:sans-serif; margin-right:4px; }
.diff-easy { background:#a3e9a4; }
.diff-hard { background:#ffb3b3; }
.diff-mid  { background:#ffeb99; }
.expected-cell { width:200px; background:#f0f4ff; font-weight:500; font-size:11px; }
.parser-cell { font-size:11px; }
.parser-cell.ok       { background:var(--ok); }
.parser-cell.fuzzy    { background:var(--fuzzy); }
.parser-cell.partial  { background:var(--partial); }
.parser-cell.miss     { background:var(--miss); }
.parser-cell.null     { background:var(--null); }
.parser-cell .score { font-size:9px; opacity:0.7; float:right; }
.thumb { width:120px; }
.thumb img { width:100%; border:1px solid #ccc; cursor:zoom-in; }
.legend { margin-top:6px; font-size:11px; }
.legend span { display:inline-block; padding:2px 8px; margin-right:6px; border-radius:3px; }
.legend .ok{background:var(--ok);} .legend .fuzzy{background:var(--fuzzy);} .legend .partial{background:var(--partial);} .legend .miss{background:var(--miss);}
.ctx { color:#666; font-size:10px; font-style:italic; }
details summary { cursor:pointer; color:#0066cc; font-size:11px; }
.hidden { display:none; }
.pdf-link { color:#0066cc; text-decoration:none; font-size:11px; }
img.zoomed { position:fixed; top:50%; left:50%; transform:translate(-50%,-50%); max-width:90vw; max-height:90vh; z-index:1000; box-shadow:0 0 50px rgba(0,0,0,0.5); }
.overlay { position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.7); z-index:999; }
"""

JS = """
function filterRows() {
  const f = document.getElementById('filter').value.toLowerCase();
  const onlyDiff = document.getElementById('onlyDiff').checked;
  document.querySelectorAll('tr.field-row').forEach(tr => {
    let visible = true;
    if (f && !tr.dataset.path.toLowerCase().includes(f) &&
        !(tr.dataset.expected || '').toLowerCase().includes(f)) {
      visible = false;
    }
    if (onlyDiff && tr.dataset.differs !== '1') visible = false;
    tr.classList.toggle('hidden', !visible);
  });
}
function zoomImg(src) {
  const ov = document.createElement('div');
  ov.className = 'overlay';
  ov.onclick = () => ov.remove();
  const img = document.createElement('img');
  img.src = src; img.className = 'zoomed';
  ov.appendChild(img); document.body.appendChild(ov);
}
"""


def build_report() -> None:
    cells, parses = load_data()

    # Collect docs from the manifest
    import yaml
    manifest = yaml.safe_load((DATA / "manifest.yaml").read_text())
    docs = [d["doc_id"] for d in manifest.get("docs", [])]

    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'><title>realdoc-bench Part B — visual comparison</title>")
    parts.append(f"<style>{CSS}</style></head><body>")
    parts.append("<h1>📊 realdoc-bench Part B — visual comparison</h1>")
    parts.append("<div class='meta'>Each row is one schema field. Cells show what each parser's pipeline produced. Color = score: <span class='legend'><span class='ok'>exact</span><span class='fuzzy'>fuzzy</span><span class='partial'>partial</span><span class='miss'>mismatch</span></span></div>")
    parts.append("<div class='controls'>")
    parts.append("<input id='filter' placeholder='Filter by field path or expected value…' oninput='filterRows()' style='width:340px'>")
    parts.append("<label><input type='checkbox' id='onlyDiff' onchange='filterRows()'> Only rows where parsers disagree</label>")
    parts.append("</div>")

    for doc in docs:
        # Collect unique field paths for this doc + the canonical schema row
        fields_seen = {}
        for (d, p, a, path), s in cells.items():
            if d != doc:
                continue
            if path not in fields_seen:
                fields_seen[path] = {"expected": s["expected"], "difficulty": s.get("difficulty"), "required": s.get("required")}

        if not fields_seen:
            continue

        parts.append(f"<h2>📄 {doc}</h2>")
        parts.append("<table>")
        # header row
        parts.append("<tr><th class='field-cell'>Field</th><th class='expected-cell'>Expected</th>")
        for p in PARSER_ORDER:
            parts.append(f"<th class='parser-cell'>{PARSER_LABELS.get(p, p)}</th>")
        parts.append("<th class='thumb'>Source page</th>")
        parts.append("</tr>")

        # Stable field ordering: required first, then easy→hard, then alphabetical
        def _sort_key(item):
            path, meta = item
            d = (meta.get("difficulty") or "zzz")
            return (not meta.get("required", False), d, path)

        for path, meta in sorted(fields_seen.items(), key=_sort_key):
            exp = meta["expected"]
            diff = meta.get("difficulty") or "—"
            diff_cls = {"easy": "diff-easy", "medium": "diff-mid", "hard": "diff-hard"}.get(diff, "")
            exp_attr = html.escape(str(exp) if exp is not None else "")
            row_parts = [f"<tr class='field-row' data-path='{html.escape(path)}' data-expected='{exp_attr}'"]
            # Compute parser-disagree flag
            actuals_for_diff = set()
            for p in PARSER_ORDER:
                for a in ("claude_sonnet_4_6", "gemini_3_flash"):
                    s = cells.get((doc, p, a, path))
                    if s:
                        actuals_for_diff.add(json.dumps(s.get("actual"), default=str, sort_keys=True))
                        break
            differs = "1" if len(actuals_for_diff) > 1 else "0"
            row_parts.append(f" data-differs='{differs}'>")
            row_parts.append(
                f"<td class='field-cell'><span class='difficulty {diff_cls}'>{diff}</span>{html.escape(path)}</td>"
            )
            row_parts.append(f"<td class='expected-cell'>{html.escape(trunc(exp, 120))}</td>")
            # parser cells (averaging over agents → pick best one for display)
            for p in PARSER_ORDER:
                best = None
                best_score = -1
                for a in ("claude_sonnet_4_6", "gemini_3_flash"):
                    s = cells.get((doc, p, a, path))
                    if s and s["score"] > best_score:
                        best, best_score = s, s["score"]
                if best is None:
                    row_parts.append("<td class='parser-cell null'><span class='score'>—</span></td>")
                else:
                    cls = score_class(best["match"], best["score"])
                    actual = trunc(best["actual"], 90)
                    sc = f"{best['score']:.2f}" if best["score"] != 1.0 else "1.0"
                    actual_tooltip = html.escape(str(best["actual"]) if best["actual"] is not None else "")[:300]
                    row_parts.append(
                        f"<td class='parser-cell {cls}' title='{actual_tooltip}'>"
                        f"<span class='score'>{sc}</span>{html.escape(actual)}</td>"
                    )

            # Thumbnail of source page where pymupdf finds the expected value
            page, _ = find_page_in_md(str(exp) if exp is not None else "", parses.get((doc, "pymupdf"), ""))
            if page is None and isinstance(exp, str):
                # Try a shorter substring
                page, _ = find_page_in_md(exp[:30], parses.get((doc, "pymupdf"), ""))
            thumb_cell = "<td class='thumb'>"
            if page:
                src = render_thumb(doc, page, dpi=72)
                if src:
                    thumb_cell += f"<img src='{src}' onclick='zoomImg(this.src)' title='page {page} (click to zoom)'>"
                thumb_cell += f"<br><a class='pdf-link' href='file://{DATA / 'sources' / f'{doc}.pdf'}'>open PDF</a> · p{page}"
            else:
                thumb_cell += f"<a class='pdf-link' href='file://{DATA / 'sources' / f'{doc}.pdf'}'>open PDF</a>"
            thumb_cell += "</td>"
            row_parts.append(thumb_cell)

            row_parts.append("</tr>")
            parts.append("".join(row_parts))

        parts.append("</table>")

    parts.append(f"<script>{JS}</script></body></html>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(parts))
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    build_report()
