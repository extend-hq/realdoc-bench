#!/usr/bin/env python3
"""Build an HTML viewer for per-doc extraction comparison.

Outputs:
  datasets/extraction/v1.1/_viewer/extractions/index.html         — list of 101 docs
  datasets/extraction/v1.1/_viewer/extractions/<safe_id>.html     — per-doc page

Each per-doc page shows:
  - PDF embedded inline (iframe)
  - Schema title + field count
  - Big table: rows = schema fields, cols = parsers, cells = extracted values
    color-coded: green = matches consensus, red = differs, gray = null
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
V1_1 = REPO / "datasets" / "extraction" / "v1.1"
SCHEMAS_DIR = V1_1 / "_schemas"
VIEWER = V1_1 / "_viewer"
EXT_DIR = VIEWER / "extractions"

PARSER_ORDER = [
    "extend_performance_v2_0_0",
    "reducto",
    "azure_di",
    "aws_textract",
    "llamaparse",
    "pymupdf",
]
PARSER_LABEL = {
    "extend_performance_v2_0_0": "extend v2",
    "reducto": "reducto",
    "azure_di": "azure DI",
    "aws_textract": "AWS textract",
    "llamaparse": "llamaparse",
    "pymupdf": "pymupdf",
}


def normalize_value(v):
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    if isinstance(v, str):
        s = v.strip().lower()
        s = " ".join(s.split())
        return s if s else None
    if isinstance(v, list):
        return ("list", len(v))
    if isinstance(v, dict):
        return ("dict", len(v))
    return str(v)[:200]


def format_cell_display(v) -> str:
    """Display-friendly version of a value, truncated."""
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return "✓ true" if v else "✗ false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        s = v.strip()
        return s[:120] + ("…" if len(s) > 120 else "")
    if isinstance(v, list):
        return f"[{len(v)} items]"
    if isinstance(v, dict):
        return f"{{{len(v)} keys}}"
    return str(v)[:120]


def _short(s: str, n: int = 40) -> str:
    s = str(s) if s is not None else ""
    s = s.strip()
    return s[:n] + ("…" if len(s) > n else "")


def format_cell_html(v, max_rows: int = 3, max_cols: int = 6) -> str:
    """Like format_cell_display but returns sanitized HTML — arrays of dicts
    render as mini-tables so table fields are visible in the viewer."""
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return "✓ true" if v else "✗ false"
    if isinstance(v, (int, float)):
        return html.escape(str(v))
    if isinstance(v, str):
        s = v.strip()
        return html.escape(s[:120] + ("…" if len(s) > 120 else ""))

    if isinstance(v, list):
        if not v:
            return "<i>[]</i>"
        # List of dicts → mini-table
        if all(isinstance(item, dict) for item in v):
            # Union of keys, ordered by first-appearance
            cols: list[str] = []
            seen: set = set()
            for item in v:
                for k in item.keys():
                    if k not in seen:
                        seen.add(k); cols.append(k)
            if len(cols) > max_cols:
                cols = cols[:max_cols]
            shown_rows = v[:max_rows]
            parts = [f"<div class='nested-table-wrap'><table class='nested-table'><thead><tr>"]
            for c in cols:
                parts.append(f"<th>{html.escape(_short(c, 16))}</th>")
            parts.append("</tr></thead><tbody>")
            for item in shown_rows:
                parts.append("<tr>")
                for c in cols:
                    cell = item.get(c, "")
                    parts.append(f"<td>{html.escape(_short(cell, 24))}</td>")
                parts.append("</tr>")
            parts.append("</tbody></table>")
            if len(v) > max_rows:
                parts.append(f"<div class='nested-more'>+{len(v) - max_rows} more rows</div>")
            parts.append("</div>")
            return "".join(parts)
        # List of primitives → " · " joined, truncated
        prims = [_short(x, 20) for x in v[:6]]
        suffix = f" · +{len(v) - 6}" if len(v) > 6 else ""
        return html.escape(" · ".join(prims) + suffix)

    if isinstance(v, dict):
        if not v:
            return "<i>{}</i>"
        parts = ["<div class='nested-kv'>"]
        for i, (k, val) in enumerate(list(v.items())[:5]):
            parts.append(f"<div><b>{html.escape(_short(k, 16))}</b>: "
                         f"{html.escape(_short(val, 24))}</div>")
        if len(v) > 5:
            parts.append(f"<div class='nested-more'>+{len(v) - 5} more keys</div>")
        parts.append("</div>")
        return "".join(parts)

    return html.escape(str(v)[:120])


def safe_id(domain: str, type_name: str, fname: str) -> str:
    """File-safe id from doc coordinates."""
    raw = f"{domain}__{type_name}__{fname.replace('.pdf', '')}"
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw)


def load_doc_extractions(sweep_dir: Path, parsers: list[str]) -> dict:
    """Returns {(domain, type, filename) → {parser → extracted_dict}}"""
    out: dict = {}
    for parser_dir in sorted(sweep_dir.iterdir()):
        if not (parser_dir.is_dir() or parser_dir.is_symlink()):
            continue
        parser = parser_dir.name
        if parser not in parsers:
            continue
        ext_root = parser_dir / "_extractions"
        if not ext_root.exists():
            continue
        for jpath in ext_root.rglob("*.json"):
            if jpath.name.endswith(".meta.json"):
                continue
            rel = jpath.relative_to(ext_root)
            if len(rel.parts) < 3:
                continue
            domain, typ, fname = rel.parts[0], rel.parts[1], rel.parts[2].replace(".json", ".pdf")
            try:
                blob = json.loads(jpath.read_text())
            except Exception:
                continue
            if not isinstance(blob, dict):
                continue
            out.setdefault((domain, typ, fname), {})[parser] = blob
    return out


def load_parsed_mds(sweep_dir: Path, domain: str, type_name: str, fname: str,
                    parsers: list[str]) -> dict[str, str]:
    """Return {parser → raw markdown string} for one doc."""
    md_name = fname.replace(".pdf", ".md")
    out: dict[str, str] = {}
    for p in parsers:
        parser_dir = sweep_dir / p
        if not parser_dir.exists():
            continue
        real = parser_dir.resolve()  # follow symlinks
        md = real / domain / type_name / md_name
        if md.exists():
            try:
                out[p] = md.read_text()
            except Exception:
                continue
    return out


# Pattern for markdown table blocks: 2+ consecutive lines containing "|"
TABLE_BLOCK_RE = re.compile(
    r"((?:^[ \t]*\|[^\n]*\n){2,})",
    re.MULTILINE,
)
HEADING_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


def md_metrics(md: str) -> dict:
    """Quick stats over a parsed markdown: char count, line count, heading count,
    table-block count, table-row count, code-block count, image count."""
    if not md:
        return {"chars": 0, "lines": 0, "headings": 0, "tables": 0, "table_rows": 0,
                "code_blocks": 0, "images": 0}
    tables = TABLE_BLOCK_RE.findall(md)
    table_rows = sum(t.count("\n") for t in tables)
    return {
        "chars": len(md),
        "lines": md.count("\n") + 1,
        "headings": len(HEADING_RE.findall(md)),
        "tables": len(tables),
        "table_rows": table_rows,
        "code_blocks": md.count("```") // 2,
        "images": md.count("!["),
    }


def extract_table_blocks(md: str, max_blocks: int | None = None,
                         max_lines_per_block: int | None = None) -> list[str]:
    """Pull markdown table blocks for preview.

    max_blocks=None → return all blocks.
    max_lines_per_block=None → keep all rows."""
    if not md:
        return []
    blocks = TABLE_BLOCK_RE.findall(md)
    if max_blocks is not None:
        blocks = blocks[:max_blocks]
    out = []
    for b in blocks:
        lines = b.strip().split("\n")
        if max_lines_per_block is not None and len(lines) > max_lines_per_block:
            lines = lines[:max_lines_per_block] + [f"  …({len(lines) - max_lines_per_block} more rows)"]
        out.append("\n".join(lines))
    return out


def load_schema(type_name: str) -> dict | None:
    p = SCHEMAS_DIR / f"{type_name}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def field_categories(schema: dict) -> dict[str, str]:
    """Return {field_name → category ∈ {checkbox, table, key_value, other}}."""
    cats: dict[str, str] = {}
    for name, v in (schema.get("properties") or {}).items():
        if not isinstance(v, dict):
            cats[name] = "other"
            continue
        caps = v.get("x-capabilities") or []
        if "checkbox" in caps:
            cats[name] = "checkbox"
        elif "table_structured" in caps or v.get("type") == "array":
            cats[name] = "table"
        elif "form_region" in caps or "key_value" in caps:
            cats[name] = "key_value"
        else:
            cats[name] = "other"
    return cats


CSS = """
<style>
  :root {
    --bg:#0f1419; --card:#171c23; --border:#2a323d; --fg:#d6dde6;
    --muted:#7d8a99; --accent:#7da2ff;
    --agree:#1a3a1a; --agree-text:#a8e5a8;
    --disagree:#3a1a1a; --disagree-text:#e5a8a8;
    --null:#1a1d22; --null-text:#5a6470;
    --tag-cb:#3d2a5a; --tag-tb:#2a3d5a; --tag-kv:#3d3a2a; --tag-other:#2a2a2a;
    --accepted-row:#1c2d1c; --rejected-row:#2d1c1c; --picked-cell:#2d3d6a;
  }
  html, body { background:var(--bg); color:var(--fg); margin:0;
               font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
               font-size:13px; }
  header { padding:18px 28px; border-bottom:1px solid var(--border);
           background:var(--bg); position:sticky; top:0; z-index:10; }
  header h1 { margin:0 0 4px 0; font-size:18px; }
  header .meta { color:var(--muted); font-size:12px; }
  header a { color:var(--accent); text-decoration:none; }
  header a:hover { text-decoration:underline; }
  main { padding:18px 28px 40px; }

  /* Index page */
  .domain-block { margin-bottom:32px; }
  .domain-block h2 { font-size:15px; margin:24px 0 8px;
                     padding-top:18px; border-top:1px solid var(--border); }
  .doc-list { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr));
              gap:8px; }
  .doc-item { background:var(--card); border:1px solid var(--border);
              border-radius:5px; padding:8px 10px; transition:border-color .12s; }
  .doc-item:hover { border-color:var(--accent); }
  .doc-item a { color:inherit; text-decoration:none; display:block;
                font-family:ui-monospace,SFMono-Regular,monospace; font-size:11px;
                word-break:break-all; line-height:1.3; }
  .doc-item .type { color:var(--muted); font-size:10px; text-transform:uppercase;
                    letter-spacing:.04em; margin-bottom:2px; }

  /* Detail page */
  .doc-detail { --pdf-width: 480px;
                display:grid; grid-template-columns: var(--pdf-width) 8px 1fr;
                gap:0; height:calc(100vh - 130px); }
  .pdf-pane { background:var(--card); border:1px solid var(--border);
              border-radius:6px; overflow:hidden; position:sticky; top:120px;
              max-height:calc(100vh - 140px); }
  .pdf-pane iframe { width:100%; height:100%; border:0; background:#fff; }
  .ext-pane { overflow:auto; padding-left:14px; }
  .pane-divider { background:transparent; cursor:col-resize; user-select:none;
                   position:relative; z-index:6; }
  .pane-divider::after { content:""; position:absolute; left:3px; top:0;
                          width:2px; height:100%; background:var(--border);
                          transition:background .12s; }
  .pane-divider:hover::after, .pane-divider.dragging::after {
                          background:var(--accent); }
  body.col-resizing { cursor:col-resize !important; user-select:none; }
  body.col-resizing iframe { pointer-events:none; }

  table.ext { width:100%; border-collapse:collapse; font-size:11px; }
  table.ext th, table.ext td {
    padding:5px 7px; border:1px solid var(--border);
    vertical-align:top; text-align:left;
  }
  table.ext thead th {
    background:var(--card); color:var(--fg); font-weight:600;
    position:sticky; top:0; z-index:2; white-space:nowrap;
  }
  table.ext tbody td.field-name {
    font-family:ui-monospace,SFMono-Regular,monospace; font-size:10px;
    color:#c9d1da; background:var(--card); white-space:nowrap; max-width:240px;
    overflow:hidden; text-overflow:ellipsis;
  }
  table.ext tbody td.cat {
    text-align:center; font-size:9px; text-transform:uppercase; letter-spacing:.04em;
  }
  table.ext .cat-checkbox { background:var(--tag-cb); color:#d5c5ff; }
  table.ext .cat-table { background:var(--tag-tb); color:#c5d5ff; }
  table.ext .cat-key_value { background:var(--tag-kv); color:#ddd5c5; }
  table.ext .cat-other { background:var(--tag-other); color:var(--muted); }

  td.cell-agree { background:var(--agree); color:var(--agree-text); }
  td.cell-disagree { background:var(--disagree); color:var(--disagree-text); }
  td.cell-null { background:var(--null); color:var(--null-text); font-style:italic; }
  td.consensus-col { background:#1a2030; font-weight:600; }

  td.cell-agree, td.cell-disagree, td.cell-null, td.consensus-col {
    max-width:220px; word-wrap:break-word;
  }
  td.cell-agree[title], td.cell-disagree[title] { cursor:help; }

  .legend { font-size:11px; color:var(--muted); padding:8px 0 12px; }
  .legend span { display:inline-block; padding:2px 8px; border-radius:3px; margin-right:8px; }
  .legend .agree { background:var(--agree); color:var(--agree-text); }
  .legend .disagree { background:var(--disagree); color:var(--disagree-text); }
  .legend .null { background:var(--null); color:var(--null-text); font-style:italic; }

  .row-stats { color:var(--muted); font-size:10px; padding:0 5px; }
  .nav-links { font-size:12px; margin-bottom:12px; }
  .nav-links a { color:var(--accent); text-decoration:none; margin-right:12px; }
  .nav-links a:hover { text-decoration:underline; }
  .schema-info { background:var(--card); border:1px solid var(--border);
                 border-radius:6px; padding:10px 14px; margin-bottom:14px;
                 font-size:11px; }
  .schema-info strong { color:var(--accent); }

  /* Toolbar */
  .toolbar { display:flex; align-items:center; gap:14px; flex-wrap:wrap;
             background:var(--card); border:1px solid var(--border);
             border-radius:6px; padding:8px 12px; margin-bottom:12px;
             font-size:11px; position:sticky; top:120px; z-index:5; }
  .toolbar label { color:var(--muted); cursor:pointer; user-select:none; }
  .toolbar label input { vertical-align:middle; margin-right:4px; }
  .toolbar button { background:#2a3a5a; color:#cfe0ff; border:1px solid #3a4a6a;
                    padding:4px 10px; border-radius:4px; cursor:pointer;
                    font-size:11px; font-family:inherit; }
  .toolbar button:hover { background:#3a4a6a; }
  .toolbar .stats { margin-left:auto; color:var(--muted); }
  .toolbar .stats span { margin-left:10px; }

  /* Decision controls */
  td.actions { white-space:nowrap; text-align:center; padding:2px 4px; }
  td.actions button { background:var(--card); border:1px solid var(--border);
                      color:var(--muted); cursor:pointer; padding:2px 6px;
                      font-size:10px; border-radius:3px; margin:0 1px;
                      font-family:inherit; }
  td.actions button:hover { color:var(--fg); border-color:var(--accent); }
  td.actions button.btn-accept.active { background:#2a4a2a; color:#a8e5a8;
                                         border-color:#3a6a3a; }
  td.actions button.btn-reject.active { background:#4a2a2a; color:#e5a8a8;
                                         border-color:#6a3a3a; }

  /* Clickable parser cells (pick this as truth) */
  table.ext tbody td.cell-agree, table.ext tbody td.cell-disagree {
    cursor:pointer; transition:outline .1s;
  }
  table.ext tbody td.cell-agree:hover, table.ext tbody td.cell-disagree:hover {
    outline:2px solid var(--accent);
  }
  table.ext tbody td.picked { background:var(--picked-cell) !important;
                              color:#cfe0ff !important; font-weight:600;
                              outline:2px solid var(--accent); }
  table.ext tbody td.consensus-col { cursor:pointer; }
  table.ext tbody td.consensus-col:hover { outline:2px solid var(--accent); }

  /* Row state */
  tr.accepted td.field-name, tr.accepted td.cat { background:var(--accepted-row) !important; }
  tr.rejected td.field-name, tr.rejected td.cat { background:var(--rejected-row) !important; }
  tr.hidden-allnull { display:none; }

  /* Decision counter */
  .decision-summary { display:inline-block; padding:1px 6px; border-radius:3px;
                      font-size:10px; margin-left:6px; }
  .ds-accepted { background:#2a4a2a; color:#a8e5a8; }
  .ds-rejected { background:#4a2a2a; color:#e5a8a8; }
  .ds-pending { background:#3a3a3a; color:#999; }

  /* Markdown / tables panel (for non-form / prose docs) */
  .md-section { margin-top:28px; }
  .md-section > h2 { font-size:14px; margin:0 0 10px; padding-top:18px;
                     border-top:1px solid var(--border); }
  .md-section > .sub { color:var(--muted); font-size:11px; margin-bottom:12px; }
  .md-tabs { display:flex; gap:2px; margin-bottom:10px; flex-wrap:wrap; }
  .md-tabs button { background:var(--card); border:1px solid var(--border);
                    color:var(--muted); padding:4px 10px; border-radius:4px 4px 0 0;
                    cursor:pointer; font-family:inherit; font-size:11px; }
  .md-tabs button.active { background:#1f2530; color:var(--accent);
                            border-color:var(--accent); border-bottom-color:#1f2530; }
  .md-tabs button:hover { color:var(--fg); }
  .md-stats-bar { display:grid; grid-template-columns:repeat(auto-fit, minmax(140px, 1fr));
                  gap:6px; margin-bottom:14px; }
  .md-stat { background:var(--card); border:1px solid var(--border); border-radius:5px;
             padding:6px 10px; font-size:10px; }
  .md-stat .name { color:var(--muted); }
  .md-stat .vals { font-family:ui-monospace,monospace; font-size:10px; margin-top:3px; }
  .md-stat .vals span { display:inline-block; min-width:50px; margin-right:6px; }

  .md-pane { background:#0c1014; border:1px solid var(--border); border-radius:6px;
             padding:12px 16px; max-height:80vh; overflow:auto; }
  .md-pane pre { white-space:pre-wrap; word-wrap:break-word;
                 font-family:ui-monospace,SFMono-Regular,monospace; font-size:11px;
                 color:#c9d1da; margin:0; line-height:1.45; }
  .md-pane.hidden { display:none; }
  .md-tables-list { display:flex; flex-direction:column; gap:14px; }
  .md-tables-list .table-block { background:var(--card); border:1px solid var(--border);
                                   border-radius:5px; padding:10px 14px;
                                   font-family:ui-monospace,monospace; font-size:11px;
                                   white-space:pre; overflow-x:auto; color:#c9d1da; }
  .md-tables-list .none { color:var(--muted); font-style:italic; padding:14px;
                          background:var(--card); border-radius:5px;
                          border:1px dashed var(--border); }

  /* Nested mini-tables inside extraction cells */
  .nested-table-wrap { max-width:100%; overflow-x:auto; }
  table.nested-table { font-size:10px; border-collapse:collapse; margin:0;
                        font-family:ui-monospace,SFMono-Regular,monospace; }
  table.nested-table th { background:rgba(125,162,255,.12); color:#a5b8d8;
                           padding:1px 5px; border:1px solid rgba(125,162,255,.18);
                           font-weight:600; text-align:left; white-space:nowrap; }
  table.nested-table td { padding:1px 5px; border:1px solid rgba(125,162,255,.10);
                           white-space:nowrap; max-width:140px; overflow:hidden;
                           text-overflow:ellipsis; }
  .nested-more { color:var(--muted); font-style:italic; font-size:9px;
                  padding:2px 0; }
  .nested-kv { font-size:10px; font-family:ui-monospace,SFMono-Regular,monospace;
                line-height:1.4; }
  .nested-kv b { color:#a5b8d8; font-weight:600; }
  /* Table cells holding nested-tables should grow wider */
  table.ext tbody td:has(.nested-table) { max-width:none; min-width:240px;
                                            vertical-align:top; }
  table.ext tbody td:has(.nested-table) { padding:4px; }
</style>
"""


DETAIL_JS = """
// Per-doc decisions:
//   decisions[field] = { decision: 'accept'|'reject', source: 'consensus'|'parser:<name>', value: <raw>, ts: ms }
let decisions = {};
try { decisions = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); } catch (e) { decisions = {}; }

function saveDecisions() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(decisions));
}

function pickRowForField(field) {
  return document.querySelector(`tr[data-field="${CSS.escape(field)}"]`);
}

function refreshRow(field) {
  const row = pickRowForField(field);
  if (!row) return;
  // Reset state classes
  row.classList.remove('accepted', 'rejected');
  // Remove any picked-cell marks
  row.querySelectorAll('td.picked').forEach(c => c.classList.remove('picked'));
  row.querySelectorAll('button.active').forEach(b => b.classList.remove('active'));

  const d = decisions[field];
  if (!d) return;
  if (d.decision === 'reject') {
    row.classList.add('rejected');
    row.querySelector('button.btn-reject').classList.add('active');
    return;
  }
  // accept
  row.classList.add('accepted');
  row.querySelector('button.btn-accept').classList.add('active');
  if (d.source === 'consensus') {
    row.querySelector('td.consensus-col')?.classList.add('picked');
  } else if (d.source && d.source.startsWith('parser:')) {
    const p = d.source.slice('parser:'.length);
    const cell = row.querySelector(`td[data-parser="${CSS.escape(p)}"]`);
    if (cell) cell.classList.add('picked');
  }
}

function refreshStats() {
  const allRows = document.querySelectorAll('tr.data-row');
  const total = allRows.length;
  let allNullHidden = 0;
  const hideAllNull = document.getElementById('toggle-allnull').checked;
  allRows.forEach(r => {
    const isAllNull = r.dataset.allnull === '1';
    if (hideAllNull && isAllNull && !decisions[r.dataset.field]) {
      r.classList.add('hidden-allnull');
      allNullHidden++;
    } else {
      r.classList.remove('hidden-allnull');
    }
  });
  const acc = Object.values(decisions).filter(d => d.decision === 'accept').length;
  const rej = Object.values(decisions).filter(d => d.decision === 'reject').length;
  const visible = total - allNullHidden;
  const pen = visible - acc - rej;
  document.getElementById('stat-total').textContent = `${visible} visible / ${total} total`;
  document.getElementById('stat-acc').textContent = acc;
  document.getElementById('stat-rej').textContent = rej;
  document.getElementById('stat-pen').textContent = Math.max(0, pen);
  document.getElementById('stat-hidden').textContent = allNullHidden;
}

// Handle click on a parser/consensus cell → mark that as picked + auto-accept
document.querySelectorAll('table.ext tbody td[data-parser]').forEach(cell => {
  cell.addEventListener('click', () => {
    const row = cell.closest('tr');
    const field = row.dataset.field;
    const parser = cell.dataset.parser;
    const isConsensus = (parser === '__consensus__');
    const value = isConsensus ? CONSENSUS[field] : (FIELD_VALUES[field] || {})[parser];
    decisions[field] = {
      decision: 'accept',
      source: isConsensus ? 'consensus' : `parser:${parser}`,
      value: value,
      ts: Date.now(),
    };
    saveDecisions();
    refreshRow(field);
    refreshStats();
  });
});

// Accept button (no specific pick) — accept consensus if present, else mark accepted with first non-null value
document.querySelectorAll('button.btn-accept').forEach(btn => {
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const row = btn.closest('tr');
    const field = row.dataset.field;
    const cons = CONSENSUS[field];
    if (cons !== null && cons !== undefined) {
      decisions[field] = { decision: 'accept', source: 'consensus', value: cons, ts: Date.now() };
    } else {
      // fall back to first non-null parser value
      const pvs = FIELD_VALUES[field] || {};
      let chosen = null, chosenParser = null;
      for (const [p, v] of Object.entries(pvs)) {
        if (v !== null && v !== undefined && v !== '' && !(Array.isArray(v) && v.length === 0)) {
          chosen = v; chosenParser = p; break;
        }
      }
      decisions[field] = {
        decision: 'accept',
        source: chosenParser ? `parser:${chosenParser}` : 'none',
        value: chosen, ts: Date.now(),
      };
    }
    saveDecisions();
    refreshRow(field);
    refreshStats();
  });
});

// Reject all values for this field
document.querySelectorAll('button.btn-reject').forEach(btn => {
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const row = btn.closest('tr');
    const field = row.dataset.field;
    decisions[field] = { decision: 'reject', source: 'none', value: null, ts: Date.now() };
    saveDecisions();
    refreshRow(field);
    refreshStats();
  });
});

// Toggle null-row visibility
document.getElementById('toggle-allnull').addEventListener('change', refreshStats);

// Clear decisions
document.getElementById('btn-clear').addEventListener('click', () => {
  if (!confirm('Clear all decisions for THIS doc?')) return;
  decisions = {};
  saveDecisions();
  document.querySelectorAll('tr.data-row').forEach(r => refreshRow(r.dataset.field));
  refreshStats();
});

// Export decisions as JSON download
document.getElementById('btn-export').addEventListener('click', () => {
  const payload = {
    doc: DOC,
    exported_at: new Date().toISOString(),
    decisions: decisions,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `decisions_${DOC.type}_${DOC.filename.replace('.pdf', '')}.json`;
  a.click();
});

// Initial render
document.querySelectorAll('tr.data-row').forEach(r => refreshRow(r.dataset.field));
refreshStats();

// Tab switching for markdown panes
document.querySelectorAll('.md-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.md-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const paneId = btn.dataset.pane;
    document.querySelectorAll('.md-pane').forEach(p => p.classList.add('hidden'));
    document.getElementById(paneId)?.classList.remove('hidden');
  });
});

// Resizable PDF pane (drag the vertical divider).
// Width persists via localStorage across docs.
(function() {
  const PDF_WIDTH_KEY = 'realdoc_v11_pdf_width';
  const root = document.querySelector('.doc-detail');
  const divider = document.querySelector('.pane-divider');
  if (!root || !divider) return;
  // Restore saved width
  const saved = parseInt(localStorage.getItem(PDF_WIDTH_KEY) || '', 10);
  if (saved && saved > 200) {
    root.style.setProperty('--pdf-width', saved + 'px');
  }
  let dragging = false;
  divider.addEventListener('mousedown', (e) => {
    dragging = true;
    divider.classList.add('dragging');
    document.body.classList.add('col-resizing');
    e.preventDefault();
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    divider.classList.remove('dragging');
    document.body.classList.remove('col-resizing');
  });
  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const rect = root.getBoundingClientRect();
    const w = Math.max(220, Math.min(rect.width - 240, e.clientX - rect.left));
    root.style.setProperty('--pdf-width', w + 'px');
    localStorage.setItem(PDF_WIDTH_KEY, String(Math.round(w)));
  });
  // Double-click divider → reset to default
  divider.addEventListener('dblclick', () => {
    root.style.setProperty('--pdf-width', '480px');
    localStorage.setItem(PDF_WIDTH_KEY, '480');
  });
})();
"""


def compute_doc_disagreement(data: dict, parsers: list[str]) -> dict:
    """For each doc, count fields where ≥3 parsers extracted but ≥3 unique values.
    Returns {(domain, type, fname) → {'hot': N, 'total': T, 'cons': C}}"""
    out: dict = {}
    for key, parser_outputs in data.items():
        all_fields: set = set()
        for p in parsers:
            if p in parser_outputs and isinstance(parser_outputs[p], dict):
                all_fields.update(parser_outputs[p].keys())
        hot = 0; total = 0; cons = 0
        for fld in all_fields:
            total += 1
            norm_vals: list = []
            for p in parsers:
                if p not in parser_outputs:
                    continue
                v = parser_outputs[p].get(fld)
                nv = normalize_value(v)
                if nv is not None:
                    norm_vals.append(nv)
            if not norm_vals:
                continue
            unique = set(norm_vals)
            n_parsers_w_val = len(norm_vals)
            tally = Counter(norm_vals)
            top, top_c = tally.most_common(1)[0]
            has_consensus = top_c * 2 > n_parsers_w_val
            if has_consensus:
                cons += 1
            # Hotspot: ≥3 parsers extracted AND ≥3 unique normalized values
            if n_parsers_w_val >= 3 and len(unique) >= 3:
                hot += 1
        out[key] = {"hot": hot, "total": total, "cons": cons,
                    "fillable": total - 0}  # placeholder
    return out


def build_index(docs_by_domain: dict, doc_to_safe_id: dict, parsers: list[str],
                disagreement_by_doc: dict | None = None) -> str:
    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             "<title>v1.1 extractions index</title>", CSS, "</head><body>"]
    n_docs = sum(len(by_type) for by_type in docs_by_domain.values()
                 for by_type in by_type.values()) if False else (
        sum(len(docs) for by_type in docs_by_domain.values() for docs in by_type.values()))
    parts.append("<header>")
    parts.append("<h1>v1.1 extractions — by doc</h1>")
    parts.append(f"<div class='meta'>{n_docs} docs · {len(parsers)} parsers · "
                 f"sorted by parser disagreement (highest first) · "
                 f"<a href='../index.html'>← back to source viewer</a></div>")
    parts.append("</header><main>")
    disagreement_by_doc = disagreement_by_doc or {}
    for domain, by_type in docs_by_domain.items():
        d_docs = sum(len(v) for v in by_type.values())
        # Domain total hotspots
        d_hot = sum(disagreement_by_doc.get((domain, t, f), {}).get("hot", 0)
                    for t, fnames in by_type.items() for f in fnames)
        parts.append(f"<div class='domain-block'>")
        parts.append(f"<h2>{html.escape(domain.replace('_', ' '))} "
                     f"<span style='color:var(--muted);font-weight:normal;font-size:12px;'>"
                     f"· {d_docs} docs · {len(by_type)} types · "
                     f"<span style='color:#e5a8a8;'>{d_hot} hotspot fields</span></span></h2>")
        # Sort type sections by total hotspots desc
        type_items = sorted(
            by_type.items(),
            key=lambda kv: -sum(disagreement_by_doc.get((domain, kv[0], f), {}).get("hot", 0) for f in kv[1])
        )
        for type_name, docs in type_items:
            t_hot = sum(disagreement_by_doc.get((domain, type_name, f), {}).get("hot", 0) for f in docs)
            parts.append(f"<div style='margin:14px 0 18px;'>")
            parts.append(f"<div style='font-size:12px;color:#9aa6b3;font-weight:600;margin-bottom:6px;font-family:ui-monospace,monospace;'>"
                         f"{html.escape(type_name)} "
                         f"<span style='color:#e5a8a8;font-weight:normal;'>· {t_hot} hotspots</span></div>")
            parts.append("<div class='doc-list'>")
            # Sort docs by hotspot count desc, then by name
            sorted_docs = sorted(
                docs,
                key=lambda f: (-disagreement_by_doc.get((domain, type_name, f), {}).get("hot", 0), f),
            )
            for fname in sorted_docs:
                sid = doc_to_safe_id[(domain, type_name, fname)]
                stats = disagreement_by_doc.get((domain, type_name, fname), {})
                hot = stats.get("hot", 0)
                cons = stats.get("cons", 0)
                total = stats.get("total", 0)
                badge_color = "#e5a8a8" if hot >= 10 else ("#e5d8a8" if hot >= 3 else "#7d8a99")
                badge = (f"<span style='float:right;color:{badge_color};font-size:10px;font-weight:600;'>"
                         f"🔥{hot}</span>") if hot > 0 else ""
                stats_line = (f"<div style='color:var(--muted);font-size:9px;margin-top:2px;'>"
                              f"{cons}/{total} consensus · {hot} hotspots</div>") if total else ""
                parts.append(f"<div class='doc-item'><a href='{html.escape(sid)}.html'>"
                             f"<div class='type'>{html.escape(type_name)}{badge}</div>"
                             f"{html.escape(fname)}{stats_line}</a></div>")
            parts.append("</div></div>")
        parts.append("</div>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


def build_detail(domain: str, type_name: str, fname: str,
                 schema: dict | None, extractions: dict,
                 parsers: list[str], doc_prev_id: str | None, doc_next_id: str | None,
                 parsed_mds: dict[str, str] | None = None,
                 source_page_count: int = 0) -> str:
    parsed_mds = parsed_mds or {}
    # Truncate ONLY if the parsed md is huge (>500KB), regardless of source pages.
    # The viewer scroll handles long content fine; only protect against absurdly large dumps.
    preview_limit = 500_000  # chars
    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             f"<title>{html.escape(fname)}</title>", CSS, "</head><body>"]
    parts.append("<header>")
    parts.append(f"<h1>{html.escape(fname)}</h1>")
    parts.append(f"<div class='meta'>{html.escape(domain.replace('_', ' '))} · "
                 f"<code>{html.escape(type_name)}</code> · ")
    if schema:
        n_fields = len(schema.get("properties") or {})
        parts.append(f"schema <code>{html.escape(type_name)}.json</code> · {n_fields} fields")
    parts.append("</div></header><main>")
    # Nav
    nav_bits = ['<a href="index.html">← all docs</a>']
    if doc_prev_id:
        nav_bits.append(f'<a href="{html.escape(doc_prev_id)}.html">← prev doc</a>')
    if doc_next_id:
        nav_bits.append(f'<a href="{html.escape(doc_next_id)}.html">next doc →</a>')
    parts.append(f"<div class='nav-links'>{' '.join(nav_bits)}</div>")

    # Schema info card
    if schema:
        props = schema.get("properties") or {}
        cats = field_categories(schema)
        cat_counts = Counter(cats.values())
        parts.append("<div class='schema-info'>")
        parts.append(f"<strong>{html.escape(schema.get('title', type_name))}</strong> · "
                     f"{len(props)} fields · "
                     f"{cat_counts.get('checkbox', 0)} checkboxes · "
                     f"{cat_counts.get('table', 0)} tables · "
                     f"{cat_counts.get('key_value', 0)} key-values · "
                     f"{cat_counts.get('other', 0)} other")
        parts.append("</div>")

    # Detail layout
    parts.append("<div class='doc-detail'>")

    # PDF pane + draggable divider
    pdf_rel = f"../../{domain}/{type_name}/sources/{fname}"
    parts.append(f"<div class='pdf-pane'><iframe src='{html.escape(pdf_rel)}'></iframe></div>")
    parts.append("<div class='pane-divider' title='drag to resize PDF pane'></div>")

    # Extractions table
    parts.append("<div class='ext-pane'>")
    # Toolbar
    parts.append("<div class='toolbar'>")
    parts.append("<label><input type='checkbox' id='toggle-allnull' checked> Hide fields where all parsers returned null</label>")
    parts.append("<button id='btn-clear'>Clear decisions</button>")
    parts.append("<button id='btn-export'>Export decisions JSON</button>")
    parts.append("<div class='stats'>"
                 "<span id='stat-total'>0 fields</span>"
                 "<span><span class='decision-summary ds-accepted'>✓ <span id='stat-acc'>0</span></span></span>"
                 "<span><span class='decision-summary ds-rejected'>✗ <span id='stat-rej'>0</span></span></span>"
                 "<span><span class='decision-summary ds-pending'><span id='stat-pen'>0</span> pending</span></span>"
                 "<span><span id='stat-hidden'>0</span> all-null hidden</span>"
                 "</div></div>")
    parts.append("<div class='legend'>"
                 "<span class='agree'>matches consensus</span>"
                 "<span class='disagree'>differs</span>"
                 "<span class='null'>null / not extracted</span>"
                 "&nbsp;&nbsp;<span style='color:var(--muted);'>Click a cell to mark that value as truth.</span></div>")
    parts.append("<table class='ext'><thead><tr>")
    parts.append("<th style='min-width:90px;'>decision</th>")
    parts.append("<th>Field</th><th>kind</th>")
    for p in parsers:
        parts.append(f"<th title='{html.escape(p)}'>{html.escape(PARSER_LABEL.get(p, p))}</th>")
    parts.append("<th>consensus</th>")
    parts.append("</tr></thead><tbody>")

    # Iterate schema fields in order; fall back to union of parser outputs if no schema
    if schema and schema.get("properties"):
        field_names = list(schema["properties"].keys())
        cats = field_categories(schema)
    else:
        all_fields: set = set()
        for p in parsers:
            if p in extractions and isinstance(extractions[p], dict):
                all_fields.update(extractions[p].keys())
        field_names = sorted(all_fields)
        cats = {f: "other" for f in field_names}

    for fld in field_names:
        # collect per-parser values
        values: dict[str, object] = {}
        norms: dict[str, object] = {}
        for p in parsers:
            if p not in extractions:
                values[p] = None; norms[p] = None
                continue
            v = extractions[p].get(fld) if isinstance(extractions[p], dict) else None
            values[p] = v
            norms[p] = normalize_value(v)
        # consensus
        nn = [n for n in norms.values() if n is not None]
        consensus = None
        if nn:
            tally = Counter(nn)
            top, c = tally.most_common(1)[0]
            if c * 2 > len(nn):
                consensus = top
        all_null = len(nn) == 0  # no parser extracted anything

        cat = cats.get(fld, "other")
        row_class = "data-row" + (" all-null" if all_null else "")
        parts.append(
            f"<tr class='{row_class}' data-field='{html.escape(fld)}' "
            f"data-allnull='{1 if all_null else 0}'>"
            # decision column FIRST (left)
            f"<td class='actions'>"
            f"<button class='btn-accept' title='Accept consensus or picked parser'>✓</button>"
            f"<button class='btn-reject' title='Reject all values for this field'>✗</button>"
            f"</td>"
            f"<td class='field-name' title='{html.escape(fld)}'>{html.escape(fld)}</td>"
            f"<td class='cat cat-{cat}'>{cat[:2]}</td>"
        )
        for p in parsers:
            v = values[p]
            n = norms[p]
            disp_html = format_cell_html(v)
            if n is None:
                klass = "cell-null"
                txt = "<i>null</i>" if not disp_html else f"<i>{disp_html}</i>"
                clickable = ""
            elif consensus is not None and n == consensus:
                klass = "cell-agree"
                txt = disp_html or "·"
                clickable = f"data-parser='{html.escape(p)}'"
            elif consensus is not None and n != consensus:
                klass = "cell-disagree"
                txt = disp_html or "·"
                clickable = f"data-parser='{html.escape(p)}'"
            else:
                klass = ""
                txt = disp_html or "·"
                clickable = f"data-parser='{html.escape(p)}'"
            full_v = json.dumps(v, default=str)[:500]
            parts.append(f"<td class='{klass}' {clickable} title='{html.escape(full_v)}'>{txt}</td>")
        # consensus column (rightmost) — render the raw value from a parser that matches consensus
        consensus_raw = None
        if consensus is not None:
            for p in parsers:
                if norms.get(p) == consensus and values.get(p) is not None:
                    consensus_raw = values[p]
                    break
        cdisp_html = format_cell_html(consensus_raw) if consensus_raw is not None else ""
        c_full = json.dumps(consensus_raw, default=str)[:500] if consensus_raw is not None else ""
        c_clickable = "data-parser='__consensus__'" if consensus is not None else ""
        parts.append(f"<td class='consensus-col' {c_clickable} title='{html.escape(c_full)}'>"
                     f"{cdisp_html or '—'}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table>")
    parts.append("</div></div>")

    # ============================================================
    # Markdown / tables panel — surfaces what the schema fields
    # don't capture (prose, tables, headings, table count, etc.)
    # ============================================================
    if parsed_mds:
        # Per-parser metrics + extracted tables
        per_parser_metrics: dict[str, dict] = {}
        per_parser_tables: dict[str, list[str]] = {}
        for p in parsers:
            md = parsed_mds.get(p, "")
            per_parser_metrics[p] = md_metrics(md)
            # Show ALL tables, full row counts
            per_parser_tables[p] = extract_table_blocks(md, max_blocks=None, max_lines_per_block=None)

        parts.append("<div class='md-section'>")
        parts.append("<h2>Raw markdown + tables (per parser)</h2>")
        parts.append("<div class='sub'>Useful for non-form docs where most signal is in prose and tables, not schema-shaped fields. "
                     "Stats compare what each parser surfaced from the same source PDF.</div>")

        # Stats grid: one card per metric, with all 9 parsers stacked
        metrics_to_show = [
            ("chars", "chars"),
            ("lines", "lines"),
            ("headings", "headings"),
            ("tables", "tables"),
            ("table_rows", "table rows"),
            ("images", "images"),
        ]
        parts.append("<div class='md-stats-bar'>")
        for key, label in metrics_to_show:
            vals = [(p, per_parser_metrics[p].get(key, 0)) for p in parsers if p in per_parser_metrics]
            max_v = max((v for _, v in vals), default=0)
            parts.append("<div class='md-stat'>")
            parts.append(f"<div class='name'>{html.escape(label)}</div>")
            parts.append("<div class='vals'>")
            for p, v in vals:
                pct = (v / max_v * 100) if max_v > 0 else 0
                hl = "color:var(--accent);font-weight:600;" if v == max_v and max_v > 0 else ""
                parts.append(f"<div style='{hl}'>{html.escape(PARSER_LABEL.get(p, p))}: <b>{v:,}</b></div>")
            parts.append("</div></div>")
        parts.append("</div>")

        # Tabs: one per parser, plus an "all tables" view
        parts.append("<div class='md-tabs'>")
        for i, p in enumerate(parsers):
            if p not in per_parser_metrics:
                continue
            active = " active" if i == 0 else ""
            parts.append(f"<button class='md-tab{active}' data-pane='md-pane-{html.escape(p)}'>"
                         f"{html.escape(PARSER_LABEL.get(p, p))}</button>")
        parts.append("</div>")

        # One pane per parser
        for i, p in enumerate(parsers):
            if p not in per_parser_metrics:
                continue
            hide = "" if i == 0 else " hidden"
            md = parsed_mds.get(p, "")
            tables = per_parser_tables.get(p, [])
            if len(md) > preview_limit:
                preview = md[:preview_limit] + (
                    f"\n\n… [truncated — md is {len(md):,} chars > {preview_limit:,} cap; "
                    f"full file at runs/parse_sweep/.../{p}/{domain}/{type_name}/{fname.replace('.pdf', '.md')}]")
            else:
                preview = md
            parts.append(f"<div class='md-pane{hide}' id='md-pane-{html.escape(p)}'>")
            if tables:
                parts.append(f"<div style='color:var(--muted);font-size:11px;margin-bottom:8px;'>"
                             f"<strong>{len(tables)} table block{'s' if len(tables) != 1 else ''} found</strong> "
                             f"(showing all, full rows)</div>")
                parts.append("<div class='md-tables-list'>")
                for ti, t in enumerate(tables, 1):
                    n_rows = t.count("\n") + 1
                    parts.append(f"<div class='table-block'>"
                                 f"<div style='color:var(--muted);font-size:10px;margin-bottom:4px;'>"
                                 f"Table {ti} / {len(tables)} — {n_rows} rows</div>"
                                 f"{html.escape(t)}</div>")
                parts.append("</div>")
                parts.append("<div style='color:var(--muted);font-size:11px;margin:14px 0 6px;'>Full markdown preview:</div>")
            else:
                parts.append("<div class='none'>No markdown table blocks found in this parser's output.</div>")
                parts.append("<div style='color:var(--muted);font-size:11px;margin:14px 0 6px;'>Markdown preview:</div>")
            parts.append(f"<pre>{html.escape(preview)}</pre>")
            parts.append("</div>")

        parts.append("</div>")  # /md-section

    parts.append("</main>")

    # JS for interactions
    # Embed parser values per field so we can serialize decisions correctly
    js_data: dict = {}
    for fld in field_names:
        per_parser = {}
        for p in parsers:
            if p in extractions and isinstance(extractions[p], dict):
                per_parser[p] = extractions[p].get(fld)
        js_data[fld] = per_parser
    # consensus values per field (for accept-without-pick = accept consensus)
    js_consensus: dict = {}
    for fld in field_names:
        norms_local: dict = {}
        for p in parsers:
            if p in extractions and isinstance(extractions[p], dict):
                norms_local[p] = normalize_value(extractions[p].get(fld))
        nn = [v for v in norms_local.values() if v is not None]
        cons_val = None
        if nn:
            tally = Counter(nn)
            top, c = tally.most_common(1)[0]
            if c * 2 > len(nn):
                # pick the raw value from the FIRST parser that had this normalized value
                for p, n in norms_local.items():
                    if n == top:
                        cons_val = extractions[p].get(fld)
                        break
        js_consensus[fld] = cons_val

    doc_storage_key = f"realdoc_v11_decisions__{safe_id(domain, type_name, fname)}"
    parts.append(
        f"<script>\n"
        f"const FIELD_VALUES = {json.dumps(js_data, default=str)};\n"
        f"const CONSENSUS = {json.dumps(js_consensus, default=str)};\n"
        f"const STORAGE_KEY = {json.dumps(doc_storage_key)};\n"
        f"const DOC = {json.dumps({'domain': domain, 'type': type_name, 'filename': fname})};\n"
        + DETAIL_JS +
        "</script>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True,
                    help="runs/parse_sweep/<ts>/ to read extractions from")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    if not sweep_dir.exists():
        print(f"!! {sweep_dir} not found")
        return 1

    EXT_DIR.mkdir(parents=True, exist_ok=True)

    # Use the predefined parser order; only include those actually present
    parsers = [p for p in PARSER_ORDER if (sweep_dir / p).exists()]
    print(f"Parsers in sweep: {parsers}")

    print("Loading extractions...")
    doc_extractions = load_doc_extractions(sweep_dir, parsers)
    print(f"  {len(doc_extractions)} docs with at least one extraction")

    # Group by domain → type
    docs_by_domain: dict = {}
    doc_to_safe_id: dict = {}
    for (domain, type_name, fname) in doc_extractions:
        docs_by_domain.setdefault(domain, {}).setdefault(type_name, []).append(fname)
        doc_to_safe_id[(domain, type_name, fname)] = safe_id(domain, type_name, fname)

    # Build ordered list for prev/next nav
    ordered: list = []
    for domain in docs_by_domain:
        for type_name in sorted(docs_by_domain[domain]):
            for fname in sorted(docs_by_domain[domain][type_name]):
                ordered.append((domain, type_name, fname))

    # Per-doc pages
    for i, (domain, type_name, fname) in enumerate(ordered):
        schema = load_schema(type_name)
        sid = doc_to_safe_id[(domain, type_name, fname)]
        prev_sid = doc_to_safe_id[ordered[i - 1]] if i > 0 else None
        next_sid = doc_to_safe_id[ordered[i + 1]] if i + 1 < len(ordered) else None
        ext = doc_extractions[(domain, type_name, fname)]
        mds = load_parsed_mds(Path(args.sweep_dir).resolve(),
                              domain, type_name, fname, parsers)
        # Get source PDF page count for truncation decision
        src_pdf = V1_1 / domain / type_name / "sources" / fname
        page_count = 0
        if src_pdf.exists():
            try:
                import fitz
                with fitz.open(src_pdf) as doc:
                    page_count = doc.page_count
            except Exception:
                page_count = 0
        html_text = build_detail(domain, type_name, fname, schema, ext, parsers,
                                 prev_sid, next_sid, parsed_mds=mds,
                                 source_page_count=page_count)
        (EXT_DIR / f"{sid}.html").write_text(html_text)
    print(f"  wrote {len(ordered)} per-doc pages")

    # Compute disagreement counts per doc + sort domains by total hotspots desc
    print("Computing per-doc disagreement counts for ranking...")
    disagreement_by_doc = compute_doc_disagreement(doc_extractions, parsers)
    docs_by_domain_sorted = dict(sorted(
        docs_by_domain.items(),
        key=lambda kv: -sum(disagreement_by_doc.get((kv[0], t, f), {}).get("hot", 0)
                            for t, fnames in kv[1].items() for f in fnames),
    ))

    # Index
    index_html = build_index(docs_by_domain_sorted, doc_to_safe_id, parsers,
                             disagreement_by_doc=disagreement_by_doc)
    (EXT_DIR / "index.html").write_text(index_html)
    print(f"  wrote {EXT_DIR.relative_to(REPO)}/index.html")
    n_hot_total = sum(d.get("hot", 0) for d in disagreement_by_doc.values())
    print(f"  total hotspot fields across all docs: {n_hot_total}")

    print(f"\nOpen: open '{EXT_DIR}/index.html'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
