"""Generate a per-doc HTML review tool for the v0_tax dataset.

For each doc, produces a self-contained HTML page with:
  - LEFT pane: PDF rendered page-by-page as inline thumbnails (click to zoom)
  - RIGHT pane: every schema field with its definition, current GT value
    (editable in-browser), and what each parser×agent produced (for context)
  - Bottom: a notes pane for marking schema-level changes ("drop this field",
    "rename to X", etc.) — saved to a sidecar .notes.json
  - Top toolbar: "Export GT JSON" copies the edited GT to clipboard; you
    paste it back into the ground_truth/<doc>.json file

Two-stage workflow:
  1. python scripts/review_v0tax.py --build-thumbs    # one-time, slow
  2. python scripts/review_v0tax.py                   # regenerate HTML; fast
  3. open runs/v0tax_review/<doc_id>.html

Usage:
  python scripts/review_v0tax.py                     # all 6 docs
  python scripts/review_v0tax.py --doc tax_harris_2023
  python scripts/review_v0tax.py --schema-template _proposed_1040_individual.json
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "datasets" / "extraction" / "v0_tax"
SOURCES = DATA / "sources"
SCHEMAS = DATA / "schemas"
GT = DATA / "ground_truth"
OUT = REPO / "runs" / "v0tax_review"
THUMBS = OUT / ".thumbs"

# Schema-to-doc mapping (which proposed schema applies to which doc)
DOC_SCHEMA_MAP = {
    "tax_harris_2023":    "_proposed_1040_individual.json",
    "tax_obama_2009":     "_proposed_1040_individual.json",
    "tax_biden_2018":     "_proposed_1040_individual.json",
    "tax_harris_2020":    "_proposed_1040_individual.json",
    "tax_trump_lfb_2015": "_proposed_1065_partnership.json",
    "tax_trump_2019":     "_proposed_4562_attachment.json",
}

# Where to read existing parser outputs from (for the "what each parser said" context column)
PARSER_RUNS = {
    "Extend v2 GA":  ("runs/bench_v0tax_extend_performance_v2_0_0", "extend_performance_v2_0_0"),
    "AWS Textract":  ("runs/bench_v0tax_aws_textract",              "aws_textract"),
    "Azure DI":      ("runs/bench_v0tax_azure_di",                  "azure_di"),
    "Reducto":       ("runs/bench_v0tax_reducto",                   "reducto"),
    "pymupdf":       ("runs/bench_v0tax_pymupdf",                   "pymupdf"),
}


# ── PDF rendering ───────────────────────────────────────────────────────────


def render_thumbs(doc_id: str, max_pages: int = 200, dpi: int = 100) -> list[Path]:
    """Render every PDF page to a cached PNG. Returns list of file paths."""
    pdf_path = SOURCES / f"{doc_id}.pdf"
    out_dir = THUMBS / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        import pymupdf
    except ImportError:
        print("pymupdf not installed — page thumbnails will be missing.")
        return []
    pdf = pymupdf.open(str(pdf_path))
    for i, page in enumerate(pdf):
        if i >= max_pages:
            break
        cached = out_dir / f"p{i+1:04d}.png"
        if not cached.exists():
            pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72))
            pix.save(str(cached))
        paths.append(cached)
    pdf.close()
    return paths


# ── Schema introspection ────────────────────────────────────────────────────


def schema_fields(schema: dict) -> list[dict]:
    """Walk top-level properties (we keep the visual UI top-level for editability).

    Returns one entry per top-level property with:
      path, type, description, x-difficulty, x-capabilities, required, nested_shape
    """
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    out = []
    for name, p in props.items():
        out.append({
            "path": name,
            "type": p.get("type"),
            "description": p.get("description", ""),
            "difficulty": p.get("x-difficulty"),
            "capabilities": p.get("x-capabilities", []),
            "required": name in required,
            "enum": p.get("enum"),
            "nested": p,  # full subschema for arrays/objects, used for display
            "is_array": p.get("type") == "array" or (isinstance(p.get("type"), list) and "array" in p.get("type")),
            "is_object": p.get("type") == "object" or (isinstance(p.get("type"), list) and "object" in p.get("type")),
        })
    return out


# ── Parser-output lookup ────────────────────────────────────────────────────


def load_agent_outputs(doc_id: str) -> dict[str, dict[str, dict]]:
    """Returns {parser_label: {agent: {field_path: actual_value}}}."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for label, (run_dir, parser_name) in PARSER_RUNS.items():
        run = REPO / run_dir / doc_id / parser_name
        if not run.exists():
            continue
        for agent_dir in run.iterdir():
            if not agent_dir.is_dir():
                continue
            ao = agent_dir / "agent_output.json"
            if not ao.exists():
                continue
            try:
                data = json.load(open(ao))
            except Exception:
                continue
            out[label][agent_dir.name] = data
    return out


# ── HTML rendering ──────────────────────────────────────────────────────────


CSS = """
* { box-sizing: border-box; }
body { margin:0; padding:0; font-family:-apple-system,BlinkMacSystemFont,system-ui,sans-serif; color:#1a1a1a; background:#fafafa; }
.toolbar { position:sticky; top:0; z-index:100; background:#1a1a1a; color:white; padding:10px 18px; display:flex; gap:14px; align-items:center; box-shadow:0 2px 8px rgba(0,0,0,0.15); }
.toolbar h1 { margin:0; font-size:16px; flex:1; }
.toolbar button { background:#4a90e2; color:white; border:0; padding:8px 14px; border-radius:5px; font-size:13px; cursor:pointer; }
.toolbar button:hover { background:#3a7bc8; }
.toolbar .schema-tag { background:#444; padding:4px 10px; border-radius:4px; font-size:12px; font-family:monospace; }
.split { display:flex; height:calc(100vh - 50px); }
.pane-left { width:38%; overflow-y:auto; background:#202020; padding:14px; }
.pane-right { width:62%; overflow-y:auto; padding:0 18px; }
.pane-left h2 { color:white; font-size:13px; margin:0 0 10px; opacity:0.7; text-transform:uppercase; letter-spacing:0.5px; }
.thumb { display:block; width:100%; margin-bottom:14px; border:1px solid #444; cursor:zoom-in; background:white; }
.thumb-label { color:#bbb; font-size:11px; margin-bottom:4px; }

.field-card { background:white; margin:18px 0; padding:14px 18px; border-radius:6px; box-shadow:0 1px 3px rgba(0,0,0,0.08); border-left:4px solid transparent; }
.field-card.required { border-left-color:#e74c3c; }
.field-card.status-verified  { background:#f4faf4; border-left-color:#16a04c; }
.field-card.status-corrected { background:#fff8e8; border-left-color:#e6a82e; }
.field-card.status-disputed  { background:#fff0f0; border-left-color:#c0392b; }
.field-card.status-needs_review { background:#fffce8; border-left-color:#c8a800; }
.status-badge { display:inline-block; padding:3px 8px; border-radius:3px; font-size:11px; font-weight:600; margin-right:8px; }
.status-badge.status-verified  { background:#16a04c; color:white; }
.status-badge.status-corrected { background:#e6a82e; color:white; }
.status-badge.status-disputed  { background:#c0392b; color:white; }
.status-badge.status-needs_review { background:#a37800; color:white; }
.status-badge.status-unverified { background:#999; color:white; }

.verifier-row { display:flex; gap:10px; margin-top:6px; padding:6px 10px; border-radius:4px; background:#fafafa; border-left:3px solid #ccc; font-size:11px; }
.verifier-row.verifier-confirm { border-left-color:#16a04c; }
.verifier-row.verifier-correct { border-left-color:#e6a82e; }
.verifier-row.verifier-unsure  { border-left-color:#999; }
.verifier-name { width:130px; font-family:monospace; font-weight:600; color:#444; }
.verifier-body { flex:1; }
.verifier-corrected { font-family:ui-monospace,monospace; color:#7a5d00; margin-top:3px; }
.verifier-corrected code { background:#fff5d9; padding:1px 5px; border-radius:2px; }
.verifier-expl { color:#666; font-style:italic; margin-top:3px; line-height:1.4; }
.field-card .field-header { display:flex; align-items:baseline; gap:10px; margin-bottom:6px; }
.field-card .field-name { font-family:ui-monospace,monospace; font-weight:600; font-size:14px; color:#0050a0; }
.field-card .field-type { color:#888; font-size:11px; font-family:monospace; }
.field-card .field-diff { padding:2px 8px; border-radius:3px; font-size:10px; font-weight:600; text-transform:uppercase; }
.field-diff.easy { background:#d4f1d4; color:#1a7f1a; }
.field-diff.medium { background:#fff4c2; color:#a07000; }
.field-diff.hard { background:#ffd4d4; color:#a02020; }
.field-card .field-caps { color:#666; font-size:11px; font-style:italic; }
.field-card .field-desc { color:#333; font-size:13px; line-height:1.5; margin:6px 0 10px; }

.gt-row { display:flex; align-items:flex-start; gap:10px; padding:8px 0; border-top:1px solid #eee; }
.gt-label { width:80px; font-weight:600; color:#1a7f1a; font-size:12px; }
.gt-editor { flex:1; background:#f5fdf5; border:1px solid #b8e0b8; padding:8px 10px; border-radius:4px; font-family:ui-monospace,monospace; font-size:12px; white-space:pre-wrap; min-height:1.5em; }
.gt-editor:focus { outline:2px solid #4a90e2; background:white; }

.parser-row { display:flex; align-items:flex-start; gap:10px; padding:5px 0; font-size:12px; }
.parser-name { width:80px; color:#666; font-size:11px; font-family:monospace; }
.parser-agent { width:60px; color:#999; font-size:10px; }
.parser-value { flex:1; font-family:ui-monospace,monospace; font-size:11px; color:#444; padding:3px 8px; background:#f8f8f8; border-radius:3px; max-height:120px; overflow-y:auto; white-space:pre-wrap; }
.parser-value.match-gt { background:#e9f7e9; color:#1a7f1a; }
.parser-value.differs { background:#fff5e8; }

details summary { cursor:pointer; padding:6px 0; color:#666; font-size:12px; }
details[open] summary { color:#1a1a1a; }

.notes-pane { background:#fffbe6; border:1px solid #ffe48a; padding:14px 18px; margin:24px 0; border-radius:6px; }
.notes-pane h3 { margin:0 0 8px; font-size:13px; color:#7a5d00; }
.notes-pane textarea { width:100%; min-height:80px; border:1px solid #d9c060; padding:8px; font-family:ui-monospace,monospace; font-size:12px; border-radius:4px; }

.zoom-overlay { position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.85); z-index:1000; display:flex; align-items:center; justify-content:center; cursor:zoom-out; }
.zoom-overlay img { max-width:90vw; max-height:90vh; box-shadow:0 0 40px rgba(255,255,255,0.2); }
"""

JS = r"""
function zoomImg(src) {
  const ov = document.createElement('div');
  ov.className = 'zoom-overlay';
  ov.onclick = () => ov.remove();
  const img = document.createElement('img');
  img.src = src;
  ov.appendChild(img);
  document.body.appendChild(ov);
}
function exportGT() {
  const fields = document.querySelectorAll('[data-gt-field]');
  const out = {};
  for (const el of fields) {
    const k = el.dataset.gtField;
    let v = el.innerText.trim();
    try { v = JSON.parse(v); } catch(e) {/* keep as string */}
    out[k] = v;
  }
  const json = JSON.stringify(out, null, 2);
  navigator.clipboard.writeText(json).then(() => {
    alert(`Copied GT JSON (${Object.keys(out).length} fields) to clipboard.\nPaste it into:\n` + window._GT_PATH);
  });
}
function saveNotes() {
  const txt = document.getElementById('schema-notes').value;
  const blob = new Blob([txt], {type:'text/plain'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = window._DOC_ID + '.notes.txt';
  link.click();
}
"""


def render_value(v, max_len=200) -> str:
    if v is None:
        return "null"
    if isinstance(v, (dict, list)):
        s = json.dumps(v, indent=2, default=str)
    else:
        s = str(v)
    if len(s) > max_len:
        s = s[:max_len] + "…"
    return s


def load_verification_sidecar(doc_id: str) -> dict:
    """Read ground_truth/<doc>.verification.json if present."""
    p = GT / f"{doc_id}.verification.json"
    if p.exists():
        try:
            return json.load(open(p))
        except Exception:
            return {}
    return {}


def render_html(doc_id: str, schema_name: str) -> str:
    schema = json.loads((DATA / schema_name).read_text())
    fields = schema_fields(schema)
    # Existing GT (the unified-schema GT) — surface its values as the "current GT" starting point
    current_gt = {}
    gt_path = GT / f"{doc_id}.json"
    if gt_path.exists():
        try:
            current_gt = json.load(open(gt_path))
        except Exception:
            pass

    verification = load_verification_sidecar(doc_id)
    agent_outputs = load_agent_outputs(doc_id)
    thumbs = render_thumbs(doc_id)

    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append(f"<title>{doc_id} — v0_tax review</title>")
    parts.append(f"<style>{CSS}</style></head><body>")
    parts.append(
        f"<div class='toolbar'><h1>📄 {doc_id}</h1>"
        f"<span class='schema-tag'>{schema_name}</span>"
        f"<span style='color:#aaa;font-size:12px'>{len(fields)} fields • {len(thumbs)} pages</span>"
        f"<button onclick='exportGT()'>Export GT → clipboard</button>"
        f"<button onclick='saveNotes()'>Download notes</button>"
        f"</div>"
    )
    parts.append("<div class='split'>")
    # LEFT: PDF pages
    parts.append("<div class='pane-left'>")
    parts.append("<h2>PDF pages</h2>")
    for i, p in enumerate(thumbs, 1):
        b64 = base64.b64encode(p.read_bytes()).decode()
        src = f"data:image/png;base64,{b64}"
        parts.append(f"<div class='thumb-label'>Page {i}</div>")
        parts.append(f"<img class='thumb' src='{src}' onclick='zoomImg(this.src)'>")
    parts.append("</div>")

    # RIGHT: fields
    parts.append("<div class='pane-right'>")
    parts.append(f"<p style='color:#666; font-size:13px; margin-top:16px'>"
                 f"Edit GT values in the green boxes. Click <strong>Export GT</strong> to copy the JSON to clipboard, "
                 f"then paste into <code>{GT.relative_to(REPO)}/{doc_id}.json</code>. "
                 f"Use the yellow notes pane below for schema-level comments (drop/rename/add fields).</p>")
    for f in fields:
        v = verification.get(f["path"]) or {}
        v_status = v.get("status", "unverified")
        cls = "field-card status-" + v_status + (" required" if f["required"] else "")
        parts.append(f"<div class='{cls}'>")
        parts.append(f"<div class='field-header'>")
        # Verification status badge
        badge_emoji = {"verified": "✅", "corrected": "🔧", "disputed": "⚠️", "needs_review": "🟡"}.get(v_status, "◯")
        badge_label = {"verified": "verified-by-VLMs", "corrected": "corrected-by-VLMs",
                       "disputed": "disputed (verify!)", "needs_review": "needs review", "unverified": "unverified"}.get(v_status, "unverified")
        parts.append(f"<span class='status-badge status-{v_status}' title='Pages inspected: {v.get(\"pages_inspected\")}'>{badge_emoji} {badge_label}</span>")
        parts.append(f"<span class='field-name'>{html.escape(f['path'])}</span>")
        t = json.dumps(f["type"]) if isinstance(f["type"], list) else f["type"]
        parts.append(f"<span class='field-type'>{html.escape(str(t))}</span>")
        if f["difficulty"]:
            parts.append(f"<span class='field-diff {f['difficulty']}'>{f['difficulty']}</span>")
        if f["required"]:
            parts.append(f"<span style='color:#e74c3c;font-size:11px'>required</span>")
        parts.append("</div>")
        parts.append(f"<div class='field-desc'>{html.escape(f['description'])}</div>")
        if f["capabilities"]:
            parts.append(f"<div class='field-caps'>capabilities: {', '.join(f['capabilities'])}</div>")
        if f["enum"]:
            parts.append(f"<div class='field-caps'>enum: {', '.join(repr(x) for x in f['enum'])}</div>")
        if f["is_object"] and "properties" in f["nested"]:
            sub = [f"  - {k}: {(v.get('description','') or '')[:60]}" for k, v in f["nested"]["properties"].items()]
            parts.append(f"<details><summary>object shape ({len(sub)} fields)</summary><pre style='font-size:11px;color:#444;background:#f5f5f5;padding:8px'>{html.escape(chr(10).join(sub))}</pre></details>")
        if f["is_array"] and "items" in f["nested"]:
            it = f["nested"]["items"]
            if it.get("properties"):
                sub = [f"  - {k}: {(v.get('description','') or '')[:60]}" for k, v in it["properties"].items()]
                parts.append(f"<details><summary>array element shape ({len(sub)} fields per item)</summary><pre style='font-size:11px;color:#444;background:#f5f5f5;padding:8px'>{html.escape(chr(10).join(sub))}</pre></details>")
            else:
                parts.append(f"<div class='field-caps'>array items: {it.get('type','?')}</div>")

        # GT editor
        gt_val = current_gt.get(f["path"], None)
        parts.append(f"<div class='gt-row'>")
        parts.append(f"<div class='gt-label'>GT</div>")
        parts.append(f"<div class='gt-editor' contenteditable='true' data-gt-field='{html.escape(f['path'])}'>{html.escape(render_value(gt_val, max_len=2000))}</div>")
        parts.append(f"</div>")

        # Verifier verdicts (if available)
        verdicts = v.get("verdicts", {})
        for vname, label in [("gemini_3_pro", "Gemini-3-Pro"), ("claude_opus_4_7", "Claude-Opus-4.7")]:
            vrd = verdicts.get(vname) or {}
            if not vrd:
                continue
            verdict = vrd.get("verdict", "?")
            v_emoji = {"confirm": "✓", "correct": "🔧", "unsure": "?"}.get(verdict, "?")
            parts.append(f"<div class='verifier-row verifier-{verdict}'>")
            parts.append(f"<div class='verifier-name'>{v_emoji} {label}</div>")
            parts.append(f"<div class='verifier-body'>")
            parts.append(f"<div class='verifier-conf'>verdict: <strong>{verdict}</strong>")
            if vrd.get("confidence"):
                parts.append(f"  · confidence: {vrd['confidence']}")
            parts.append("</div>")
            if verdict == "correct" and "corrected_value" in vrd:
                parts.append(f"<div class='verifier-corrected'>corrected to: <code>{html.escape(render_value(vrd['corrected_value'], max_len=300))}</code></div>")
            if vrd.get("explanation"):
                parts.append(f"<div class='verifier-expl'>{html.escape(vrd['explanation'][:400])}</div>")
            parts.append(f"</div></div>")

        # Parser outputs (context)
        for label in PARSER_RUNS:
            for agent in ("claude_sonnet_4_6", "gemini_3_flash"):
                actual = agent_outputs.get(label, {}).get(agent, {}).get(f["path"], "<<not extracted>>")
                if actual == "<<not extracted>>":
                    continue
                # color-code: green if matches GT, beige if differs
                matches = (gt_val == actual)
                cls = "match-gt" if matches else "differs"
                parts.append(f"<div class='parser-row'>")
                parts.append(f"<div class='parser-name'>{label}</div>")
                parts.append(f"<div class='parser-agent'>{agent.split('_')[0]}</div>")
                parts.append(f"<div class='parser-value {cls}'>{html.escape(render_value(actual, max_len=400))}</div>")
                parts.append(f"</div>")
        parts.append("</div>")

    # Notes pane
    parts.append("<div class='notes-pane'>")
    parts.append("<h3>📝 Schema notes (free text — &lt;Download notes&gt; saves to .notes.txt)</h3>")
    parts.append("<textarea id='schema-notes' placeholder='Comments on schema: drop X, rename Y to Z, add field for foreign tax credit array, etc.'></textarea>")
    parts.append("</div>")
    parts.append("</div>")  # pane-right
    parts.append("</div>")  # split

    parts.append(f"<script>window._DOC_ID = {json.dumps(doc_id)}; window._GT_PATH = {json.dumps(str(GT / f'{doc_id}.json'))};</script>")
    parts.append(f"<script>{JS}</script>")
    parts.append("</body></html>")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", help="Single doc_id (default: all 6).")
    ap.add_argument("--schema", help="Override schema file (relative to v0_tax/).")
    ap.add_argument("--build-thumbs", action="store_true", help="(no-op; thumbs are auto-cached on first render).")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    docs = [args.doc] if args.doc else list(DOC_SCHEMA_MAP)
    for d in docs:
        schema = args.schema or DOC_SCHEMA_MAP[d]
        html_text = render_html(d, schema)
        out_path = OUT / f"{d}.html"
        out_path.write_text(html_text)
        print(f"wrote {out_path}   ({out_path.stat().st_size / 1024:.0f} KB)")
    print(f"\nopen the html files: open {OUT}/<doc>.html")


if __name__ == "__main__":
    main()
