#!/usr/bin/env python3
"""Build a doc picker HTML for selecting hard docs from v1.2_final.

Shows every page NOT already in runs/v500/sample.csv. User picks via checkboxes,
exports as JSON. Filters: domain, kind, category, density, type.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
V12F = REPO / "datasets" / "extraction" / "v1.2_final"
PICKER = REPO / "runs" / "picker"


def load_data():
    manifest = list(csv.DictReader((V12F / "_manifest.csv").open()))
    in_v500 = {r["dest_path"] for r in
               csv.DictReader((REPO / "runs" / "v500" / "sample.csv").open())}
    classes = {r["dest_path"]: r["category"]
               for r in csv.DictReader(open(REPO / "datasets" / "extraction" / "v1.2" / "_real_classifications.csv"))}
    densities = {r["dest_path"]: int(r["score"])
                 for r in csv.DictReader(open(REPO / "datasets" / "extraction" / "v1.2" / "_real_density_scores.csv"))}

    rows = []
    for r in manifest:
        if r["dest_path"] in in_v500:
            continue
        rows.append({
            "dest_path": r["dest_path"],
            "stem": Path(r["dest_path"]).stem,
            "kind": r["kind"],
            "domain": r["domain"],
            "type": r["type"],
            "doc_stem": r["doc_stem"],
            "page_num": int(r["page_num"]),
            "doc_total_pages": int(r["doc_total_pages"]),
            "category": classes.get(r["dest_path"], "?"),
            "density": densities.get(r["dest_path"], 0),
        })
    return rows


def emit_html(rows, out_path: Path):
    rows_json = json.dumps(rows)
    HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>doc picker</title>
<style>
  :root { color-scheme: dark light; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
         margin: 0; padding: 16px 24px; background: #0e1117; color: #e6edf3; font-size: 12px; }
  h1 { font-size: 17px; margin: 4px 0 6px; }
  .sticky-bar { position: sticky; top: 0; z-index: 50; background: #161b22;
                border-bottom: 1px solid #30363d; padding: 10px 14px; margin: 0 -24px 14px;
                display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .sticky-bar select, .sticky-bar input, .sticky-bar button {
    background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 4px;
    padding: 4px 8px; font-size: 11px; }
  .sticky-bar button.primary { background: #1f6feb; border-color: #1f6feb; cursor: pointer; }
  .sticky-bar button.danger { background: #f85149; border-color: #f85149; color: white; cursor: pointer; }
  .sticky-bar .count { font-weight: 700; color: #58a6ff; }
  .counts span { padding: 1px 6px; border-radius: 3px; font-weight: 600; margin-right: 3px; }
  .c-sel { background: #3fb950; color: #000; }
  .c-rej { background: #6e7681; color: white; }

  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
          gap: 8px; }
  .card { border: 2px solid #30363d; border-radius: 5px; overflow: hidden;
          background: #0d1117; cursor: pointer; transition: border-color 0.1s; position: relative; }
  .card.selected { border-color: #3fb950; box-shadow: 0 0 0 2px rgba(63,185,80,0.3); }
  .card.rejected { opacity: 0.4; border-color: #f85149; }
  .card img { width: 100%; display: block; max-height: 280px; object-fit: contain;
              background: #fff; }
  .card .meta { padding: 4px 6px; font-size: 10px; background: #161b22;
                line-height: 1.4; }
  .card .meta .type { color: #58a6ff; font-weight: 600; font-family: "SF Mono", monospace;
                       font-size: 10px; }
  .card .meta .stem { color: #c9d1d9; font-family: "SF Mono", monospace; font-size: 9px;
                       white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .card .meta .tags { display: flex; gap: 4px; margin-top: 2px; flex-wrap: wrap; }
  .card .meta .tag { padding: 0 5px; border-radius: 2px; font-size: 9px;
                      background: #30363d; color: #c9d1d9; }
  .card .meta .tag.density { background: #d29922; color: #000; font-weight: 700; }
  .card .meta .tag.hw { background: #f0883e; color: #000; font-weight: 700; }
  .card .meta .tag.tbl { background: #1f6feb; color: white; font-weight: 700; }
  .card .meta .tag.fig { background: #8b5cf6; color: white; font-weight: 700; }
  .card .check { position: absolute; top: 6px; right: 6px; width: 24px; height: 24px;
                 background: rgba(0,0,0,0.65); border: 2px solid white; border-radius: 4px;
                 display: flex; align-items: center; justify-content: center;
                 font-weight: 700; color: white; font-size: 14px; }
  .card.selected .check { background: #3fb950; }
  #export-panel { position: fixed; top: 5vh; left: 50%; transform: translateX(-50%);
                  background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                  padding: 20px; width: 90vw; max-width: 900px; max-height: 90vh;
                  overflow: auto; z-index: 100; display: none;
                  box-shadow: 0 8px 24px rgba(0,0,0,0.7); }
  #export-panel.show { display: block; }
  #export-panel textarea { width: 100%; min-height: 60vh; background: #0d1117;
                           color: #e6edf3; border: 1px solid #30363d; border-radius: 3px;
                           padding: 8px; font-family: "SF Mono", monospace; font-size: 10px; }
  #overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 99;
             display: none; }
  #overlay.show { display: block; }
  #zoom { position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 200;
          display: none; align-items: center; justify-content: center; cursor: zoom-out; }
  #zoom.show { display: flex; }
  #zoom img { max-width: 95vw; max-height: 95vh; }
</style></head><body>
<div class="sticky-bar">
  <h1 style="margin:0">Pick more hard docs</h1>
  <span class="counts">
    <span class="c-sel">selected <span id="n-sel">0</span></span>
  </span>
  <span style="color:#8b949e">/ <span id="n-total">0</span> shown · <span id="n-all">0</span> total</span>
  <span style="color:#d29922">target: ~500</span>

  <label>kind
    <select id="f-kind" onchange="renderAll()">
      <option value="">all</option><option>real</option><option>synthetic</option>
    </select>
  </label>
  <label>domain
    <select id="f-domain" onchange="renderAll()">
      <option value="">all</option>
      <option>finance</option><option>medical_healthcare</option>
      <option>mortgage</option><option>supply_chain</option>
    </select>
  </label>
  <label>category
    <select id="f-cat" onchange="renderAll()">
      <option value="">all</option>
      <option>handwriting</option><option>tables</option><option>form</option>
      <option>checkboxes</option><option>charts</option><option>figures</option>
      <option>mixed</option><option value="?">unclassified (synth)</option>
    </select>
  </label>
  <label>type
    <select id="f-type" onchange="renderAll()">
      <option value="">all</option>
    </select>
  </label>
  <label>density ≥
    <select id="f-density" onchange="renderAll()">
      <option value="0">0 (any)</option>
      <option value="5">5</option>
      <option value="7">7</option>
      <option value="8">8</option>
      <option value="9">9</option>
      <option value="10">10 (densest)</option>
    </select>
  </label>
  <label>state
    <select id="f-state" onchange="renderAll()">
      <option value="">all</option>
      <option value="selected">selected only</option>
      <option value="unselected">unselected only</option>
    </select>
  </label>

  <button onclick="selectAllVisible()">Select all visible</button>
  <button class="danger" onclick="deselectAll()">Deselect all</button>
  <button class="primary" onclick="exportSelection()">Export ↓</button>
</div>

<div id="grid" class="grid"></div>

<div id="overlay" onclick="closeExport()"></div>
<div id="export-panel">
  <h3 style="margin:0 0 12px">Selected docs JSON (paste back to Claude)</h3>
  <textarea id="export-text" readonly></textarea>
  <div style="margin-top:8px">
    <button class="primary" onclick="copyExport()">Copy</button>
    <button onclick="closeExport()">Close</button>
  </div>
</div>
<div id="zoom" onclick="closeZoom()"><img id="zoom-img"></div>
"""
    SCRIPT = """
<script>
const ROWS = __ROWS__;
const KEY = "picker-selected-v1";
let SEL = JSON.parse(localStorage.getItem(KEY) || "{}");

// populate type dropdown
const types = [...new Set(ROWS.map(r => r.type))].sort();
const fType = document.getElementById("f-type");
types.forEach(t => {
  const o = document.createElement("option");
  o.value = t; o.textContent = t;
  fType.appendChild(o);
});

function save() {
  localStorage.setItem(KEY, JSON.stringify(SEL));
  document.getElementById("n-sel").textContent = Object.keys(SEL).length;
}
function toggle(dp) {
  if (SEL[dp]) delete SEL[dp];
  else SEL[dp] = true;
  save();
  const el = document.getElementById("c-" + cssId(dp));
  if (el) el.classList.toggle("selected");
}
function cssId(dp) { return dp.replace(/[^a-z0-9]/gi, "_"); }
function selectAllVisible() {
  if (!confirm("Select all currently visible?")) return;
  currentVisible().forEach(r => SEL[r.dest_path] = true);
  save(); renderAll();
}
function deselectAll() {
  if (!confirm("Clear all selections?")) return;
  SEL = {}; localStorage.removeItem(KEY); save(); renderAll();
}

function currentVisible() {
  const kindF = document.getElementById("f-kind").value;
  const domF = document.getElementById("f-domain").value;
  const catF = document.getElementById("f-cat").value;
  const typeF = document.getElementById("f-type").value;
  const densMin = parseInt(document.getElementById("f-density").value);
  const stateF = document.getElementById("f-state").value;
  return ROWS.filter(r => {
    if (kindF && r.kind !== kindF) return false;
    if (domF && r.domain !== domF) return false;
    if (catF && r.category !== catF) return false;
    if (typeF && r.type !== typeF) return false;
    if (r.density < densMin) return false;
    const isSel = !!SEL[r.dest_path];
    if (stateF === "selected" && !isSel) return false;
    if (stateF === "unselected" && isSel) return false;
    return true;
  });
}

function renderAll() {
  const visible = currentVisible();
  const grid = document.getElementById("grid");
  // Limit to first 1000 for performance — show count
  const display = visible.slice(0, 1000);
  grid.innerHTML = display.map(r => {
    const isSel = !!SEL[r.dest_path];
    const tags = [
      r.category !== "?" ? `<span class="tag ${r.category === "handwriting" ? "hw" : r.category === "tables" ? "tbl" : r.category === "charts" || r.category === "figures" ? "fig" : ""}">${r.category}</span>` : "",
      r.density >= 5 ? `<span class="tag density">d${r.density}</span>` : "",
    ].filter(Boolean).join("");
    return `<div class="card ${isSel ? "selected" : ""}" id="c-${cssId(r.dest_path)}" onclick="toggle('${r.dest_path}')">
      <img src="thumbs/${r.stem}.png" loading="lazy" onclick="event.stopPropagation(); zoom('thumbs/${r.stem}.png');">
      <div class="check">${isSel ? "✓" : ""}</div>
      <div class="meta">
        <div class="type">${r.type}</div>
        <div class="stem" title="${r.dest_path}">${r.doc_stem.slice(0, 32)} · p${r.page_num}/${r.doc_total_pages}</div>
        <div class="tags">${tags}<span class="tag">${r.kind === "real" ? "REAL" : "SYNTH"} · ${r.domain.slice(0,3)}</span></div>
      </div>
    </div>`;
  }).join("");
  document.getElementById("n-total").textContent = visible.length + (visible.length > 1000 ? ` (showing first 1000)` : "");
  document.getElementById("n-all").textContent = ROWS.length;
}

function exportSelection() {
  const out = { selected: Object.keys(SEL).sort(), count: Object.keys(SEL).length };
  document.getElementById("export-text").value = JSON.stringify(out, null, 2);
  document.getElementById("overlay").classList.add("show");
  document.getElementById("export-panel").classList.add("show");
}
function closeExport() {
  document.getElementById("overlay").classList.remove("show");
  document.getElementById("export-panel").classList.remove("show");
}
function copyExport() {
  document.getElementById("export-text").select();
  document.execCommand("copy"); alert("Copied!");
}
function zoom(src) {
  document.getElementById("zoom-img").src = src;
  document.getElementById("zoom").classList.add("show");
}
function closeZoom() { document.getElementById("zoom").classList.remove("show"); }

renderAll();
save();
</script></body></html>
"""
    out_path.write_text(HEAD + SCRIPT.replace("__ROWS__", rows_json))


def main() -> int:
    PICKER.mkdir(parents=True, exist_ok=True)
    rows = load_data()
    print(f"loaded {len(rows)} candidate docs (not in v500)")
    from collections import Counter
    print("\nBy (kind, domain):")
    for k, n in Counter((r["kind"], r["domain"]) for r in rows).most_common():
        print(f"  {k[0]:10}/{k[1]:20} {n:>5}")
    print("\nBy category (top 10):")
    for c, n in Counter(r["category"] for r in rows).most_common(10):
        print(f"  {c:25} {n:>5}")
    print("\nBy density (top 5):")
    for d, n in Counter(r["density"] for r in rows).most_common(11):
        print(f"  density {d:>2} : {n:>5}")

    out = PICKER / "select.html"
    emit_html(rows, out)
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
