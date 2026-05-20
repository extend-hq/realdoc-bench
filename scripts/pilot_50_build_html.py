#!/usr/bin/env python3
"""Build the pilot_50 review HTML.

Two modes:
  --mode pre   : pre-eval review (image + Qs + gold). User approves/rejects gold.
  --mode post  : post-eval review (image + Qs + gold + 4 parser answers + difficulty)
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PILOT = REPO / "runs" / "pilot_50"


def _set_run_dir(run_dir: Path) -> None:
    global PILOT
    PILOT = run_dir


def load_data(mode: str):
    bank = json.loads((PILOT / "qa" / "bank_r1.json").read_text())
    sample = {r["dest_path"]: r for r in csv.DictReader((PILOT / "sample.csv").open())}

    # source_file in bank == PDF stem; reverse-lookup dest_path via sample
    stem_to_dest = {Path(r["dest_path"]).stem: r["dest_path"] for r in sample.values()}

    # group questions by page
    by_page = defaultdict(list)
    for it in bank["items"]:
        dp = stem_to_dest.get(it["source_file"])
        if not dp:
            continue
        by_page[dp].append(it)

    eval_data = {}
    parser_md = {}  # {(dest_path, parser_name): markdown_text}
    if mode == "post":
        # Prefer rescored if available (has parsers_correct_strict/fuzzy/llm)
        rescored = PILOT / "eval" / "eval_r1_rescored.json"
        if rescored.exists():
            eval_data = json.loads(rescored.read_text())
        else:
            eval_path = PILOT / "eval" / "eval_r1.json"
            if eval_path.exists():
                eval_data = json.loads(eval_path.read_text())
        # Load parser markdowns (truncated to keep HTML reasonable)
        for dp in sample:
            stem = Path(dp).with_suffix("").as_posix()
            for pname in ("extend_performance_v1_0_0", "extend_performance_v2_0_0_advanced",
                          "extend_performance_v2_0_0_agentic",
                          "aws_textract", "azure_di", "reducto", "llamaparse", "llamaparse_cost_effective"):
                md_path = PILOT / "parses" / pname / f"{stem}.md"
                if md_path.exists():
                    md = md_path.read_text()
                    parser_md[(dp, pname)] = md[:8000]  # cap per parser per page

    pages = []
    for dp, qs in sorted(by_page.items()):
        meta = sample[dp]
        page_eval = eval_data.get(dp, {}) if eval_data else {}
        page_mds = {}
        if mode == "post":
            for pname in ("extend_performance_v1_0_0", "extend_performance_v2_0_0_advanced",
                          "extend_performance_v2_0_0_agentic",
                          "aws_textract", "azure_di", "reducto", "llamaparse", "llamaparse_cost_effective"):
                md = parser_md.get((dp, pname))
                if md is not None:
                    page_mds[pname] = md
        pages.append({
            "dest_path": dp,
            "stem": Path(dp).stem,
            "stratum": meta["stratum"],
            "domain": meta["domain"],
            "kind": meta["kind"],
            "type": meta["type"],
            "category": meta["category"],
            "density": meta["density"],
            "questions": qs,
            "eval": page_eval,
            "parser_mds": page_mds,
        })
    return pages, eval_data


def emit_html(pages, mode: str, out_path: Path):
    pages_json = json.dumps(pages, ensure_ascii=False)

    HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>pilot_50 review</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
         margin: 0; padding: 16px 24px; background: #ffffff; color: #1f2328; font-size: 13px; }
  h1 { font-size: 18px; margin: 4px 0 8px; }
  .sticky-bar { position: sticky; top: 0; z-index: 50; background: #f6f8fa;
                border-bottom: 1px solid #d1d9e0; padding: 10px 14px; margin: 0 -24px 16px;
                display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
  .sticky-bar select, .sticky-bar input, .sticky-bar button {
    background: #f6f8fa; color: #1f2328; border: 1px solid #d1d9e0; border-radius: 5px;
    padding: 5px 9px; font-size: 12px; }
  .sticky-bar button.primary { background: #0969da; border-color: #0969da; cursor: pointer; }
  .counts span { padding: 1px 7px; border-radius: 3px; font-weight: 600; margin-right: 4px; }
  .c-app { background: #1a7f37; color: #fff; }
  .c-rej { background: #cf222e; color: #fff; }
  .c-und { background: #9a6700; color: #fff; }

  .page-card { display: grid; grid-template-columns: 320px 1fr; gap: 14px;
               border: 1px solid #d1d9e0; border-radius: 8px; padding: 12px;
               margin-bottom: 18px; background: #ffffff; transition: border-color 0.1s; }
  .page-card.doc-approved { border-color: #1a7f37; box-shadow: inset 0 0 0 1px #1a7f3744; }
  .page-card.doc-rejected { border-color: #cf222e; opacity: 0.45; box-shadow: inset 0 0 0 1px #cf222e44; }
  .doc-actions { display: flex; gap: 6px; margin-top: 8px; align-items: center; flex-wrap: wrap; }
  .doc-actions button { font-size: 11px; padding: 4px 10px; cursor: pointer;
                        background: #f6f8fa; color: #1f2328; border: 1px solid #d1d9e0;
                        border-radius: 4px; font-weight: 600; }
  .doc-actions button:hover { background: #d1d9e0; }
  .doc-actions button.d-app { color: #1a7f37; }
  .doc-actions button.d-rej { color: #cf222e; }
  .page-card.doc-approved .doc-actions button.d-app { background: #1a7f37; color: #fff; }
  .page-card.doc-rejected .doc-actions button.d-rej { background: #cf222e; color: #fff; }
  .doc-actions .doc-status { font-size: 10px; color: #59636e; margin-left: 6px; }
  .doc-actions .doc-hardness { font-size: 10px; padding: 1px 6px; border-radius: 3px;
                               background: #d1d9e0; color: #1f2328; font-family: "SF Mono", monospace; }
  .page-card .left img { width: 100%; cursor: zoom-in; border: 1px solid #d1d9e0;
                          background: #fff; }
  .page-card .meta { font-size: 11px; color: #59636e; margin-top: 6px; }
  .page-card .meta b { color: #1f2328; font-weight: 600; }
  .page-card .stratum { background: #0969da; color: white; padding: 1px 8px;
                         border-radius: 3px; display: inline-block; font-size: 10px;
                         font-weight: 700; }
  .page-card .right { display: flex; flex-direction: column; gap: 8px; }
  .qcard { border: 1px solid #d1d9e0; border-radius: 5px; padding: 9px 12px;
           background: #f6f8fa; transition: border-color 0.1s; }
  .qcard.approved { border-color: #1a7f37; }
  .qcard.rejected { border-color: #cf222e; opacity: 0.55; }
  .qcard.edited { border-color: #9a6700; }
  .qcard .qhead { display: flex; gap: 8px; align-items: center; margin-bottom: 4px; }
  .qcard .qid { font-family: "SF Mono", monospace; font-size: 10px; color: #59636e; }
  .qcard .caps span { font-size: 9px; padding: 0 5px; border-radius: 2px;
                       background: #d1d9e0; color: #1f2328; margin-right: 3px; }
  .qcard .q { font-weight: 600; margin: 4px 0; }
  .qcard .rf { font-size: 11px; color: #59636e; font-family: "SF Mono", monospace; }
  .qcard .gold { font-family: "SF Mono", monospace; font-size: 11px; background: #ffffff;
                 padding: 4px 8px; border-radius: 3px; border: 1px solid #d1d9e0;
                 margin-top: 4px; word-break: break-all; }
  .qcard .gold.editing { background: #f6f8fa; }
  .qcard textarea.edit-gold { width: 100%; min-height: 60px; background: #ffffff;
                               color: #1f2328; border: 1px solid #d1d9e0; border-radius: 3px;
                               padding: 4px 8px; font-family: "SF Mono", monospace;
                               font-size: 11px; }
  .qcard .actions { display: flex; gap: 4px; margin-top: 6px; }
  .qcard .actions button { font-size: 10px; padding: 3px 8px; cursor: pointer;
                           background: #f6f8fa; color: #1f2328; border: 1px solid #d1d9e0;
                           border-radius: 3px; }
  .qcard .actions button:hover { background: #d1d9e0; }
  .qcard .actions button.b-app { color: #1a7f37; }
  .qcard .actions button.b-rej { color: #cf222e; }
  .qcard .actions button.b-edit { color: #9a6700; }
  .qcard .actions button.b-save { color: #0969da; }
  .qcard.approved .actions button.b-app { background: #1a7f37; color: #fff; font-weight: 700; }
  .qcard.rejected .actions button.b-rej { background: #cf222e; color: #fff; font-weight: 700; }

  /* POST-EVAL extras */
  .difficulty { font-size: 10px; padding: 1px 6px; border-radius: 3px; font-weight: 700;
                color: #fff; }
  /* difficulty buckets: 0..N parsers correct → gray→red→orange→amber→green */
  .d-0 { background: #6e7781; }   /* all parsers wrong = hard */
  .d-1 { background: #cf222e; }
  .d-2 { background: #bc4c00; }
  .d-3 { background: #9a6700; }
  .d-4 { background: #bf8700; }
  .d-5 { background: #2da44e; }
  .d-6 { background: #1a7f37; }   /* all parsers right = easy */

  /* Overall scoreboard */
  #scoreboard { border: 1px solid #d1d9e0; border-radius: 8px; margin-bottom: 18px;
                background: #f6f8fa; padding: 12px 16px; }
  #scoreboard h2 { font-size: 11px; margin: 0 0 10px; color: #59636e;
                   text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; }
  #scoreboard table { border-collapse: collapse; font-size: 12px; min-width: 540px; }
  #scoreboard th, #scoreboard td { padding: 5px 14px; text-align: right;
                   border-bottom: 1px solid #d1d9e0; white-space: nowrap; }
  #scoreboard th { color: #59636e; font-weight: 600; font-size: 10px;
                   text-transform: uppercase; }
  #scoreboard td.sb-parser { text-align: left; font-weight: 600; }
  #scoreboard td.sb-rank { text-align: center; color: #59636e; width: 26px; }
  #scoreboard tr.sb-top td { background: #dafbe1; }
  #scoreboard .sb-pct { font-variant-numeric: tabular-nums; font-weight: 700;
                   display: inline-block; min-width: 38px; }
  #scoreboard .sb-frac { color: #59636e; font-size: 10px;
                   font-family: "SF Mono", monospace; }
  #scoreboard .sb-bar-wrap { display: inline-block; width: 120px; height: 9px;
                   background: #eaeef2; border-radius: 2px; vertical-align: middle;
                   margin-left: 8px; overflow: hidden; }
  #scoreboard .sb-bar { display: block; height: 100%; background: #0969da; }

  /* Iteration-history dashboard panel */
  #iter-history { border: 1px solid #d1d9e0; border-radius: 8px; margin-bottom: 18px;
                  background: #fff; padding: 12px 16px; }
  #iter-history h2 { font-size: 11px; margin: 0 0 4px; color: #59636e;
                     text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; }
  #iter-history .ih-note { font-size: 11px; color: #59636e; margin-bottom: 10px; }
  #iter-history table { border-collapse: collapse; font-size: 12px; }
  #iter-history th, #iter-history td { padding: 5px 12px; text-align: right;
                     border-bottom: 1px solid #d1d9e0; white-space: nowrap;
                     font-variant-numeric: tabular-nums; }
  #iter-history th { color: #59636e; font-weight: 600; font-size: 10px;
                     text-transform: uppercase; }
  #iter-history td.ih-parser { text-align: left; font-weight: 600; }
  #iter-history td.ih-iter0 { color: #59636e; }
  #iter-history .ih-delta-up { color: #1a7f37; font-size: 10px; }
  #iter-history .ih-delta-dn { color: #cf222e; font-size: 10px; }
  #iter-history tr.ih-top td { background: #dafbe1; }
  /* Side-by-side parser answer grid (4 columns) */
  .parser-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                 gap: 6px; margin-top: 8px; }
  .parser-cell { border: 1px solid #d1d9e0; border-radius: 4px; padding: 6px 8px;
                 background: #ffffff; display: flex; flex-direction: column; gap: 3px; }
  .parser-cell.correct   { border-color: #1a7f37; background: #dafbe1; }
  .parser-cell.incorrect { border-color: #cf222e; background: #ffebe9; }
  .parser-cell .phead { display: flex; align-items: center; gap: 5px;
                        font-size: 10px; color: #59636e; font-family: "SF Mono", monospace; }
  .parser-cell .ok { font-weight: 700; font-size: 12px; }
  .parser-cell .ok.y { color: #1a7f37; }
  .parser-cell .ok.n { color: #cf222e; }
  .parser-cell .ans { font-family: "SF Mono", monospace; font-size: 11px;
                       color: #1f2328; word-break: break-word; white-space: pre-wrap;
                       max-height: 120px; overflow-y: auto; }
  .gold-row { background: #ddf4ff; border: 1px solid #0969da; border-radius: 4px;
              padding: 6px 9px; margin-top: 6px; font-family: "SF Mono", monospace;
              font-size: 11px; color: #1f2328; white-space: pre-wrap; word-break: break-word; }
  .gold-row::before { content: "GOLD"; display: inline-block; background: #0969da;
                       color: white; font-weight: 700; padding: 0 5px; border-radius: 2px;
                       margin-right: 6px; font-size: 9px; vertical-align: middle; }
  /* Per-sub-answer matrix: rows = answer keys, cols = GOLD + parsers */
  .kv-matrix { border-collapse: collapse; margin-top: 8px; font-size: 11px;
               width: 100%; table-layout: auto; }
  .kv-matrix th, .kv-matrix td { border: 1px solid #d1d9e0; padding: 4px 7px;
               text-align: left; font-family: "SF Mono", monospace;
               vertical-align: top; word-break: break-word; }
  .kv-matrix th { background: #f6f8fa; color: #59636e; font-weight: 600;
               font-size: 10px; white-space: nowrap; }
  .kv-matrix td.kv-key { color: #9a6700; font-weight: 700; white-space: nowrap; }
  .kv-matrix td.kv-gold { background: #ddf4ff; color: #1f2328; font-weight: 600; }
  .kv-matrix td.cell-ok { background: #dafbe1; color: #1f2328; }
  .kv-matrix td.cell-no { background: #ffebe9; color: #1f2328; }
  .kv-matrix .mk { font-weight: 700; margin-right: 4px; }
  .kv-matrix .mk.y { color: #1a7f37; }
  .kv-matrix .mk.n { color: #cf222e; }
  .kv-matrix tr.key-hard td.kv-key { color: #cf222e; }
  /* show full parser markdowns per page (collapsible) */
  .md-toggle { font-size: 10px; padding: 3px 8px; background: #f6f8fa; color: #0969da;
               border: 1px solid #d1d9e0; border-radius: 3px; cursor: pointer; margin-top: 8px; }
  .md-grid { display: none; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
             gap: 6px; margin-top: 8px; }
  .md-grid.show { display: grid; }
  .md-cell { border: 1px solid #d1d9e0; border-radius: 4px; padding: 6px 8px;
             background: #ffffff; font-family: "SF Mono", monospace; font-size: 10px;
             color: #1f2328; white-space: pre-wrap; word-break: break-word;
             max-height: 400px; overflow-y: auto; }
  .md-cell .ph { font-weight: 700; color: #0969da; margin-bottom: 4px;
                 font-size: 11px; position: sticky; top: 0; background: #ffffff;
                 padding-bottom: 3px; border-bottom: 1px solid #d1d9e0; }

  #export-panel { position: fixed; top: 5vh; left: 50%; transform: translateX(-50%);
                  background: #f6f8fa; border: 1px solid #d1d9e0; border-radius: 8px;
                  padding: 20px; width: 90vw; max-width: 800px; max-height: 90vh;
                  overflow: auto; z-index: 100; display: none; box-shadow: 0 8px 24px rgba(0,0,0,0.7); }
  #export-panel.show { display: block; }
  #export-panel textarea { width: 100%; min-height: 50vh; background: #ffffff;
                           color: #1f2328; border: 1px solid #d1d9e0; border-radius: 3px;
                           padding: 8px; font-family: "SF Mono", monospace; font-size: 11px; }
  #overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 99;
             display: none; }
  #overlay.show { display: block; }
  #zoom { position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 200;
          display: none; align-items: center; justify-content: center; cursor: zoom-out; }
  #zoom.show { display: flex; }
  #zoom img { max-width: 95vw; max-height: 95vh; box-shadow: 0 8px 24px rgba(0,0,0,0.7); }
</style></head><body>"""

    BODY = """
<div class="sticky-bar">
  <h1 style="margin:0">{run} review — {mode}</h1>
  <span class="counts">
    <span class="c-app">Q approved <span id="n-app">0</span></span>
    <span class="c-rej">Q rejected <span id="n-rej">0</span></span>
    <span class="c-und">Q undecided <span id="n-und">0</span></span>
  </span>
  <span class="counts">
    <span class="c-app">doc ✓ <span id="n-doc-app">0</span></span>
    <span class="c-rej">doc ✗ <span id="n-doc-rej">0</span></span>
  </span>
  <span style="color:#59636e">/ <span id="n-total">0</span> Qs across <span id="n-pages">0</span> pages</span>

  <label>Filter domain
    <select id="f-domain" onchange="renderAll()">
      <option value="">all</option>
      <option>finance</option><option>medical_healthcare</option>
    </select>
  </label>
  <label>Filter stratum
    <select id="f-stratum" onchange="renderAll()">
      <option value="">all</option>
    </select>
  </label>
  <label>Question state
    <select id="f-state" onchange="renderAll()">
      <option value="">all</option>
      <option value="undecided">undecided</option>
      <option value="approved">approved</option>
      <option value="rejected">rejected</option>
      <option value="edited">edited</option>
    </select>
  </label>
  <label id="f-diff-wrap" style="display:none">Difficulty
    <select id="f-diff" onchange="renderAll()">
      <option value="">all</option>
      <option value="0">0 (all wrong)</option>
      <option value="1">1</option>
      <option value="2">2</option>
      <option value="3">3</option>
      <option value="4">4</option>
      <option value="5">5</option>
      <option value="6">6 (all right)</option>
    </select>
  </label>
  <label id="f-scoring-wrap" style="display:none">Scoring
    <select id="f-scoring" onchange="renderAll()">
      <option value="llm">LLM-judge (recommended)</option>
      <option value="fuzzy">fuzzy (value-level)</option>
      <option value="strict">strict (exact match)</option>
    </select>
  </label>
  <label>Doc state
    <select id="f-doc-state" onchange="renderAll()">
      <option value="">all</option>
      <option value="undecided">undecided</option>
      <option value="approved">approved</option>
      <option value="rejected">rejected</option>
    </select>
  </label>
  <label id="f-sort-wrap" style="display:none">Sort
    <select id="f-sort" onchange="renderAll()">
      <option value="hardest">hardest first</option>
      <option value="easiest">easiest first</option>
      <option value="disagreement">most parser disagreement</option>
      <option value="alpha">alphabetical</option>
    </select>
  </label>
  <button class="primary" onclick="exportDecisions()">Export ↓</button>
</div>

<div id="iter-history" style="display:none"></div>

<div id="scoreboard" style="display:none"></div>

<div id="cards"></div>

<div id="overlay" onclick="closeExport()"></div>
<div id="export-panel">
  <h3 style="margin:0 0 12px">Decisions JSON (paste back to Claude)</h3>
  <textarea id="export-text" readonly></textarea>
  <div style="margin-top:8px">
    <button class="primary" onclick="copyExport()">Copy</button>
    <button onclick="closeExport()">Close</button>
  </div>
</div>
<div id="zoom" onclick="closeZoom()"><img id="zoom-img"></div>
"""

    SCRIPT_TMPL = """
<script>
const PAGES = __PAGES__;
const MODE = "__MODE__";
const RUN = "__RUN__";
const ITER_HISTORY = __ITER_HISTORY__;
const KEY = "rdb-decisions-" + RUN + "-v2";
const DOC_KEY = "rdb-doc-decisions-" + RUN + "-v2";
let DEC = JSON.parse(localStorage.getItem(KEY) || "{}");
let DOC_DEC = JSON.parse(localStorage.getItem(DOC_KEY) || "{}");
// shape per qid:  { action: "approved"|"rejected"|"edited", edited_gold: "...", note: "..." }
// shape per stem: { action: "approved"|"rejected", note: "..." }

function save() { localStorage.setItem(KEY, JSON.stringify(DEC)); updateCounts(); }
function saveDoc() { localStorage.setItem(DOC_KEY, JSON.stringify(DOC_DEC)); updateCounts(); }
function setAction(qid, action) {
  DEC[qid] = DEC[qid] || {};
  DEC[qid].action = action;
  save();
  applyCard(qid);
}
function setDoc(stem, action) {
  DOC_DEC[stem] = DOC_DEC[stem] || {};
  // toggle: clicking the active state again clears it
  if (DOC_DEC[stem].action === action) {
    delete DOC_DEC[stem];
  } else {
    DOC_DEC[stem].action = action;
  }
  saveDoc();
  applyDocCard(stem);
}
function applyDocCard(stem) {
  const el = document.querySelector(`[data-stem="${stem}"]`);
  if (!el) return;
  const d = DOC_DEC[stem] || {};
  el.classList.remove("doc-approved", "doc-rejected");
  if (d.action === "approved") el.classList.add("doc-approved");
  if (d.action === "rejected") el.classList.add("doc-rejected");
  const status = el.querySelector(".doc-status");
  if (status) status.textContent = d.action ? `doc: ${d.action}` : "";
}
function setEdit(qid, newGold) {
  DEC[qid] = DEC[qid] || {};
  DEC[qid].action = "edited";
  DEC[qid].edited_gold = newGold;
  save();
  applyCard(qid);
}
function applyCard(qid) {
  const el = document.getElementById("q-" + qid);
  if (!el) return;
  const d = DEC[qid] || {};
  el.className = "qcard" + (d.action ? " " + d.action : "");
}

// stratum dropdown
const strataSet = new Set();
PAGES.forEach(p => strataSet.add(p.stratum));
const fStratum = document.getElementById("f-stratum");
[...strataSet].sort().forEach(s => {
  const o = document.createElement("option");
  o.value = s; o.textContent = s;
  fStratum.appendChild(o);
});

if (MODE === "post") {
  document.getElementById("f-diff-wrap").style.display = "inline-block";
  document.getElementById("f-scoring-wrap").style.display = "inline-block";
  document.getElementById("f-sort-wrap").style.display = "inline-block";
}

// Per-doc difficulty signal: average sub-answer correctness (under active
// scoring) across every (key, parser) cell + a disagreement proxy = sum of
// (distinct parser answers − 1) per sub-answer key.
function docStats(p) {
  if (!p.eval || !p.eval.questions) return { avg: null, disagree: 0, nQs: p.questions.length };
  const subKey = "sub_correct_" + activeScoring();
  let sumCorrect = 0, nCells = 0, disagree = 0;
  for (const q of p.questions) {
    const qev = p.eval.questions[q.question_id];
    if (!qev) continue;
    const keys = qev.keys || Object.keys(qev.gold_struct || {});
    const sub = qev[subKey] || {};
    const pa = qev.parser_answers || {};
    const parsers = Object.keys(pa);
    if (parsers.length === 0 || keys.length === 0) continue;
    for (const k of keys) {
      const okSet = sub[k] || [];
      for (const pn of parsers) {
        nCells += 1;
        if (okSet.includes(pn)) sumCorrect += 1;
      }
      // disagreement among parser answers for this sub-answer
      const seen = new Set();
      for (const pn of parsers) {
        const v = (pa[pn] || {})[k];
        if (v != null) seen.add(String(v).trim().toLowerCase().replace(/\\s+/g, " "));
      }
      if (seen.size > 1) disagree += seen.size - 1;
    }
  }
  return { avg: nCells ? sumCorrect / nCells : null, disagree, nQs: p.questions.length };
}

function activeScoring() {
  const sel = document.getElementById("f-scoring");
  return sel ? sel.value : "llm";
}

function correctSetKey() {
  const s = activeScoring();
  return s === "strict" ? "parsers_correct_strict"
       : s === "fuzzy"  ? "parsers_correct_fuzzy"
       : "parsers_correct_llm";
}

function getDifficulty(qid) {
  // count parsers correct under the ACTIVE scoring strategy
  const key = correctSetKey();
  for (const p of PAGES) {
    for (const q of p.questions) {
      if (q.question_id !== qid) continue;
      const ev = p.eval || {};
      const qev = (ev.questions || {})[qid];
      if (!qev) return null;
      const correct = qev[key] || qev.parsers_correct || [];
      return correct.length;
    }
  }
  return null;
}

function renderAll() {
  const domF = document.getElementById("f-domain").value;
  const strF = document.getElementById("f-stratum").value;
  const stateF = document.getElementById("f-state").value;
  const diffF = MODE === "post" ? document.getElementById("f-diff").value : "";
  const docStateF = document.getElementById("f-doc-state").value;
  const sortMode = MODE === "post"
    ? (document.getElementById("f-sort").value || "hardest")
    : "alpha";

  // Decorate + sort pages by selected mode
  const pages = PAGES.slice();
  if (MODE === "post" && sortMode !== "alpha") {
    pages.forEach(p => p.__stats = docStats(p));
    if (sortMode === "hardest") {
      pages.sort((a, b) => (a.__stats.avg ?? 1) - (b.__stats.avg ?? 1));
    } else if (sortMode === "easiest") {
      pages.sort((a, b) => (b.__stats.avg ?? 0) - (a.__stats.avg ?? 0));
    } else if (sortMode === "disagreement") {
      pages.sort((a, b) => b.__stats.disagree - a.__stats.disagree);
    }
  } else {
    pages.sort((a, b) => a.dest_path.localeCompare(b.dest_path));
  }

  const root = document.getElementById("cards");
  root.innerHTML = "";

  let totalQ = 0, totalP = 0;
  const visiblePages = [];
  for (const p of pages) {
    if (domF && p.domain !== domF) continue;
    if (strF && p.stratum !== strF) continue;
    // doc state filter
    const docAction = (DOC_DEC[p.stem] || {}).action || "undecided";
    if (docStateF && docAction !== docStateF) continue;

    const qs = p.questions.filter(q => {
      const d = DEC[q.question_id] || {};
      const action = d.action || "undecided";
      if (stateF && action !== stateF) return false;
      if (diffF !== "") {
        const diff = getDifficulty(q.question_id);
        if (diff === null || diff !== parseInt(diffF)) return false;
      }
      return true;
    });
    if (!qs.length) continue;

    totalP += 1;
    totalQ += qs.length;
    visiblePages.push(p);
    const card = document.createElement("div");
    card.className = "page-card";
    card.dataset.stem = p.stem;
    if (docAction === "approved") card.classList.add("doc-approved");
    if (docAction === "rejected") card.classList.add("doc-rejected");
    const stats = (MODE === "post") ? (p.__stats || docStats(p)) : null;
    const hardnessBadge = (stats && stats.avg !== null)
      ? `<span class="doc-hardness">avg correct: ${(stats.avg * 100).toFixed(0)}% · disagree: ${stats.disagree}</span>`
      : "";
    const docActionsRow = `
      <div class="doc-actions">
        <button class="d-app" onclick="setDoc('${p.stem}', 'approved')">✓ approve doc</button>
        <button class="d-rej" onclick="setDoc('${p.stem}', 'rejected')">✗ reject doc</button>
        ${hardnessBadge}
        <span class="doc-status">${docAction !== 'undecided' ? `doc: ${docAction}` : ''}</span>
      </div>`;
    const imgUrl = "images/" + p.stem + ".png";
    // build page-level "show full parser outputs side by side" toggle
    let mdToggle = "";
    if (MODE === "post" && p.parser_mds && Object.keys(p.parser_mds).length) {
      const labels = {
        "extend_performance_v1_0_0": "Extend v1",
        "extend_performance_v2_0_0_advanced": "Extend v2 (adv)",
        "extend_performance_v2_0_0_agentic": "Extend v2 (agentic)",
        "aws_textract": "Textract",
        "azure_di": "Azure DI",
        "reducto": "Reducto",
        "llamaparse": "LlamaParse", "llamaparse_cost_effective": "LlamaParse CE",
      };
      const order = ["extend_performance_v1_0_0", "extend_performance_v2_0_0_advanced",
                     "extend_performance_v2_0_0_agentic",
                     "aws_textract", "azure_di", "reducto", "llamaparse", "llamaparse_cost_effective"];
      const cells = order.filter(pn => p.parser_mds[pn] !== undefined)
                        .map(pn => `<div class="md-cell"><div class="ph">${labels[pn]}</div>${escapeHtml(p.parser_mds[pn])}</div>`)
                        .join("");
      const mdId = "md-" + p.stem.replace(/[^a-z0-9]/gi, "_");
      mdToggle = `
        <button class="md-toggle" onclick="document.getElementById('${mdId}').classList.toggle('show')">
          show / hide full parser outputs side-by-side
        </button>
        <div class="md-grid" id="${mdId}">${cells}</div>`;
    }
    card.innerHTML = `
      <div class="left">
        <img src="${imgUrl}" onclick="zoom('${imgUrl}')">
        <div class="meta">
          <span class="stratum">${p.stratum}</span><br>
          <b>type</b> ${p.type}<br>
          <b>domain</b> ${p.domain} · <b>kind</b> ${p.kind}<br>
          <b>category</b> ${p.category} · <b>density</b> ${p.density}<br>
          <b>page</b> ${p.dest_path}
        </div>
        ${docActionsRow}
      </div>
      <div class="right">
        ${qs.map(q => renderQCard(q, p)).join("")}
        ${mdToggle}
      </div>`;
    root.appendChild(card);
  }
  document.getElementById("n-pages").textContent = totalP;
  // Apply decisions to rendered cards
  for (const p of PAGES) for (const q of p.questions) applyCard(q.question_id);
  renderScoreboard(visiblePages);
}

// Iteration-history panel — rendered once on load. Shows how each parser's
// accuracy moved as questions were iteratively hardened.
function renderIterationHistory() {
  const ih = document.getElementById("iter-history");
  if (!ITER_HISTORY || !ITER_HISTORY.length) { ih.style.display = "none"; return; }
  ih.style.display = "block";
  const parserOrder = ["extend_performance_v1_0_0", "extend_performance_v2_0_0_advanced",
                       "extend_performance_v2_0_0_agentic", "aws_textract", "azure_di", "reducto", "llamaparse", "llamaparse_cost_effective"];
  const labels = {
    "extend_performance_v1_0_0": "Extend v1",
    "extend_performance_v2_0_0_advanced": "Extend v2 (adv)",
        "extend_performance_v2_0_0_agentic": "Extend v2 (agentic)",
    "aws_textract": "Textract", "azure_di": "Azure DI",
    "reducto": "Reducto", "llamaparse": "LlamaParse", "llamaparse_cost_effective": "LlamaParse CE",
  };
  const metric = "sub_llm";   // sub-answer accuracy, LLM-judge
  const pct = (s) => s && s.sub_t ? (s[metric] / s.sub_t * 100) : null;
  let html = `<h2>Question-hardening progression</h2>`;
  html += `<div class="ih-note">Per-parser sub-answer accuracy (LLM-judge) after each `
        + `hardening pass. Easy questions — ones ≥5/6 parsers nailed — were replaced `
        + `with harder ones generated from the page image.</div>`;
  html += `<table><tr><th>Parser</th>`
        + ITER_HISTORY.map(h => `<th>${h.label || ("iter " + h.iter)}<br>`
            + `<span style="font-weight:400">${h.n_questions} Qs</span></th>`).join("")
        + `<th>net Δ</th></tr>`;
  // rank parsers by final-iteration accuracy
  const last = ITER_HISTORY[ITER_HISTORY.length - 1];
  const ranked = parserOrder.filter(p => last.scores && last.scores[p])
    .sort((a, b) => (pct(last.scores[b]) ?? 0) - (pct(last.scores[a]) ?? 0));
  ranked.forEach((p, i) => {
    html += `<tr class="${i === 0 ? "ih-top" : ""}"><td class="ih-parser">${labels[p]}</td>`;
    let first = null, lastv = null;
    ITER_HISTORY.forEach((h, hi) => {
      const v = pct((h.scores || {})[p]);
      if (v === null) { html += `<td>—</td>`; return; }
      if (first === null) first = v;
      lastv = v;
      let delta = "";
      if (hi > 0) {
        const prev = pct((ITER_HISTORY[hi - 1].scores || {})[p]);
        if (prev !== null) {
          const d = v - prev;
          delta = ` <span class="${d >= 0 ? "ih-delta-up" : "ih-delta-dn"}">`
                + `${d >= 0 ? "+" : ""}${d.toFixed(1)}</span>`;
        }
      }
      html += `<td class="${hi === 0 ? "ih-iter0" : ""}">${v.toFixed(1)}%${delta}</td>`;
    });
    const net = (first !== null && lastv !== null) ? lastv - first : 0;
    html += `<td class="${net >= 0 ? "ih-delta-up" : "ih-delta-dn"}">`
          + `${net >= 0 ? "+" : ""}${net.toFixed(1)}</td></tr>`;
  });
  html += `</table>`;
  ih.innerHTML = html;
}

// Overall scores across the currently-visible (filtered) docs — recomputed on
// every renderAll so it tracks the domain/stratum/state filters live.
function renderScoreboard(visiblePages) {
  const sb = document.getElementById("scoreboard");
  if (MODE !== "post") { sb.style.display = "none"; return; }
  const layer = activeScoring();
  const parserOrder = ["extend_performance_v1_0_0", "extend_performance_v2_0_0_advanced",
                       "extend_performance_v2_0_0_agentic", "aws_textract", "azure_di", "reducto", "llamaparse", "llamaparse_cost_effective"];
  const labels = {
    "extend_performance_v1_0_0": "Extend v1",
    "extend_performance_v2_0_0_advanced": "Extend v2 (adv)",
        "extend_performance_v2_0_0_agentic": "Extend v2 (agentic)",
    "aws_textract": "Textract",
    "azure_di": "Azure DI",
    "reducto": "Reducto",
    "llamaparse": "LlamaParse", "llamaparse_cost_effective": "LlamaParse CE",
  };
  const stat = {};
  for (const pn of parserOrder) stat[pn] = { subC: 0, subT: 0, qC: 0, qT: 0 };
  let nQ = 0;
  for (const p of visiblePages) {
    if (!p.eval || !p.eval.questions) continue;
    for (const q of p.questions) {
      const qev = p.eval.questions[q.question_id];
      if (!qev) continue;
      nQ += 1;
      const keys = qev.keys || Object.keys(qev.gold_struct || {});
      const subC = qev["sub_correct_" + layer] || {};
      const qC = qev["parsers_correct_" + layer] || qev.parsers_correct || [];
      for (const pn of parserOrder) {
        if (!(qev.parser_answers || {})[pn]) continue;
        stat[pn].qT += 1;
        if (qC.includes(pn)) stat[pn].qC += 1;
        for (const k of keys) {
          stat[pn].subT += 1;
          if ((subC[k] || []).includes(pn)) stat[pn].subC += 1;
        }
      }
    }
  }
  const rows = parserOrder.filter(pn => stat[pn].subT > 0)
    .sort((a, b) => (stat[b].subC / stat[b].subT) - (stat[a].subC / stat[a].subT));
  if (!rows.length) { sb.style.display = "none"; return; }
  sb.style.display = "block";
  let html = `<h2>Overall scores — ${layer} · ${visiblePages.length} docs · ${nQ} questions</h2>`;
  html += `<table><tr><th></th><th>Parser</th>`
        + `<th>Sub-answer accuracy</th><th>Question accuracy</th></tr>`;
  rows.forEach((pn, i) => {
    const s = stat[pn];
    const subPct = s.subT ? s.subC / s.subT * 100 : 0;
    const qPct = s.qT ? s.qC / s.qT * 100 : 0;
    html += `<tr class="${i === 0 ? "sb-top" : ""}">
      <td class="sb-rank">${i + 1}</td>
      <td class="sb-parser">${labels[pn]}</td>
      <td><span class="sb-pct">${subPct.toFixed(1)}%</span>
          <span class="sb-frac">${s.subC}/${s.subT}</span>
          <span class="sb-bar-wrap"><span class="sb-bar" style="width:${subPct}%"></span></span></td>
      <td><span class="sb-pct">${qPct.toFixed(1)}%</span>
          <span class="sb-frac">${s.qC}/${s.qT}</span></td>
    </tr>`;
  });
  html += `</table>`;
  sb.innerHTML = html;
}

function renderQCard(q, p) {
  const d = DEC[q.question_id] || {};
  const action = d.action || "";
  const editedGold = d.edited_gold;
  const displayGold = editedGold !== undefined ? editedGold : q.gold_answer;
  const caps = (q.capabilities || []).map(c => `<span>${c}</span>`).join("");

  let evalHtml = "";
  if (MODE === "post" && p.eval) {
    const qev = (p.eval.questions || {})[q.question_id];
    if (qev) {
      const layer = activeScoring();
      const qLevelSet = qev["parsers_correct_" + layer] || qev.parsers_correct || [];
      const subCorrect = qev["sub_correct_" + layer] || {};
      const goldStruct = qev.gold_struct || {};
      const keys = qev.keys || Object.keys(goldStruct);
      const parserOrder = ["extend_performance_v1_0_0", "extend_performance_v2_0_0_advanced",
                           "extend_performance_v2_0_0_agentic", "aws_textract", "azure_di", "reducto", "llamaparse", "llamaparse_cost_effective"];
      const labels = {
        "extend_performance_v1_0_0": "Extend v1",
        "extend_performance_v2_0_0_advanced": "Extend v2 (adv)",
        "extend_performance_v2_0_0_agentic": "Extend v2 (agentic)",
        "aws_textract": "Textract",
        "azure_di": "Azure DI",
        "reducto": "Reducto",
        "llamaparse": "LlamaParse", "llamaparse_cost_effective": "LlamaParse CE",
      };
      const present = parserOrder.filter(pn => (qev.parser_answers || {})[pn] !== undefined);
      const nParsers = present.length || 6;
      const diff = qLevelSet.length;

      evalHtml = `<div style="margin-top:6px">
        <span class="difficulty d-${diff}">${diff}/${nParsers} parsers fully correct
        · ${keys.length} sub-answer${keys.length===1?"":"s"} (${layer})</span></div>`;

      // key × parser matrix
      let head = `<tr><th>sub-answer</th><th>GOLD</th>` +
        present.map(pn => `<th>${labels[pn]}</th>`).join("") + `</tr>`;
      let rows = "";
      for (const k of keys) {
        const gv = goldStruct[k];
        const okSet = subCorrect[k] || [];
        const keyHard = okSet.length === 0 ? " key-hard" : "";
        let cells = present.map(pn => {
          const av = ((qev.parser_answers || {})[pn] || {})[k];
          const ok = okSet.includes(pn);
          const txt = (av === null || av === undefined) ? "∅ null" : String(av);
          return `<td class="${ok?"cell-ok":"cell-no"}">` +
                 `<span class="mk ${ok?"y":"n"}">${ok?"✓":"✗"}</span>${escapeHtml(txt)}</td>`;
        }).join("");
        rows += `<tr class="${keyHard.trim()}"><td class="kv-key">${escapeHtml(k)}</td>` +
                `<td class="kv-gold">${escapeHtml(gv===null||gv===undefined?"null":String(gv))}</td>` +
                `${cells}</tr>`;
      }
      evalHtml += `<table class="kv-matrix"><thead>${head}</thead><tbody>${rows}</tbody></table>`;
    }
  }

  return `<div class="qcard ${action}" id="q-${q.question_id}">
    <div class="qhead">
      <span class="qid">${q.question_id}</span>
      <span class="caps">${caps}</span>
    </div>
    <div class="q">${escapeHtml(q.question)}</div>
    <div class="rf">${escapeHtml(q.response_format)}</div>
    <div class="gold" id="gold-${q.question_id}">${escapeHtml(displayGold)}</div>
    <div class="actions">
      <button class="b-app" onclick="setAction('${q.question_id}', 'approved')">✓ approve</button>
      <button class="b-rej" onclick="setAction('${q.question_id}', 'rejected')">✗ reject</button>
      <button class="b-edit" onclick="startEdit('${q.question_id}')">✎ edit gold</button>
    </div>
    ${evalHtml}
  </div>`;
}

function startEdit(qid) {
  const goldDiv = document.getElementById("gold-" + qid);
  const current = goldDiv.textContent;
  goldDiv.innerHTML = `<textarea class="edit-gold" id="edit-${qid}">${escapeHtml(current)}</textarea>`;
  const card = document.getElementById("q-" + qid);
  const actionsDiv = card.querySelector(".actions");
  actionsDiv.innerHTML = `
    <button class="b-save" onclick="saveEdit('${qid}')">💾 save</button>
    <button onclick="cancelEdit('${qid}')">cancel</button>`;
}
function saveEdit(qid) {
  const ta = document.getElementById("edit-" + qid);
  if (!ta) return;
  setEdit(qid, ta.value);
  renderAll();
}
function cancelEdit(qid) {
  renderAll();
}

function updateCounts() {
  let app = 0, rej = 0, edit = 0, und = 0, total = 0;
  for (const p of PAGES) for (const q of p.questions) {
    total += 1;
    const a = (DEC[q.question_id] || {}).action;
    if (a === "approved") app++;
    else if (a === "rejected") rej++;
    else if (a === "edited") edit++;
    else und++;
  }
  let docApp = 0, docRej = 0;
  for (const stem of Object.keys(DOC_DEC)) {
    const a = (DOC_DEC[stem] || {}).action;
    if (a === "approved") docApp++;
    else if (a === "rejected") docRej++;
  }
  document.getElementById("n-app").textContent = app + (edit ? `+${edit}` : "");
  document.getElementById("n-rej").textContent = rej;
  document.getElementById("n-und").textContent = und;
  document.getElementById("n-total").textContent = total;
  document.getElementById("n-doc-app").textContent = docApp;
  document.getElementById("n-doc-rej").textContent = docRej;
}

function exportDecisions() {
  const out = {
    run: RUN, mode: MODE,
    decisions: DEC,
    doc_decisions: DOC_DEC,
  };
  document.getElementById("export-text").value = JSON.stringify(out, null, 2);
  document.getElementById("overlay").classList.add("show");
  document.getElementById("export-panel").classList.add("show");
}
function closeExport() {
  document.getElementById("overlay").classList.remove("show");
  document.getElementById("export-panel").classList.remove("show");
}
function copyExport() {
  const ta = document.getElementById("export-text");
  ta.select(); document.execCommand("copy"); alert("Copied!");
}
function zoom(src) {
  document.getElementById("zoom-img").src = src;
  document.getElementById("zoom").classList.add("show");
}
function closeZoom() {
  document.getElementById("zoom").classList.remove("show");
}
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

renderIterationHistory();
renderAll();
updateCounts();
</script></body></html>
"""
    # RUN id is the run-dir name (e.g. "v500", "picks31", "pilot_50") — used to
    # namespace localStorage so each review surface keeps its own decisions.
    run_id = PILOT.name
    # Optional iteration-history (written by the question-hardening chain).
    hist_path = PILOT / "eval" / "iteration_history.json"
    iter_history_json = (hist_path.read_text() if hist_path.exists() else "[]")
    body = BODY.replace("{mode}", mode).replace("{run}", run_id)
    script = (SCRIPT_TMPL
              .replace("__PAGES__", pages_json)
              .replace("__MODE__", mode)
              .replace("__RUN__", run_id)
              .replace("__ITER_HISTORY__", iter_history_json))
    out_path.write_text(HEAD + body + script)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pre", "post"], required=True)
    ap.add_argument("--run-dir", type=Path, default=REPO / "runs" / "pilot_50")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    _set_run_dir(args.run_dir.resolve())

    pages, eval_data = load_data(args.mode)
    out = (args.out or (PILOT / f"review_{args.mode}.html")).resolve()
    emit_html(pages, args.mode, out)
    try:
        shown = out.relative_to(REPO)
    except ValueError:
        shown = out
    print(f"Wrote {shown} ({sum(len(p['questions']) for p in pages)} Qs across {len(pages)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
