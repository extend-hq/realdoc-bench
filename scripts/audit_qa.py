#!/usr/bin/env python3
"""Benchmark-integrity audit for a run's Q&A bank.

Two independent checks, run in batches of 20 questions:

  CHECK A — hallucination.  For every (parser, sub-answer) the extractor agent
    marked correct, verify the value is actually present in THAT parser's
    markdown. If a parser is scored correct but the value cannot be found in
    its output, the agent likely hallucinated / inferred it — a false positive
    that inflates the parser's score. Conservative: skips null answers and
    interpreted enum values (selected/yes/checkbox) where a literal substring
    test is meaningless.

  CHECK B — gold correctness.  A vision call (Gemini, image only) re-answers
    each question from the page image and judges whether the stored GOLD is
    correct. Verdict: gold_correct / gold_wrong / uncertain.

Anything flagged (wrong gold, suspected hallucination, or uncertain) is
surfaced in audit/needs_review.html for manual inspection + correction.

Phases:
  --phase vision    CHECK B only — independent of eval results, run any time.
  --phase finalize  CHECK A + merge vision cache + build needs_review.html.
  --phase all       both (default).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from pilot_50_eval import gold_to_struct, normalize_exact, format_to_struct  # noqa: E402
from pilot_50_gen_qa import gemini_generate, encoded_image, load_env_file  # noqa: E402

PARSERS = ["extend_performance_v1_0_0", "extend_performance_v2_0_0_advanced",
           "aws_textract", "azure_di", "reducto", "llamaparse"]

# Enum / interpreted values — the literal word is not expected in the markdown
# (the parser emits a symbol like [x]); skip these in the hallucination check.
INTERPRETED = {"selected", "unselected", "yes", "no", "checked", "unchecked",
               "marked", "unmarked", "true", "false", "n/a", "na", "null",
               "none", "blank", "x", "present", "absent"}

# Interpreted tokens (>=3 chars) — if EVERY significant token of an answer is
# one of these, the whole value is an interpretation of checkbox/circle states
# (e.g. "not circled", "(No, null)", "unchecked, unchecked") — not extractable
# verbatim text, so skip it in the hallucination check.
INTERPRETED_TOKENS = {
    "selected", "unselected", "checked", "unchecked", "marked", "unmarked",
    "circled", "uncircled", "yes", "true", "false", "null", "none", "blank",
    "empty", "both", "neither", "not", "present", "absent", "applicable",
    "nil", "filled", "unfilled",
}

AUDIT_SYSTEM = (
    "You are auditing a benchmark question's stored GOLD answer against the "
    "actual document page image. Use ONLY the image — no outside knowledge. "
    "First answer the question yourself by reading the image, then compare "
    "your reading to the provided GOLD.\n\n"
    "Return exactly one JSON object:\n"
    '  {"image_answer": "<what you read from the image, in the question\'s '
    'key=value; format>",\n'
    '   "verdict": "gold_correct" | "gold_wrong" | "uncertain",\n'
    '   "note": "<one sentence; if gold_wrong/uncertain name the key(s) and why>"}\n\n'
    "verdict=gold_correct  — every gold value matches what the image shows.\n"
    "verdict=gold_wrong    — one or more gold values clearly contradict the image.\n"
    "verdict=uncertain     — the image is ambiguous, cropped, or illegible for "
    "the asked field(s)."
)


def norm_blob(s: str) -> str:
    """Aggressive alphanumeric-only normalization for substring presence tests."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def vision_check(image_path: Path, question: str, response_format: str,
                 gold: str) -> dict:
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"REQUIRED ANSWER FORMAT:\n{response_format}\n\n"
        f"STORED GOLD ANSWER (to audit):\n{gold}\n\n"
        "Answer the question from the image, then judge the gold."
    )
    parts = [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/png", "data": encoded_image(image_path)}},
    ]
    try:
        payload = gemini_generate(parts, system=AUDIT_SYSTEM, temperature=0.0)
    except Exception as exc:
        return {"verdict": "error", "image_answer": "", "note": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {"verdict": "error", "image_answer": "", "note": "non-dict response"}
    v = payload.get("verdict")
    if v not in ("gold_correct", "gold_wrong", "uncertain"):
        v = "uncertain"
    return {"verdict": v, "image_answer": str(payload.get("image_answer", "")),
            "note": str(payload.get("note", ""))}


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _tokens(s) -> set[str]:
    """Significant tokens (alphanumeric runs >= 3 chars), comma-stripped so
    '33,900' yields '33900'."""
    return set(_TOKEN_RE.findall(re.sub(r"(?<=\d),(?=\d)", "", str(s).lower())))


def _present(ans: str, gold: str, md: str, md_tokens: set[str]) -> bool:
    """Is the value supported by the markdown? Two complementary tests — pass
    if EITHER holds:
      • contiguous: norm_blob(value) is a substring of norm_blob(md)
        (catches numbers, IDs, exact strings — comma/format agnostic)
      • token coverage: >= 60% of the value's significant tokens appear
        anywhere in the markdown (catches multi-token text split across
        table cells / line breaks)
    Checked for both the agent answer and the gold value.
    """
    blob = norm_blob(md)
    for v in (ans, gold):
        if v is None:
            continue
        nb = norm_blob(v)
        if nb and nb in blob:
            return True
        toks = _tokens(v)
        if toks and sum(1 for t in toks if t in md_tokens) / len(toks) >= 0.60:
            return True
    return False


def check_hallucination(eval_q: dict, markdowns: dict[str, str],
                        fmt_struct: dict[str, str]) -> list[dict]:
    """For each (parser, key) scored correct (LLM-judge), flag it when the
    value is NOT supported by that parser's markdown (see _present).

    Skips null answers and interpreted enum keys (the parser emits a symbol
    like [x], never the word 'selected'). Goal: high precision — a flagged
    item is genuinely worth a human look.
    """
    suspects = []
    gold_struct = eval_q.get("gold_struct", {})
    md_tokens = {p: _tokens(md) for p, md in markdowns.items()}
    for key in eval_q.get("keys", []):
        gold_val = gold_struct.get(key)
        if "|" in str(fmt_struct.get(key, "")):
            continue                              # closed-enum key — skip
        for parser in PARSERS:
            if parser not in eval_q.get("sub_correct_llm", {}).get(key, []):
                continue
            ans = (eval_q.get("parser_answers", {}).get(parser) or {}).get(key)
            if ans is None:
                continue                          # null = legitimately blank
            ans_s = str(ans)
            if ans_s.strip().lower() in INTERPRETED:
                continue
            if any(c in ans_s for c in "={}[]:"):
                continue                          # composite / still-nested value
            atoks = _tokens(ans_s)
            if not atoks and not norm_blob(ans_s):
                continue                          # glyph / dash / blank indicator
            if atoks and all(t in INTERPRETED_TOKENS for t in atoks):
                continue                          # all-interpreted state phrase
            md = markdowns.get(parser, "")
            if not md:
                continue
            if _present(str(ans), gold_val, md, md_tokens.get(parser, set())):
                continue
            strict_ok = parser in eval_q.get("sub_correct_strict", {}).get(key, [])
            suspects.append({
                "parser": parser, "key": key, "answer": ans,
                "gold": gold_val,
                "layer": "strict" if strict_ok else "fuzzy/llm",
            })
    return suspects


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=REPO / "runs" / "v500")
    ap.add_argument("--phase", choices=["vision", "finalize", "all"], default="all")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run = args.run_dir.resolve()
    load_env_file(REPO / ".env")

    bank = json.loads((run / "qa" / "bank_r1.json").read_text())
    items = bank["items"]
    if args.limit:
        items = items[: args.limit]
    sample = {r["dest_path"]: r for r in csv.DictReader((run / "sample.csv").open())}
    stem_to_dest = {Path(r["dest_path"]).stem: r["dest_path"] for r in sample.values()}

    audit_dir = run / "audit"
    vis_dir = audit_dir / "vision"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── CHECK B — vision gold audit (batched, cached per qid) ───────────────
    if args.phase in ("vision", "all"):
        todo = []
        for it in items:
            qid = it["question_id"]
            if (vis_dir / f"{qid}.json").exists():
                continue
            dp = stem_to_dest.get(it["source_file"])
            img = run / "images" / f"{it['source_file']}.png"
            if dp and img.exists():
                todo.append((it, img))
        print(f"CHECK B (vision): {len(todo)} questions to audit "
              f"({len(items)-len(todo)} cached)", flush=True)
        t0 = time.time()
        done = 0
        for b in range(0, len(todo), args.batch_size):
            batch = todo[b:b + args.batch_size]

            def do(pair):
                it, img = pair
                res = vision_check(img, it["question"], it["response_format"],
                                   it["gold_answer"])
                (vis_dir / f"{it['question_id']}.json").write_text(json.dumps(res))
                return res

            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                for fut in as_completed([ex.submit(do, p) for p in batch]):
                    fut.result()
                    done += 1
            print(f"  [{done}/{len(todo)}] vision audited "
                  f"({(time.time()-t0)/60:.1f}min)", flush=True)

    if args.phase == "vision":
        print("vision phase done.", flush=True)
        return 0

    # ── CHECK A — hallucination + merge + report ────────────────────────────
    rescored_path = run / "eval" / "eval_r1_rescored.json"
    if not rescored_path.exists():
        print(f"FATAL: {rescored_path} missing — run eval+rescore first")
        return 1
    eval_data = json.loads(rescored_path.read_text())
    eval_by_qid = {}
    for dp, page in eval_data.items():
        for qid, q in page["questions"].items():
            eval_by_qid[qid] = (dp, q)

    def md_path(parser: str, dp: str) -> Path:
        stem = Path(dp).with_suffix("").as_posix()
        return run / "parses" / parser / f"{stem}.md"

    audit = {}
    n_gold_wrong = n_uncertain = n_halluc = 0
    for it in items:
        qid = it["question_id"]
        if qid not in eval_by_qid:
            continue
        dp, eq = eval_by_qid[qid]
        markdowns = {}
        for p in PARSERS:
            mp = md_path(p, dp)
            if mp.exists():
                markdowns[p] = mp.read_text()
        fmt_struct = format_to_struct(it.get("response_format", ""))
        suspects = check_hallucination(eq, markdowns, fmt_struct)
        vp = vis_dir / f"{qid}.json"
        vision = json.loads(vp.read_text()) if vp.exists() else {
            "verdict": "missing", "image_answer": "", "note": "no vision audit"}

        flags = []
        if vision["verdict"] == "gold_wrong":
            flags.append("gold_suspect"); n_gold_wrong += 1
        if vision["verdict"] in ("uncertain", "error", "missing"):
            flags.append("uncertain"); n_uncertain += 1
        if suspects:
            flags.append("hallucination"); n_halluc += 1

        audit[qid] = {
            "source_file": it["source_file"], "dest_path": dp,
            "question": it["question"], "response_format": it["response_format"],
            "gold": it["gold_answer"], "vision": vision,
            "hallucination_suspects": suspects, "flags": flags,
        }

    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "audit_results.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False))

    flagged = {q: a for q, a in audit.items() if a["flags"]}
    print(f"\n=== audit summary ===", flush=True)
    print(f"  questions audited     : {len(audit)}")
    print(f"  gold suspect (wrong)  : {n_gold_wrong}")
    print(f"  uncertain (vision)    : {n_uncertain}")
    print(f"  hallucination suspect : {n_halluc}")
    print(f"  total flagged         : {len(flagged)}")

    _build_review_html(audit_dir / "needs_review.html", flagged, run.name)
    print(f"  written: {audit_dir / 'needs_review.html'}")
    return 0


def _build_review_html(out: Path, flagged: dict, run_id: str) -> None:
    """Light-theme inspection page for the flagged questions."""
    order = {"gold_suspect": 0, "hallucination": 1, "uncertain": 2}
    rows = sorted(flagged.items(),
                  key=lambda kv: min((order.get(f, 9) for f in kv[1]["flags"]), default=9))
    cards = []
    for qid, a in rows:
        v = a["vision"]
        susp = "".join(
            f"<tr><td>{s['parser']}</td><td>{s['key']}</td>"
            f"<td>{_esc(s['answer'])}</td><td>{_esc(s['gold'])}</td>"
            f"<td>{s['layer']}</td></tr>"
            for s in a["hallucination_suspects"])
        susp_tbl = (f"<table class='susp'><tr><th>parser</th><th>sub-answer</th>"
                    f"<th>agent answer</th><th>gold</th><th>scored</th></tr>"
                    f"{susp}</table>") if susp else ""
        badges = "".join(f"<span class='badge b-{f}'>{f}</span>" for f in a["flags"])
        cards.append(f"""
<div class="card" id="c-{qid}">
  <div class="left"><img src="../images/{a['source_file']}.png"
       onclick="zoom(this.src)"></div>
  <div class="right">
    <div class="qid">{qid} {badges}</div>
    <div class="q">{_esc(a['question'])}</div>
    <div class="rf">{_esc(a['response_format'])}</div>
    <div class="lbl">STORED GOLD</div>
    <div class="gold" id="g-{qid}">{_esc(a['gold'])}</div>
    <div class="lbl">VISION READ THE IMAGE AS — verdict: <b>{v['verdict']}</b></div>
    <div class="vis">{_esc(v.get('image_answer',''))}</div>
    <div class="note">{_esc(v.get('note',''))}</div>
    {susp_tbl}
    <div class="acts">
      <button class="a-ok" onclick="dec('{qid}','gold_ok')">✓ gold ok</button>
      <button class="a-bad" onclick="dec('{qid}','gold_wrong')">✗ gold wrong</button>
      <button class="a-edit" onclick="editGold('{qid}')">✎ edit gold</button>
    </div>
  </div>
</div>""")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{run_id} — Q&A audit</title><style>
body{{font-family:-apple-system,system-ui,sans-serif;background:#fff;color:#1f2328;
margin:0;padding:16px 24px;font-size:13px}}
h1{{font-size:18px}}
.bar{{position:sticky;top:0;background:#f6f8fa;border-bottom:1px solid #d1d9e0;
padding:10px 14px;margin:0 -24px 16px;display:flex;gap:14px;align-items:center}}
.bar button{{background:#0969da;color:#fff;border:0;border-radius:5px;padding:6px 12px;cursor:pointer}}
.card{{display:grid;grid-template-columns:300px 1fr;gap:14px;border:1px solid #d1d9e0;
border-radius:8px;padding:12px;margin-bottom:16px}}
.card.gold_ok{{border-color:#1a7f37;opacity:.55}}
.card.gold_wrong{{border-color:#cf222e}}
.left img{{width:100%;border:1px solid #d1d9e0;cursor:zoom-in}}
.qid{{font-family:monospace;font-size:11px;color:#59636e;margin-bottom:4px}}
.q{{font-weight:600;margin:4px 0}}
.rf{{font-family:monospace;font-size:11px;color:#59636e}}
.lbl{{font-size:10px;color:#59636e;text-transform:uppercase;margin-top:8px;font-weight:700}}
.gold,.vis{{font-family:monospace;font-size:11px;background:#f6f8fa;border:1px solid #d1d9e0;
border-radius:4px;padding:5px 8px;white-space:pre-wrap;word-break:break-word}}
.gold{{background:#ddf4ff}}
.note{{font-size:11px;color:#9a6700;margin-top:4px;font-style:italic}}
.badge{{font-size:9px;font-weight:700;padding:1px 6px;border-radius:3px;margin-left:5px;color:#fff}}
.b-gold_suspect{{background:#cf222e}} .b-hallucination{{background:#bc4c00}}
.b-uncertain{{background:#9a6700}}
.susp{{border-collapse:collapse;font-size:10px;margin-top:8px;font-family:monospace}}
.susp th,.susp td{{border:1px solid #d1d9e0;padding:3px 7px;text-align:left}}
.susp th{{background:#eaeef2}}
.acts{{margin-top:8px;display:flex;gap:6px}}
.acts button{{font-size:11px;padding:4px 10px;border:1px solid #d1d9e0;border-radius:4px;
cursor:pointer;background:#fff}}
.a-ok{{color:#1a7f37}} .a-bad{{color:#cf222e}} .a-edit{{color:#9a6700}}
#zoom{{position:fixed;inset:0;background:rgba(0,0,0,.8);display:none;align-items:center;
justify-content:center;cursor:zoom-out;z-index:99}}
#zoom img{{max-width:95vw;max-height:95vh}}
#zoom.show{{display:flex}}
textarea{{width:100%;font-family:monospace;font-size:11px;min-height:54px}}
</style></head><body>
<div class="bar"><h1 style="margin:0">{run_id} — Q&A audit · {len(rows)} flagged</h1>
<button onclick="exp()">Export decisions ↓</button>
<span id="cnt"></span></div>
{''.join(cards)}
<div id="zoom" onclick="this.classList.remove('show')"><img></div>
<script>
const KEY="audit-{run_id}";
let D=JSON.parse(localStorage.getItem(KEY)||"{{}}");
function save(){{localStorage.setItem(KEY,JSON.stringify(D));
 document.getElementById('cnt').textContent=Object.keys(D).length+' decided';}}
function dec(q,v){{D[q]=D[q]||{{}};D[q].verdict=v;save();
 const c=document.getElementById('c-'+q);c.className='card '+v;}}
function editGold(q){{const g=document.getElementById('g-'+q);
 const cur=g.textContent;g.innerHTML='<textarea id="t-'+q+'">'+cur+'</textarea>'+
 '<button onclick="saveGold(\\''+q+'\\')">save</button>';}}
function saveGold(q){{const t=document.getElementById('t-'+q);
 D[q]=D[q]||{{}};D[q].corrected_gold=t.value;D[q].verdict='gold_wrong';save();
 document.getElementById('g-'+q).textContent=t.value;}}
function zoom(s){{const z=document.getElementById('zoom');
 z.querySelector('img').src=s;z.classList.add('show');}}
function exp(){{const blob=JSON.stringify(D,null,2);
 navigator.clipboard.writeText(blob);alert('Copied '+Object.keys(D).length+
 ' decisions to clipboard — paste back to Claude.');}}
save();
</script></body></html>"""
    out.write_text(html)


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


if __name__ == "__main__":
    raise SystemExit(main())
