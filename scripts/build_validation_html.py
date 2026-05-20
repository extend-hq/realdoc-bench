#!/usr/bin/env python3
"""Human-validation surface for the final atomic bank.

Three-pane review tool: a doc-navigator on the left (ordered by parser
disagreement), the PDF page in the centre, and that doc's atomic questions —
exact gold + every parser's exact prediction — on the right. Keep/reject at
question AND doc level, edit gold inline. Decisions persist in localStorage
(survives refresh) and can be downloaded as a JSON backup / restored.

Usage: build_validation_html.py [run]   ->  runs/<run>/qa/validation.html
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PARSERS = ["extend_performance_v2_0_0_agentic", "extend_performance_v2_0_0_advanced",
           "llamaparse", "llamaparse_cost_effective", "extend_performance_v1_0_0",
           "reducto", "azure_di", "aws_textract"]
LBL = {"extend_performance_v2_0_0_agentic": "Extend v2 agentic",
       "extend_performance_v2_0_0_advanced": "Extend v2 adv", "llamaparse": "LlamaParse",
       "llamaparse_cost_effective": "LlamaParse CE",
       "extend_performance_v1_0_0": "Extend v1", "reducto": "Reducto",
       "azure_di": "Azure DI", "aws_textract": "Textract"}


def main() -> int:
    run = REPO / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "v500")
    bank = json.loads((run / "qa" / "bank_r1.json").read_text())
    ev = json.loads((run / "eval" / "eval_r1_rescored.json").read_text())
    sample = {Path(r["dest_path"]).stem: r
              for r in csv.DictReader((run / "sample.csv").open())}

    eval_q = {}
    for page in ev.values():
        for qid, q in page["questions"].items():
            eval_q[qid] = q

    by_doc: dict[str, list] = defaultdict(list)
    for it in bank["items"]:
        by_doc[it.get("source_file", "?")].append(it)

    pages = []
    for stem in sorted(by_doc):
        meta = sample.get(stem, {})
        qs = []
        for it in by_doc[stem]:
            qid = it["question_id"]
            eq = eval_q.get(qid, {})
            keys = eq.get("keys", ["answer"])
            k0 = keys[0] if keys else "answer"
            preds, norm = {}, set()
            for p in PARSERS:
                ans = (eq.get("parser_answers", {}).get(p) or {}).get(k0)
                preds[p] = {"a": ans, "ok": p in eq.get("sub_correct_llm", {}).get(k0, [])}
                norm.add(re.sub(r"\s+", " ", str(ans).strip().lower())
                         if ans not in (None, "") else "∅")
            prg = it.get("_pre_rewrite_gold")
            gchg = ("_pre_rewrite_gold" in it
                    and str(prg) != str(it.get("gold_answer", "")))
            qs.append({
                "qid": qid, "q": it.get("question", ""),
                "fmt": it.get("response_format", ""),
                "gold": it.get("gold_answer", ""),
                "caps": it.get("capabilities", []),
                "preds": preds, "disagree": len(norm),
                "gchg": gchg, "prevg": prg if gchg else None,
                "prevq": it.get("_pre_rewrite_question", ""),
            })
        qs.sort(key=lambda q: (-q["gchg"], -q["disagree"]))
        pages.append({
            "stem": stem, "type": meta.get("type", ""),
            "domain": meta.get("domain", ""), "kind": meta.get("kind", ""),
            "dest_path": meta.get("dest_path", ""),
            "questions": qs,
            "disagree_total": sum(q["disagree"] - 1 for q in qs),
            "gchg_n": sum(1 for q in qs if q["gchg"]),
        })
    pages.sort(key=lambda p: -p["disagree_total"])

    out = run / "qa" / "validation.html"
    out.write_text(_render(pages, run.name))
    nq = sum(len(p["questions"]) for p in pages)
    print(f"wrote {out}  ({len(pages)} docs, {nq} questions)")
    return 0


def _render(pages: list, run_id: str) -> str:
    return (_TEMPLATE
            .replace("__DATA__", json.dumps(pages, ensure_ascii=False))
            .replace("__PARSERS__", json.dumps(PARSERS))
            .replace("__LABELS__", json.dumps(LBL))
            .replace("__RUN__", run_id))


_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>__RUN__ — human validation</title><style>
*{box-sizing:border-box}
body{font-family:-apple-system,system-ui,sans-serif;margin:0;font-size:13px;color:#1f2328}
.bar{height:46px;display:flex;align-items:center;gap:14px;padding:0 16px;
border-bottom:1px solid #d1d9e0;background:#f6f8fa}
.bar h1{font-size:14px;margin:0;white-space:nowrap}
.bar button,.bar select{background:#fff;border:1px solid #d1d9e0;border-radius:5px;
padding:5px 9px;font-size:12px;cursor:pointer}
.bar button.primary{background:#0969da;color:#fff;border-color:#0969da}
.counts span{padding:1px 7px;border-radius:3px;font-weight:600}
.c-k{background:#1a7f37;color:#fff}.c-r{background:#cf222e;color:#fff}.c-u{background:#eaeef2}
#saved{font-size:11px;color:#1a7f37;font-weight:600;opacity:0;transition:opacity .3s}
.main{display:flex;height:calc(100vh - 47px)}
.nav{width:280px;flex-shrink:0;border-right:1px solid #d1d9e0;overflow-y:auto}
.nrow{padding:6px 9px;border-bottom:1px solid #eaeef2;cursor:pointer;display:flex;
gap:6px;align-items:center;font-size:11px}
.nrow:hover{background:#f6f8fa}
.nrow.sel{background:#ddf4ff}
.nrow.dk{border-left:3px solid #1a7f37}.nrow.dr{border-left:3px solid #cf222e;opacity:.55}
.nrow .ndis{background:#fff1c2;color:#9a6700;padding:0 4px;border-radius:3px;
font-weight:700;font-size:9px;min-width:26px;text-align:center}
.nrow .ndis.hi{background:#cf222e;color:#fff}
.nrow .nnm{flex:1;font-family:monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.nrow .nprog{color:#59636e;font-size:9px;white-space:nowrap}
.pdf{flex:1.05;overflow:auto;background:#eaeef2;padding:14px;text-align:center}
.pdf img{max-width:100%;border:1px solid #d1d9e0;background:#fff;cursor:zoom-in;
box-shadow:0 1px 5px rgba(0,0,0,.18)}
.qs{flex:1.3;overflow-y:auto;padding:0 16px 60px}
.qshead{position:sticky;top:0;background:#fff;padding:10px 0 8px;border-bottom:1px solid #d1d9e0;
display:flex;gap:8px;align-items:center;flex-wrap:wrap;z-index:5}
.qshead .dnm{font-family:monospace;font-size:12px;font-weight:700;flex:1;
overflow:hidden;text-overflow:ellipsis}
.qshead button{font-size:11px;padding:4px 9px;border:1px solid #d1d9e0;border-radius:4px;
background:#fff;cursor:pointer}
.qcard{border:1px solid #d1d9e0;border-radius:6px;padding:9px 11px;margin-top:9px}
.qcard.k{border-color:#1a7f37;background:#f6fff9}
.qcard.r{border-color:#cf222e;background:#fff6f6;opacity:.6}
.qid{font-family:monospace;font-size:10px;color:#59636e}
.cap{font-size:9px;background:#ddf4ff;color:#0969da;padding:0 5px;border-radius:3px;margin-left:4px}
.dis{font-size:9px;background:#fff1c2;color:#9a6700;padding:0 5px;border-radius:3px;
margin-left:4px;font-weight:700}
.dis.hi{background:#cf222e;color:#fff}
.gchg{font-size:9px;background:#9a6700;color:#fff;padding:0 5px;border-radius:3px;
margin-left:4px;font-weight:700}
.wasgold{font-family:monospace;font-size:10px;color:#9a6700;margin-top:3px}
.wasgold s{color:#cf222e}
.c-g{background:#9a6700;color:#fff}
.nrow .ngc{background:#9a6700;color:#fff;padding:0 4px;border-radius:3px;
font-weight:700;font-size:9px}
.qtext{font-weight:600;margin:4px 0}
.tbl{border-collapse:collapse;width:100%;font-size:11px;margin-top:5px}
.tbl td,.tbl th{border:1px solid #d1d9e0;padding:3px 7px;text-align:left;
font-family:monospace;vertical-align:top;word-break:break-word}
.tbl th{background:#eaeef2;font-size:10px}
.tbl tr.gold td{background:#ddf4ff;font-weight:700}
.tbl td.ok{color:#1a7f37;font-weight:700;width:16px;text-align:center}
.tbl td.no{color:#cf222e;font-weight:700;width:16px;text-align:center}
.acts{margin-top:6px;display:flex;gap:5px}
.acts button{font-size:11px;padding:3px 9px;border:1px solid #d1d9e0;border-radius:4px;
background:#fff;cursor:pointer}
.b-k{color:#1a7f37}.b-r{color:#cf222e}.b-e{color:#9a6700}
.gold-ed{width:100%;font-family:monospace;font-size:11px;min-height:40px}
.rawlinks{font-size:10px;color:#59636e;padding:6px 0 2px;border-bottom:1px solid #eaeef2}
.rawlinks b{color:#424a53}
.rawlinks a{color:#0969da;text-decoration:none;margin:0 5px;font-family:monospace}
.rawlinks a:hover{text-decoration:underline}
.rawjson{margin:6px 0;font-size:11px}
.rawjson summary{cursor:pointer;color:#0969da;font-weight:600;padding:3px 0}
.rj-h{font-family:monospace;font-weight:700;font-size:11px;margin-top:6px;color:#424a53}
.rj{background:#f6f8fa;border:1px solid #d1d9e0;border-radius:4px;padding:7px 9px;
margin:3px 0;font-family:monospace;font-size:10px;white-space:pre-wrap;
word-break:break-word;max-height:280px;overflow:auto}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;
justify-content:center;cursor:zoom-out;z-index:99}
#zoom.show{display:flex}#zoom img{max-width:97vw;max-height:97vh}
</style></head><body>
<div class="bar">
  <h1>__RUN__ · validation</h1>
  <span class="counts">
    <span class="c-k">kept <b id="nk">0</b></span>
    <span class="c-r">rej <b id="nr">0</b></span>
    <span class="c-u">left <b id="nu">0</b></span> / <b id="nt">0</b>
    <span class="c-g">gold⚑ <b id="ng">0</b></span>
  </span>
  <label>show <select id="filt" onchange="buildNav()">
    <option value="all">all docs</option>
    <option value="undecided">has undecided</option>
    <option value="goldchg">gold-corrected docs</option>
    <option value="kept">kept docs</option>
    <option value="rejected">rejected docs</option>
  </select></label>
  <button class="primary" onclick="dl()">⬇ Backup</button>
  <button onclick="document.getElementById('rf').click()">⬆ Restore</button>
  <input type="file" id="rf" style="display:none" onchange="restore(this)">
  <span id="saved">✓ saved</span>
</div>
<div class="main">
  <div class="nav" id="nav"></div>
  <div class="pdf" id="pdf"></div>
  <div class="qs" id="qs"></div>
</div>
<div id="zoom" onclick="this.classList.remove('show')"><img></div>
<script>
const PAGES=__DATA__, PARSERS=__PARSERS__, LBL=__LABELS__;
const KEY="valid-__RUN__";
let D=JSON.parse(localStorage.getItem(KEY)||'{"q":{},"doc":{}}');
if(!D.q)D.q={}; if(!D.doc)D.doc={};
let cur=0;

function save(){localStorage.setItem(KEY,JSON.stringify(D));
  const s=document.getElementById('saved');s.style.opacity=1;
  clearTimeout(window._st);window._st=setTimeout(()=>s.style.opacity=0,900);
  counts();}
function counts(){let k=0,r=0,t=0,g=0;
  for(const p of PAGES)for(const q of p.questions){t++;if(q.gchg)g++;
    const v=(D.q[q.qid]||{}).v;if(v==='keep')k++;else if(v==='reject')r++;}
  nk.textContent=k;nr.textContent=r;nu.textContent=t-k-r;nt.textContent=t;
  document.getElementById('ng').textContent=g;}
function esc(s){return String(s==null?'∅ null':s).replace(/&/g,'&amp;')
  .replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function decided(p){let d=0;for(const q of p.questions)if((D.q[q.qid]||{}).v)d++;return d;}

function buildNav(){
  const f=document.getElementById('filt').value;
  const nav=document.getElementById('nav');nav.innerHTML='';
  PAGES.forEach((p,i)=>{
    const dv=(D.doc[p.stem]||{}).v;
    let undec=p.questions.length-decided(p);
    if(f==='undecided'&&undec===0)return;
    if(f==='goldchg'&&!p.gchg_n)return;
    if(f==='kept'&&dv!=='keep')return;
    if(f==='rejected'&&dv!=='reject')return;
    const row=document.createElement('div');
    row.className='nrow'+(i===cur?' sel':'')+(dv==='keep'?' dk':dv==='reject'?' dr':'');
    row.id='nr-'+i;
    row.innerHTML='<span class="ndis'+(p.disagree_total>=10?' hi':'')+'">≠'+
      p.disagree_total+'</span>'+
      (p.gchg_n?'<span class="ngc" title="gold-corrected questions">⚑'+p.gchg_n+'</span>':'')+
      '<span class="nnm">'+p.stem+'</span>'+
      '<span class="nprog" id="np-'+i+'">'+decided(p)+'/'+p.questions.length+'</span>';
    row.onclick=()=>selectDoc(i);
    nav.appendChild(row);
  });
}
function selectDoc(i){
  cur=i;const p=PAGES[i];
  document.querySelectorAll('.nrow.sel').forEach(e=>e.classList.remove('sel'));
  const r=document.getElementById('nr-'+i);if(r){r.classList.add('sel');
    r.scrollIntoView({block:'nearest'});}
  document.getElementById('pdf').innerHTML=
    '<img src="../images/'+p.stem+'.png" onclick="zoom(this.src)">';
  renderQs();
}
function renderQs(){
  const p=PAGES[cur];const dv=(D.doc[p.stem]||{}).v;
  let h='<div class="qshead"><span class="dnm">'+p.stem+'</span>'+
    '<span class="dis">'+(p.type||'')+'</span>'+
    '<button class="b-k" onclick="decDoc(\'keep\')">✓ keep doc</button>'+
    '<button class="b-r" onclick="decDoc(\'reject\')">✗ reject doc</button>'+
    '<button onclick="nextDoc(-1)">← prev doc</button>'+
    '<button onclick="nextDoc(1)">next doc →</button></div>';
  // raw agent output (the extractor LLM's JSON) + parser markdown, per parser
  const dpne=(p.dest_path||'').replace(/\.pdf$/,'');
  let ao='<div class="rawlinks"><b>Agent raw output:</b>';
  for(const pr of PARSERS)
    ao+='<a href="../eval/answers_flat/'+p.stem+'__'+pr+'.json" target="_blank">'+
        LBL[pr]+'</a>';
  ao+='<br><b>Parser markdown:</b>';
  for(const pr of PARSERS)
    ao+='<a href="../parses/'+pr+'/'+dpne+'.md" target="_blank">'+LBL[pr]+'</a>';
  ao+='</div>';
  // exact agent JSON output, per parser (the literal extractor-LLM result)
  let rj='<details class="rawjson"><summary>▸ Exact agent JSON output (per parser)</summary>';
  for(const pr of PARSERS){
    const obj={answers:{}};
    for(const q of p.questions) obj.answers[q.qid+'::answer']=(q.preds[pr]||{}).a;
    rj+='<div class="rj-h">'+LBL[pr]+'</div><pre class="rj">'+
        esc(JSON.stringify(obj,null,2))+'</pre>';
  }
  rj+='</details>';
  h+=ao+rj+p.questions.map(qCard).join('');
  document.getElementById('qs').innerHTML=h;
  document.getElementById('qs').scrollTop=0;
}
function qCard(q){
  const d=D.q[q.qid]||{};
  const gold=d.gold!=null?d.gold:q.gold;
  const wasg=(q.gchg&&d.gold==null)?
    '<div class="wasgold">was: <s>'+esc(q.prevg)+'</s></div>':'';
  let rows='<tr class="gold"><td class="ok">★</td><td>GOLD'+(d.gold!=null?' (edited)':'')+
    '</td><td id="g-'+q.qid+'">'+esc(gold)+wasg+'</td></tr>';
  for(const p of PARSERS){const pr=q.preds[p]||{};
    rows+='<tr><td class="'+(pr.ok?'ok':'no')+'">'+(pr.ok?'✓':'✗')+'</td>'+
      '<td>'+LBL[p]+'</td><td>'+esc(pr.a)+'</td></tr>';}
  const caps=(q.caps||[]).map(c=>'<span class="cap">'+c+'</span>').join('');
  const dis=q.disagree>1?'<span class="dis'+(q.disagree>=4?' hi':'')+'">≠'+
    q.disagree+' parsers</span>':'';
  const gc=q.gchg?'<span class="gchg" title="gold auto-corrected vs page '+
    'image during rewrite">⚑ gold corrected</span>':'';
  return '<div class="qcard '+({keep:'k',reject:'r'}[d.v]||'')+'" id="q-'+q.qid+'">'+
    '<div class="qid">'+q.qid+caps+dis+gc+'</div>'+
    '<div class="qtext">'+esc(q.q)+'</div>'+
    '<table class="tbl"><tr><th></th><th>source</th><th>answer</th></tr>'+rows+'</table>'+
    '<div class="acts">'+
    '<button class="b-k" onclick="decQ(\''+q.qid+'\',\'keep\')">✓ keep</button>'+
    '<button class="b-r" onclick="decQ(\''+q.qid+'\',\'reject\')">✗ reject</button>'+
    '<button class="b-e" onclick="editGold(\''+q.qid+'\')">✎ edit gold</button></div></div>';
}
function decQ(qid,v){D.q[qid]=D.q[qid]||{};
  D.q[qid].v=(D.q[qid].v===v?null:v);save();
  const c=document.getElementById('q-'+qid);
  if(c)c.className='qcard '+({keep:'k',reject:'r'}[D.q[qid].v]||'');
  updateProg();}
function decDoc(v){const p=PAGES[cur];D.doc[p.stem]=D.doc[p.stem]||{};
  D.doc[p.stem].v=(D.doc[p.stem].v===v?null:v);save();
  buildNav();selectDoc(cur);}
function editGold(qid){const cell=document.getElementById('g-'+qid);
  const d=D.q[qid]||{};const q=PAGES[cur].questions.find(x=>x.qid===qid);
  const cur_g=d.gold!=null?d.gold:q.gold;
  cell.innerHTML='<textarea class="gold-ed" id="ge-'+qid+'">'+esc(cur_g)+'</textarea>'+
    '<button onclick="saveGold(\''+qid+'\')">save</button>';}
function saveGold(qid){const t=document.getElementById('ge-'+qid);
  D.q[qid]=D.q[qid]||{};D.q[qid].gold=t.value;D.q[qid].v='keep';save();
  renderQs();updateProg();}
function updateProg(){const np=document.getElementById('np-'+cur);
  if(np)np.textContent=decided(PAGES[cur])+'/'+PAGES[cur].questions.length;}
function nextDoc(d){let i=cur+d;if(i>=0&&i<PAGES.length)selectDoc(i);}
function zoom(s){const z=document.getElementById('zoom');
  z.querySelector('img').src=s;z.classList.add('show');}
function dl(){const blob=new Blob([JSON.stringify(D,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='__RUN___validation_'+new Date().toISOString().slice(0,16).replace(/[:T]/g,'-')+'.json';
  a.click();}
function restore(inp){const f=inp.files[0];if(!f)return;const rd=new FileReader();
  rd.onload=()=>{try{D=JSON.parse(rd.result);if(!D.q)D.q={};if(!D.doc)D.doc={};
    save();buildNav();selectDoc(cur);alert('Restored.');}catch(e){alert('Bad file');}};
  rd.readAsText(f);}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT'||
     e.target.tagName==='SELECT')return;
  if(e.key==='ArrowRight'||e.key==='ArrowDown'||e.key==='j'){nextDoc(1);e.preventDefault();}
  if(e.key==='ArrowLeft'||e.key==='ArrowUp'||e.key==='k'){nextDoc(-1);e.preventDefault();}});
buildNav();counts();selectDoc(0);
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
