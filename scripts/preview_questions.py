#!/usr/bin/env python3
"""Light-theme viewer for a generated questions JSON (chart / checkmark).

Each card shows the page image, the question, response format, gold answer,
and capability tags — with keep / drop / edit-gold controls and a clipboard
export so a human can QC a freshly generated question category.

Usage: preview_questions.py <questions.json> <run-dir> <out.html>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def main() -> int:
    qfile = Path(sys.argv[1])
    run = Path(sys.argv[2])
    out = Path(sys.argv[3])
    items = json.loads(qfile.read_text()).get("items", [])
    title = qfile.stem

    cards = []
    for it in items:
        qid = it.get("question_id", "?")
        caps = " ".join(f"<span class='cap'>{esc(c)}</span>"
                        for c in it.get("capabilities", []))
        cards.append(f"""
<div class="card" id="c-{qid}">
  <div class="left"><img src="../images/{it.get('source_file','')}.png"
       onclick="zoom(this.src)"></div>
  <div class="right">
    <div class="qid">{qid} {caps}</div>
    <div class="q">{esc(it.get('question',''))}</div>
    <div class="rf">{esc(it.get('response_format',''))}</div>
    <div class="lbl">GOLD</div>
    <div class="gold" id="g-{qid}">{esc(it.get('gold_answer',''))}</div>
    <div class="acts">
      <button class="a-ok" onclick="dec('{qid}','keep')">✓ keep</button>
      <button class="a-bad" onclick="dec('{qid}','drop')">✗ drop</button>
      <button class="a-edit" onclick="editGold('{qid}')">✎ edit gold</button>
    </div>
  </div>
</div>""")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title} — QC</title><style>
body{{font-family:-apple-system,system-ui,sans-serif;background:#fff;color:#1f2328;
margin:0;padding:16px 24px;font-size:13px}}
.bar{{position:sticky;top:0;background:#f6f8fa;border-bottom:1px solid #d1d9e0;
padding:10px 14px;margin:0 -24px 16px;display:flex;gap:14px;align-items:center}}
.bar button{{background:#0969da;color:#fff;border:0;border-radius:5px;padding:6px 12px;cursor:pointer}}
.card{{display:grid;grid-template-columns:320px 1fr;gap:14px;border:1px solid #d1d9e0;
border-radius:8px;padding:12px;margin-bottom:16px}}
.card.keep{{border-color:#1a7f37}} .card.drop{{border-color:#cf222e;opacity:.5}}
.left img{{width:100%;border:1px solid #d1d9e0;cursor:zoom-in}}
.qid{{font-family:monospace;font-size:10px;color:#59636e;margin-bottom:4px}}
.cap{{background:#ddf4ff;color:#0969da;padding:0 5px;border-radius:3px;font-size:9px}}
.q{{font-weight:600;margin:4px 0}}
.rf{{font-family:monospace;font-size:11px;color:#59636e}}
.lbl{{font-size:10px;color:#59636e;text-transform:uppercase;margin-top:8px;font-weight:700}}
.gold{{font-family:monospace;font-size:11px;background:#ddf4ff;border:1px solid #d1d9e0;
border-radius:4px;padding:5px 8px;white-space:pre-wrap;word-break:break-word}}
.acts{{margin-top:8px;display:flex;gap:6px}}
.acts button{{font-size:11px;padding:4px 10px;border:1px solid #d1d9e0;border-radius:4px;
cursor:pointer;background:#fff}}
.a-ok{{color:#1a7f37}} .a-bad{{color:#cf222e}} .a-edit{{color:#9a6700}}
#zoom{{position:fixed;inset:0;background:rgba(0,0,0,.8);display:none;align-items:center;
justify-content:center;cursor:zoom-out;z-index:99}}
#zoom img{{max-width:95vw;max-height:95vh}} #zoom.show{{display:flex}}
textarea{{width:100%;font-family:monospace;font-size:11px;min-height:50px}}
</style></head><body>
<div class="bar"><h1 style="margin:0">{title} — {len(items)} questions · QC</h1>
<button onclick="exp()">Export decisions ↓</button><span id="cnt"></span></div>
{''.join(cards)}
<div id="zoom" onclick="this.classList.remove('show')"><img></div>
<script>
const KEY="preview-{title}";
let D=JSON.parse(localStorage.getItem(KEY)||"{{}}");
function save(){{localStorage.setItem(KEY,JSON.stringify(D));
 document.getElementById('cnt').textContent=Object.keys(D).length+' decided';}}
function dec(q,v){{D[q]=D[q]||{{}};D[q].verdict=v;save();
 document.getElementById('c-'+q).className='card '+v;}}
function editGold(q){{const g=document.getElementById('g-'+q);const cur=g.textContent;
 g.innerHTML='<textarea id="t-'+q+'">'+cur+'</textarea>'+
 '<button onclick="saveGold(\\''+q+'\\')">save</button>';}}
function saveGold(q){{const t=document.getElementById('t-'+q);
 D[q]=D[q]||{{}};D[q].corrected_gold=t.value;D[q].verdict='keep';save();
 document.getElementById('g-'+q).textContent=t.value;}}
function zoom(s){{const z=document.getElementById('zoom');
 z.querySelector('img').src=s;z.classList.add('show');}}
function exp(){{navigator.clipboard.writeText(JSON.stringify(D,null,2));
 alert('Copied '+Object.keys(D).length+' decisions.');}}
save();
</script></body></html>"""
    out.write_text(html)
    print(f"wrote {out} ({len(items)} questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
