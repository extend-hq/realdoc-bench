"""Build the QA-evaluation HTML dashboard.

Reads `<run-dir>/eval/cache/*.json` + the bank → builds per-parser bar charts,
per-domain heatmap, and a browsable Q&A view. Self-contained single HTML file
at `<run-dir>/dashboard.html`. Read-only on caches; never calls models.
"""

from __future__ import annotations

import json
from collections import defaultdict
from importlib import resources
from pathlib import Path

from realdoc_bench.evaluate.runs import RunLayout


def _humanize(name: str) -> str:
    """Bare parser name → display label. Falls back to the bare name for any
    parser we haven't taught a nicer label."""
    labels = {
        "extend_performance_v2_0_0_advanced": "Extend v2 - Performance",
        "extend_performance_v1_0_0": "Extend v1",
        "extend_light_v1_0_0": "Extend Light v1",
        "llamaparse": "LlamaParse (Agentic)",
        "llamaparse_cost_effective": "LlamaParse",
        "reducto": "Reducto",
        "reducto_agentic": "Reducto (Agentic)",
        "azure_di": "Azure DI",
        "aws_textract": "AWS Textract",
        "pymupdf": "PyMuPDF",
    }
    return labels.get(name, name)


def _load_template() -> str:
    return resources.files("realdoc_bench.evaluate").joinpath(
        "report_template.html").read_text()


def build_report(layout: RunLayout, *, bank_path: Path | None = None,
                 out_path: Path | None = None,
                 merge_domains: dict[str, str] | None = None) -> Path:
    bank_path = bank_path or layout.bank_path
    out_path = out_path or layout.dashboard_path
    merge_domains = merge_domains or {}

    bank = {it["question_id"]: it
            for it in json.loads(bank_path.read_text())["items"]}
    if merge_domains:
        for it in bank.values():
            d = it.get("domain", "")
            it["domain"] = merge_domains.get(d, d)

    # cache: qid → parser → rec
    res: dict[str, dict[str, dict]] = defaultdict(dict)
    parsers_seen: set[str] = set()
    for f in layout.cache_dir.glob("*__*.json"):
        try:
            r = json.loads(f.read_text())
        except Exception:  # noqa: BLE001 — skip partial writes
            continue
        if "error" in r or "answer" not in r:
            continue
        res[r["qid"]][r["parser"]] = r
        parsers_seen.add(r["parser"])

    # Order parsers: known names first in the canonical UI order, unknown
    # appended alphabetically so the dashboard doesn't break for new parsers.
    canonical = [
        "extend_performance_v2_0_0_advanced", "extend_performance_v1_0_0",
        "extend_light_v1_0_0", "llamaparse", "llamaparse_cost_effective",
        "reducto", "reducto_agentic", "azure_di", "aws_textract", "pymupdf",
    ]
    ordered = [p for p in canonical if p in parsers_seen]
    ordered += sorted(p for p in parsers_seen if p not in canonical)
    PARSERS = [(p, _humanize(p)) for p in ordered]
    PKEYS = ordered

    domains = sorted({it.get("domain", "")
                      for it in bank.values() if it.get("domain")})

    # Per-parser summary + per-domain matrix
    summary = []
    dom_matrix: dict[str, dict[str, float]] = {}
    for pk in PKEYS:
        f_ok = f_tot = q_ok = q_tot = 0
        d_ok: dict[str, int] = defaultdict(int)
        d_tot: dict[str, int] = defaultdict(int)
        for qid, perp in res.items():
            r = perp.get(pk)
            if not r:
                continue
            dom = bank.get(qid, {}).get("domain", "")
            fm = r.get("field_matches") or {}
            f_tot += len(fm)
            f_ok += sum(1 for v in fm.values() if v)
            d_tot[dom] += len(fm)
            d_ok[dom] += sum(1 for v in fm.values() if v)
            q_tot += 1
            q_ok += 1 if r.get("match") else 0
        summary.append({
            "key": pk, "label": _humanize(pk),
            "field_pct": round(100 * f_ok / f_tot, 1) if f_tot else 0,
            "field_ok": f_ok, "field_tot": f_tot,
            "q_pct": round(100 * q_ok / q_tot, 1) if q_tot else 0,
            "q_ok": q_ok, "q_tot": q_tot,
        })
        dom_matrix[pk] = {
            d: round(100 * d_ok[d] / d_tot[d], 1) if d_tot[d] else 0
            for d in domains
        }
    summary.sort(key=lambda s: -s["field_pct"])

    # Per-question detail
    questions = []
    for qid, perp in res.items():
        it = bank.get(qid)
        if not it:
            continue
        gold = it.get("gold_dict") or {}
        fields = []
        for k, gv in gold.items():
            ans = {}
            for pk in PKEYS:
                r = perp.get(pk)
                if not r:
                    ans[pk] = None
                    continue
                a = r.get("answer")
                av = a.get(k) if isinstance(a, dict) else None
                ans[pk] = {"v": av,
                           "ok": bool((r.get("field_matches") or {}).get(k))}
            fields.append({"key": k, "gold": gv, "ans": ans})
        ncorrect = sum(1 for pk in PKEYS
                       if perp.get(pk) and perp[pk].get("match"))
        questions.append({
            "qid": qid, "doc": it.get("source_file", ""),
            "domain": it.get("domain", ""),
            "question": it.get("question", ""),
            "fields": fields, "n_parsers": len(perp), "n_correct": ncorrect,
        })
    questions.sort(key=lambda q: q["n_correct"])   # hardest first

    n_docs = len({q["doc"] for q in questions})
    data = json.dumps({
        "summary": summary, "parsers": PARSERS,
        "questions": questions, "domains": domains,
        "dom_matrix": dom_matrix,
    }, ensure_ascii=False, separators=(",", ":"))
    top = summary[0] if summary else {"label": "—", "field_pct": 0}
    evald = sum(len(p) for p in res.values())

    html = (_load_template()
            .replace("/*DATA*/", data)
            .replace("{TITLE}", "QA evaluation")
            .replace("{SUBTITLE}",
                     f"{n_docs} documents · {len(domains)} domains · "
                     f"multi-key typed-JSON extraction, per-field deep_equal")
            .replace("{RUN_DIR}", str(layout.root))
            .replace("{TOP_NAME}", top["label"])
            .replace("{TOP_PCT}", f'{top["field_pct"]}%')
            .replace("{N_Q}", f"{len(questions):,}")
            .replace("{N_DOCS}", str(n_docs))
            .replace("{N_DOM}", str(len(domains)))
            .replace("{N_CALLS}", f"{evald:,}"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
