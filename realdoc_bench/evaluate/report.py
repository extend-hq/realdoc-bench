"""Build the QA-evaluation HTML dashboard.

Reads `<run-dir>/eval/cache/*.json` + the bank → builds per-parser bar charts,
per-domain heatmap, per-capability bucket drilldowns (heatmap + bar chart +
representative failing examples per bucket), and a browsable Q&A view.
Self-contained single HTML file at `<run-dir>/dashboard.html`. Read-only on
caches; never calls models.
"""

from __future__ import annotations

import json
from collections import defaultdict
from importlib import resources
from pathlib import Path

import yaml

from realdoc_bench.evaluate.runs import RunLayout


# ── Parser display metadata ────────────────────────────────────────────────
_PARSER_LABELS = {
    "extend_performance_v2_0_0": "Extend v2 - Performance",
    "extend_performance_v1_0_0": "Extend v1",
    "llamaparse": "LlamaParse (Agentic)",
    "llamaparse_cost_effective": "LlamaParse",
    "reducto": "Reducto",
    "reducto_agentic": "Reducto (Agentic)",
    "azure_di": "Azure DI",
    "aws_textract": "AWS Textract",
    "gemini_3_5_flash": "Gemini 3.5 Flash",
}

# Parser color palette — same family for related models (Extend = blues,
# LlamaParse = emerald, Reducto = orange, Gemini = pink, etc.). Designed for
# readability across stacked bars and per-parser heatmap rows.
_PARSER_COLORS = {
    "extend_performance_v2_0_0": "#2563eb",
    "extend_performance_v1_0_0":          "#60a5fa",
    "llamaparse":                         "#059669",
    "llamaparse_cost_effective":          "#6ee7b7",
    "reducto_agentic":                    "#ea580c",
    "reducto":                            "#fdba74",
    "azure_di":                           "#7c3aed",
    "aws_textract":                       "#f59e0b",
    "gemini_3_5_flash":                   "#db2777",
}

# Canonical sort order (top to bottom of UI tables / bar charts when ties)
_CANONICAL_ORDER = [
    "extend_performance_v2_0_0", "extend_performance_v1_0_0",
    "llamaparse", "llamaparse_cost_effective",
    "reducto_agentic", "reducto", "azure_di", "aws_textract",
    "gemini_3_5_flash",
]


# ── Capability buckets ─────────────────────────────────────────────────────
# Loaded from YAML — defaults ship at
# `realdoc_bench/evaluate/capability_buckets.yaml`. Override per-run with
# `evaluate report --capabilities path/to/your.yaml`. See the bundled YAML
# for the schema.

def load_buckets(path: str | Path | None = None) -> list[tuple[str, list[str], str]]:
    """Load capability bucket definitions from YAML.

    Returns ``[(name, tags, desc), ...]`` in declared order. ``path=None``
    reads the packaged default. Unknown tags are kept as-is — they'll just
    yield 0 questions for that bucket.
    """
    if path is None:
        text = resources.files("realdoc_bench.evaluate").joinpath(
            "capability_buckets.yaml").read_text()
    else:
        text = Path(path).read_text()
    raw = yaml.safe_load(text) or []
    out: list[tuple[str, list[str], str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(
                f"capability bucket entry must be a mapping, got {type(entry).__name__}")
        if "name" not in entry or "tags" not in entry:
            raise ValueError(
                f"capability bucket missing required 'name'/'tags' field: {entry!r}")
        name, tags, desc = entry["name"], entry["tags"], entry.get("desc", "")
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise ValueError(f"bucket {name!r}: tags must be a list of strings")
        out.append((str(name), [str(t) for t in tags], str(desc)))
    return out


def _bucket_qids(tags: list[str], bank: dict) -> list[str]:
    tag_set = set(tags)
    return [qid for qid, it in bank.items()
            if tag_set & set(it.get("capabilities") or [])]


def _parser_score_for_qids(qids: list[str], pk: str,
                            res: dict[str, dict[str, dict]]) -> dict:
    """Aggregate per-field and per-question accuracy for one parser over a
    subset of qids."""
    f_ok = f_tot = q_ok = q_tot = 0
    for q in qids:
        r = res.get(q, {}).get(pk)
        if not r:
            continue
        fm = r.get("field_matches") or {}
        f_ok += sum(1 for v in fm.values() if v)
        f_tot += len(fm)
        q_tot += 1
        q_ok += 1 if r.get("match") else 0
    return {
        "field_pct": round(100 * f_ok / f_tot, 1) if f_tot else 0,
        "q_pct": round(100 * q_ok / q_tot, 1) if q_tot else 0,
        "f_ok": f_ok, "f_tot": f_tot, "q_ok": q_ok, "q_tot": q_tot,
    }


def _bucket_examples(qids: list[str], pkeys: list[str],
                      res: dict[str, dict[str, dict]], bank: dict,
                      n: int = 6) -> list[dict]:
    """Pick the N most parser-discriminative questions in a bucket — those
    where parsers disagree most (n_wrong × n_right peaks at half-and-half)."""
    scored: list[tuple[int, str]] = []
    for qid in qids:
        if qid not in res:
            continue
        n_with = sum(1 for p in pkeys if p in res[qid])
        if n_with < 5:    # need enough parsers to be informative
            continue
        n_wrong = sum(1 for p in pkeys
                      if p in res[qid] and not res[qid][p].get("match"))
        scored.append((n_wrong * (n_with - n_wrong), qid))
    scored.sort(key=lambda kv: (-kv[0], kv[1]))

    out: list[dict] = []
    for _, qid in scored[:n]:
        it = bank[qid]
        answers = {}
        for pk in pkeys:
            r = res[qid].get(pk)
            if not r:
                answers[pk] = None
                continue
            answers[pk] = {"v": r.get("answer"),
                           "ok": bool(r.get("match"))}
        out.append({
            "qid": qid,
            "question": it.get("question", ""),
            "doc": it.get("source_file", ""),
            "domain": it.get("domain", ""),
            "gold": it.get("gold_dict") or {},
            "caps": it.get("capabilities") or [],
            "answers": answers,
        })
    return out


def _load_template() -> str:
    return resources.files("realdoc_bench.evaluate").joinpath(
        "report_template.html").read_text()


def build_report(layout: RunLayout, *, bank_path: Path | None = None,
                 out_path: Path | None = None,
                 merge_domains: dict[str, str] | None = None,
                 capabilities_path: Path | None = None) -> Path:
    bank_path = bank_path or layout.bank_path
    out_path = out_path or layout.dashboard_path
    merge_domains = merge_domains or {}
    bucket_defs = load_buckets(capabilities_path)

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

    ordered = [p for p in _CANONICAL_ORDER if p in parsers_seen]
    ordered += sorted(p for p in parsers_seen if p not in _CANONICAL_ORDER)
    PARSERS = [(p, _PARSER_LABELS.get(p, p)) for p in ordered]
    PKEYS = ordered
    PCOLORS = {p: _PARSER_COLORS.get(p, "#94a3b8") for p in ordered}

    domains = sorted({it.get("domain", "")
                      for it in bank.values() if it.get("domain")})

    # ── Per-parser summary + per-domain matrix ──────────────────────────────
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
            "key": pk, "label": _PARSER_LABELS.get(pk, pk),
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

    # ── Per-capability bucket scores + drilldown examples ───────────────────
    buckets = []
    for name, tags, desc in bucket_defs:
        qids = _bucket_qids(tags, bank)
        per_parser = {pk: _parser_score_for_qids(qids, pk, res) for pk in PKEYS}
        buckets.append({
            "name": name, "desc": desc, "tags": tags,
            "n_qs": len(qids),
            "per_parser": per_parser,
            "examples": _bucket_examples(qids, PKEYS, res, bank, n=6),
        })

    # ── Per-question detail (existing browser) ──────────────────────────────
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
        "summary": summary, "parsers": PARSERS, "parser_colors": PCOLORS,
        "questions": questions, "domains": domains,
        "dom_matrix": dom_matrix, "buckets": buckets,
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
