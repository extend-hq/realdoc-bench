#!/usr/bin/env python3
"""Pilot 50 eval: for each question × parser, ask Sonnet 4.6 to answer using
the parser's markdown output. Exact-match against gold.

Output: runs/pilot_50/eval/eval_r1.json
  {
    "<dest_path>": {
      "questions": {
        "<question_id>": {
          "gold": "...",
          "parser_answers": {"<parser>": "<answer>"},
          "parsers_correct": ["<parser1>", ...]
        }
      }
    }
  }
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
PILOT = REPO / "runs" / "pilot_50"


def _set_run_dir(run_dir: Path) -> None:
    global PILOT
    PILOT = run_dir

ANTHROPIC_MODEL = os.environ.get("PILOT_ANTHROPIC_MODEL", "claude-sonnet-4-6")
# All parser candidates. The script only evals parsers whose .md output exists
# in runs/<run-dir>/parses/<parser>/, so unused ones are skipped automatically.
PARSERS = ["extend_performance_v1_0_0", "extend_performance_v2_0_0_advanced",
           "extend_performance_v2_0_0_agentic",
           "aws_textract", "azure_di", "reducto",
           "llamaparse", "llamaparse_cost_effective"]


def load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k and not os.environ.get(k):
            os.environ[k] = v


load_dotenv()


def flatten_nested(s) -> dict[str, str]:
    """Flatten a 2-level nested gold/format string into {compound_key: value}.

    Handles the shapes the QA generator emits for grouped answers:
      'OUTER: k=v, k=v; OUTER2: k=v, k=v'   colon outer label, equals inner
      'k=v, k=v; k=v, k=v'                  no outer label (grid)
      'outer: {k: v, k: v} | outer2: {...}' brace style
    Compound key is 'OUTER.k' (or just 'k' with no outer). Commas inside a
    value are tolerated by re-joining fragments that lack '='. Returns {}
    when nothing decomposes.
    """
    s = str(s or "").strip()
    if not s:
        return {}
    if "{" in s or "[" in s:                  # brace/bracket style → normalize
        s = s.replace("|", ";")
        s = re.sub(r"[\{\[]([^\{\}\[\]]*)[\}\]]",
                   lambda m: re.sub(r"(?<![0-9])\s*:\s*(?![0-9])", "=", m.group(1)),
                   s)
    out: dict[str, str] = {}
    for seg in s.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        m = re.match(r"^([^:=]+?):\s*(.+)$", seg)   # 'LABEL: rest'
        if m and "=" not in m.group(1):
            outer, rest = m.group(1).strip(), m.group(2)
        else:
            outer, rest = "", seg
        pairs: list[str] = []
        for piece in rest.split(","):
            if "=" in piece:
                pairs.append(piece)
            elif pairs:
                pairs[-1] += "," + piece            # comma was inside a value
        for p in pairs:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip().strip('"').strip("'")
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1].strip()                 # drop brace-style quoting
            if not k:
                continue
            ck = f"{outer}.{k}" if outer else k
            base, n = ck, 2
            while ck in out:                        # de-dup colliding keys
                ck, n = f"{base}#{n}", n + 1
            out[ck] = v
    return out


def gold_to_struct(gold) -> dict[str, str]:
    """Parse a gold string into an ordered {key: value} dict for per-sub-answer
    scoring — a 4-part question becomes 4 independently-scored cells.

    A clean 'k=v; k=v' string decomposes directly. A 2-level nested string
    ('OUTER: k=v, k=v; ...') is flattened to compound keys via flatten_nested.
    Only a genuine scalar (no '=') stays a single {"answer": ...} item.
    """
    g = "null" if gold is None else str(gold).strip()
    if "=" not in g:
        return {"answer": g}
    segs = [s.strip() for s in g.split(";") if s.strip()]
    out: dict[str, str] = {}
    clean = True
    for seg in segs:
        if seg.count("=") != 1:
            clean = False
            break
        k, v = seg.split("=", 1)
        k = k.strip()
        if not k or ":" in k or k in out:
            clean = False
            break
        out[k] = v.strip()
    if clean and out:
        return out
    # not a clean flat list → try the 2-level nested flatten
    nested = flatten_nested(g)
    if len(nested) >= 2:
        return nested
    return {"answer": g}


def format_to_struct(response_format) -> dict[str, str]:
    """Parse a 'Return exactly: k=<fmt>; k=<fmt>' response_format into
    {key: per-key-format-hint}. Mirrors gold_to_struct's segmenting so the
    keys line up. Returns {} when the format does not cleanly decompose —
    callers then fall back to no per-key hint for that item.
    """
    rf = str(response_format or "").strip()
    rf = re.sub(r"^\s*return[^:]*:\s*", "", rf, flags=re.IGNORECASE)
    if "=" not in rf:
        return {}
    segs = [s.strip() for s in rf.split(";") if s.strip()]
    out: dict[str, str] = {}
    clean = True
    for seg in segs:
        if seg.count("=") != 1:
            clean = False
            break
        k, v = seg.split("=", 1)
        k = k.strip()
        if not k or ":" in k or k in out:
            clean = False
            break
        out[k] = v.strip()
    if clean and out:
        return out
    nested = flatten_nested(rf)        # 2-level format → compound-key hints
    return nested if len(nested) >= 2 else {}


def normalize_exact(value: str | None) -> str:
    """Match selected_pages_v2 normalize_exact rules for plain-string gold."""
    if value is None:
        return "null"
    text = str(value).strip()
    text = text.replace("None", "null")
    text = re.sub(r"(?<=\d)\s*:\s*(?=\d)", ":", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ;", ";").replace("; ", "; ")
    text = text.replace(" =", "=").replace("= ", "=")
    return text.lower()


SYSTEM_PROMPT = (
    "You answer benchmark extraction items about ONE parsed document using "
    "ONLY the markdown text given (extracted from that document by a parser). "
    "Do not use outside knowledge or invent values.\n\n"
    "Each item asks for exactly ONE value. Return exactly one JSON object: "
    '{"answers": {"<item-id>": <value>, ...}} covering EXACTLY the item ids '
    "given — no more, no fewer, same spelling. Each value is a JSON string "
    "holding the requested value, or JSON null if that value is missing, "
    "blank, contradictory, or unclear from the markdown. Answer each item "
    "independently — do not let one item's answer influence another.\n\n"
    "CRITICAL — preserve values verbatim. Format placeholders like <value>, "
    "<text>, <number> describe what TYPE of value is expected; they are NEVER "
    "instructions to strip characters from the printed value. Preserve every "
    "value exactly as printed in the markdown — keep thousands separators "
    "(commas), decimal points and trailing zeros (e.g. \"5,280.00\" — not "
    "528000, not 5280), currency symbols, dashes, slashes, and spelling. Only "
    "collapse multi-line breaks to single spaces. Output no prose, labels, or "
    "extra keys."
)


def _coerce_value(v) -> str | None:
    """Normalize one JSON value from the model into a str answer or None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    s = str(v).strip()
    if s == "" or s.lower() == "null" or s.lower() == "none":
        return None
    return s


def _pick(answers: dict, uid: str):
    """Look up one item id in the model's answer dict, tolerating a model that
    drops the ``::answer`` suffix (Gemini does this) — but only for scalar
    items, so a multi-part ``::<key>`` id never collides on the bare qid."""
    if uid in answers:
        return answers[uid]
    if uid.endswith("::answer"):
        base = uid[: -len("::answer")]
        if base in answers:
            return answers[base]
    return None


def _build_eval_user_text(units: list[dict], parser_md: str) -> str:
    """Shared eval-agent user message: the parser markdown + one block per
    single-value item. Every eval agent (Sonnet, Gemini) sees an identical
    prompt so their answers are comparable for the consensus vote."""
    blocks = []
    for i, u in enumerate(units, 1):
        lines = [f"[Item {i}]  id: {u['id']}", f"question: {u['question']}"]
        if u.get("target"):
            lines.append(
                f'Return ONLY the value of sub-field "{u["target"]}" from this '
                f"question — exactly one value, NOT the other sub-fields and "
                f"NOT a combined string."
            )
        if u.get("response_format"):
            lines.append(f"value format: {u['response_format']}")
        blocks.append("\n".join(lines))
    return (
        f"Parsed document markdown:\n```\n{parser_md[:60000]}\n```\n\n"
        f"Answer ALL {len(units)} extraction items below using ONLY that "
        f"markdown. Each item wants exactly one value.\n\n"
        + "\n\n".join(blocks)
        + '\n\nReturn only the JSON object: {"answers": {"<id>": <value>, ...}} '
          "covering EXACTLY the item ids above."
    )


def anthropic_answer_batch(units: list[dict], parser_md: str,
                           model: str = ANTHROPIC_MODEL,
                           api_key: str | None = None) -> dict[str, str | None]:
    """One Claude call answering ALL single-value extraction items for ONE
    parsed document (one (doc, parser) pair).

    Multi-part questions are pre-flattened by the caller, so every item asks
    for exactly one value. The parser markdown — the expensive part of the
    payload — is sent ONCE instead of once per item.

    ``units`` — list of {"id", "question", "target", "response_format"};
      "target" is the sub-answer key, or None for a whole-question item.
    Returns a flat {item_id: answer-string-or-None}. Items the model omits
    or mangles fall back to None so the caller still records a result.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    fallback = {u["id"]: None for u in units}
    user_text = _build_eval_user_text(units, parser_md)
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 4096,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_text}],
    }
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    url = f"{base}/v1/messages"
    with httpx.Client(timeout=180) as client:
        r = client.post(url, headers=headers, json=body)
        if r.status_code != 200:
            raise RuntimeError(f"anthropic http {r.status_code}: {r.text[:200]}")
        data = r.json()
    text = data["content"][0]["text"].strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return fallback
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return fallback
    answers = obj.get("answers") if isinstance(obj.get("answers"), dict) else obj
    if not isinstance(answers, dict):
        return fallback
    return {u["id"]: _coerce_value(_pick(answers, u["id"])) for u in units}


def gemini_answer_batch(units: list[dict], parser_md: str,
                        model: str = "gemini-3-flash-preview",
                        api_key: str | None = None) -> dict[str, str | None]:
    """Gemini counterpart of ``anthropic_answer_batch`` — same prompt, same
    flat {item_id: answer} contract. Used as a second, independent eval agent
    so a question's score is a cross-model consensus, not one model's draw."""
    api_key = (api_key or os.environ.get("GOOGLE_API_KEY")
               or os.environ.get("GEMINI_API_KEY"))
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    fallback = {u["id"]: None for u in units}
    user_text = _build_eval_user_text(units, parser_md)
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    with httpx.Client(timeout=180) as client:
        r = client.post(url, params={"key": api_key}, json=body)
        if r.status_code != 200:
            raise RuntimeError(f"gemini http {r.status_code}: {r.text[:200]}")
        data = r.json()
    cands = data.get("candidates") or []
    if not cands:
        raise RuntimeError(f"gemini: no candidates ({str(data)[:160]})")
    text = cands[0]["content"]["parts"][0]["text"].strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return fallback
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return fallback
    answers = obj.get("answers") if isinstance(obj.get("answers"), dict) else obj
    if not isinstance(answers, dict):
        return fallback
    return {u["id"]: _coerce_value(_pick(answers, u["id"])) for u in units}


# Cross-model self-consistency: every (doc,parser) batch is answered by BOTH
# eval agents, EVAL_RUNS_PER_AGENT times each; the per-question answer is the
# majority vote across all runs. A single model's degenerate draw (e.g. a whole
# batch collapsing to all-null) is outvoted instead of poisoning the score.
EVAL_AGENTS = [("sonnet", anthropic_answer_batch),
               ("gemini", gemini_answer_batch)]
EVAL_RUNS_PER_AGENT = 3
_NULL_KEY = "\x00null"


def _consensus(runs: list[dict], ids: list[str]) -> tuple[dict, dict]:
    """Majority-vote each item id across all runs. Ties between null and a
    real value go to the value. Returns (consensus, agreement-fraction)."""
    consensus, agreement = {}, {}
    n = len(runs) or 1
    for i in ids:
        cnt: dict[str, int] = {}
        forms: dict[str, object] = {}
        for r in runs:
            v = r.get(i)
            key = _NULL_KEY if v is None else normalize_exact(v)
            cnt[key] = cnt.get(key, 0) + 1
            if key not in forms or v is not None:
                forms[key] = v
        if not cnt:
            consensus[i], agreement[i] = None, 0.0
            continue
        ranked = sorted(cnt.items(), key=lambda kv: -kv[1])
        best_key, best_n = ranked[0]
        if best_key == _NULL_KEY and len(ranked) > 1 and ranked[1][1] == best_n:
            best_key, best_n = ranked[1]            # tie → prefer a real value
        consensus[i] = forms[best_key]
        agreement[i] = best_n / n
    return consensus, agreement


def answer_batch_consensus(payload: list[dict], parser_md: str
                           ) -> tuple[dict, dict, list, bool]:
    """Run every eval agent EVAL_RUNS_PER_AGENT times (runs fired concurrently)
    and majority-vote. Returns (consensus, agreement, runs, ok)."""
    ids = [u["id"] for u in payload]
    plan = [(name, fn) for name, fn in EVAL_AGENTS
            for _ in range(EVAL_RUNS_PER_AGENT)]

    def one(nf):
        name, fn = nf
        try:
            return {"agent": name, "answers": fn(payload, parser_md)}
        except Exception as e:
            return {"agent": name, "error": str(e)[:160]}

    runs = []
    with ThreadPoolExecutor(max_workers=len(plan)) as ex:
        runs = list(ex.map(one, plan))
    good = [r["answers"] for r in runs if "answers" in r]
    consensus, agreement = _consensus(good, ids)
    return consensus, agreement, runs, len(good) > 0


def parser_md_path(parser_name: str, dest_path: str) -> Path:
    stem = Path(dest_path).with_suffix("").as_posix()
    return PILOT / "parses" / parser_name / f"{stem}.md"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=REPO / "runs" / "pilot_50")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="limit Q count for smoke")
    ap.add_argument("--domain", nargs="*", default=None,
                    help="only docs whose sample.csv 'domain' is in this list")
    ap.add_argument("--type", nargs="*", default=None,
                    help="only docs whose sample.csv 'type' is in this list")
    ap.add_argument("--force", action="store_true",
                    help="re-evaluate even when a cached answer file exists")
    args = ap.parse_args(argv)
    _set_run_dir(args.run_dir.resolve())

    bank = json.loads((PILOT / "qa" / "bank_r1.json").read_text())
    sample = {r["dest_path"]: r for r in csv.DictReader((PILOT / "sample.csv").open())}
    from realdoc_bench.pipeline.dataset import filter_rows
    if args.domain or args.type:
        kept = {r["dest_path"] for r in filter_rows(list(sample.values()),
                                                    args.domain, args.type)}
        sample = {k: v for k, v in sample.items() if k in kept}
    stem_to_dest = {Path(r["dest_path"]).stem: r["dest_path"] for r in sample.values()}

    items = bank["items"]
    if args.limit:
        items = items[: args.limit]

    # ── Flatten every question into single-value items ──────────────────────
    # A multi-part question (gold "14=..; 15=..") becomes one item per key; a
    # scalar question is one item. item id = "<qid>::<key>".
    docs: dict[str, list[dict]] = {}   # dest_path -> [flat unit, ...]
    for it in items:
        dp = stem_to_dest.get(it["source_file"])
        if not dp:
            continue
        gs = gold_to_struct(it["gold_answer"])
        fs = format_to_struct(it["response_format"])
        is_scalar = list(gs.keys()) == ["answer"]
        bucket = docs.setdefault(dp, [])
        for key, gold_val in gs.items():
            # Per-item format hint: scalar → whole format; multi-part → ONLY
            # this key's slice (passing the full multi-part format makes the
            # model answer the whole question for every sub-item).
            rf = it["response_format"] if is_scalar else fs.get(key)
            bucket.append({
                "id": f"{it['question_id']}::{key}",
                "qid": it["question_id"],
                "key": key,
                "gold_value": gold_val,
                "question": it["question"],
                "response_format": rf,
                "target": None if is_scalar else key,
                "_item": it,
            })

    # ── Jobs: one LLM call per (doc, parser) ────────────────────────────────
    jobs = []  # (dest_path, parser, units, md_path)
    for dp, units in docs.items():
        for p in PARSERS:
            md = parser_md_path(p, dp)
            if not md.exists():
                continue
            jobs.append((dp, p, units, md))
    n_items = sum(len(v) for v in docs.values())
    print(f"Eval: {len(docs)} docs · {n_items} single-value items × "
          f"{len(PARSERS)} parsers, batched per (doc,parser) = {len(jobs)} LLM calls")

    # Cache — one file per (doc, parser); holds the flat {item_id: answer} map.
    cache_dir = PILOT / "eval" / "answers_flat"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_key(dp, parser_name):
        return cache_dir / f"{Path(dp).stem}__{parser_name}.json"

    todo = jobs if args.force else [j for j in jobs if not cache_key(j[0], j[1]).exists()]
    print(f"  cached {len(jobs) - len(todo)}, todo {len(todo)}")

    def do(job):
        dp, parser_name, units, md_path = job
        try:
            md = md_path.read_text()
        except FileNotFoundError:
            # Parse output got wiped mid-run. Skip without writing a cache
            # file — re-run after parses land to fill the gap.
            return (dp, parser_name, False)
        payload = [{"id": u["id"], "question": u["question"],
                    "target": u["target"], "response_format": u["response_format"]}
                   for u in units]
        consensus, agreement, runs, ok = answer_batch_consensus(payload, md)
        ck = cache_key(dp, parser_name)
        ck.write_text(json.dumps({
            "dest_path": dp, "parser": parser_name,
            "answers": consensus, "ok": ok,
            "error": "" if ok else "all eval-agent runs failed",
            "agreement": agreement, "runs": runs,
        }))
        return (dp, parser_name, ok)

    t0 = time.time()
    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(do, j): j for j in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            dp, p, ok = fut.result()
            if ok: n_ok += 1
            else: n_fail += 1
            if i % 25 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] ok={n_ok} fail={n_fail}  ({(time.time()-t0)/60:.1f}min)",
                      flush=True)

    # ── Aggregate — expand flat answers back into per-sub-answer scoring ─────
    by_page: dict = {}
    for dp, parser_name, units, md_path in jobs:
        ck = cache_key(dp, parser_name)
        if not ck.exists():
            continue
        cached = json.loads(ck.read_text())
        flat = cached.get("answers") or {}
        page_d = by_page.setdefault(dp, {"questions": {}})
        for u in units:
            it = u["_item"]
            gs = gold_to_struct(it["gold_answer"])
            qd = page_d["questions"].setdefault(u["qid"], {
                "gold": it["gold_answer"],
                "gold_struct": gs,
                "keys": list(gs.keys()),
                "parser_answers": {},      # parser -> {key: answer}
                "sub_correct": {k: [] for k in gs},   # key -> [parsers strict-correct]
                "parsers_correct": [],     # parsers correct on ALL keys
            })
            ans = flat.get(u["id"])
            qd["parser_answers"].setdefault(parser_name, {})[u["key"]] = ans
            if normalize_exact(ans) == normalize_exact(u["gold_value"]):
                qd["sub_correct"][u["key"]].append(parser_name)

    # question-level: a parser is correct iff correct on every key
    for page_d in by_page.values():
        for qd in page_d["questions"].values():
            keys = qd["keys"]
            qd["parsers_correct"] = [
                p for p in qd["parser_answers"]
                if all(p in qd["sub_correct"].get(k, []) for k in keys)
            ]

    eval_path = PILOT / "eval" / "eval_r1.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(by_page, indent=2, ensure_ascii=False))

    # Summary — per-sub-answer + question-level
    total_q = sum(1 for p in by_page.values() for q in p["questions"].values())
    total_sub = sum(len(q["keys"]) for p in by_page.values()
                    for q in p["questions"].values())
    n_parsers = len(PARSERS)
    by_diff = [0] * (n_parsers + 1)
    for p in by_page.values():
        for q in p["questions"].values():
            n_correct = len(q["parsers_correct"])
            if 0 <= n_correct <= n_parsers:
                by_diff[n_correct] += 1
    print(f"\n=== eval summary (per-sub-answer scoring) ===")
    print(f"  questions: {total_q}   sub-answers: {total_sub}")
    print(f"  question-level difficulty (parsers fully correct on all keys):")
    for i, n in enumerate(by_diff):
        bar = "█" * (n // 5)
        print(f"    {i}/{n_parsers}: {n:>4} {bar}")
    print(f"\n  per-parser accuracy (strict):")
    for p in PARSERS:
        sub_c = sub_t = q_c = q_t = 0
        for pg in by_page.values():
            for q in pg["questions"].values():
                if p not in q["parser_answers"]:
                    continue
                q_t += 1
                if p in q["parsers_correct"]:
                    q_c += 1
                for k in q["keys"]:
                    sub_t += 1
                    if p in q["sub_correct"][k]:
                        sub_c += 1
        if sub_t:
            print(f"    {p:38} sub {sub_c:>5}/{sub_t} ({sub_c/sub_t*100:.0f}%)"
                  f"  ·  Q {q_c:>4}/{q_t} ({q_c/q_t*100:.0f}%)")

    print(f"\nwritten: {eval_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
