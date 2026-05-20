#!/usr/bin/env python3
"""Re-score the pilot_50 eval with 3 strategies, layered:

  1. strict   — existing normalize_exact (current eval_r1.json score)
  2. fuzzy    — parse "key=val; key=val" into dict, per-value normalized
                + rapidfuzz token-set match (≥90)
  3. llm_judge — only for items strict+fuzzy disagree on; ask Gemini 3 Flash
                if answer is semantically equivalent to gold

Output:
  runs/pilot_50/eval/eval_r1_rescored.json — adds 3 parallel score sets:
    parsers_correct_strict, parsers_correct_fuzzy, parsers_correct_llm

Cost: only the (parser × Q) pairs where strict says wrong AND fuzzy says
right OR both say wrong (the residual) go to LLM. ~300 of the 960 cells
max → ~$0.30 with Gemini 3 Flash.
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
from rapidfuzz import fuzz

REPO = Path(__file__).resolve().parent.parent
PILOT = REPO / "runs" / "pilot_50"


def _set_run_dir(run_dir: Path) -> None:
    global PILOT
    PILOT = run_dir


def load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists(): return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        if line.startswith("export "): line = line[len("export "):]
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k and not os.environ.get(k): os.environ[k] = v


load_dotenv()


# ──────────────────────────────────────────────────────────────────────────
# Layer 1 — strict (current normalize_exact)
# ──────────────────────────────────────────────────────────────────────────
def normalize_strict(value: str | None) -> str:
    if value is None: return "null"
    text = str(value).strip()
    text = text.replace("None", "null")
    text = re.sub(r"(?<=\d)\s*:\s*(?=\d)", ":", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ;", ";").replace("; ", "; ")
    text = text.replace(" =", "=").replace("= ", "=")
    return text.lower()


def strict_match(answer, gold) -> bool:
    return normalize_strict(answer) == normalize_strict(gold)


# ──────────────────────────────────────────────────────────────────────────
# Layer 2 — fuzzy (per-value)
# ──────────────────────────────────────────────────────────────────────────
NULL_SYNONYMS = {"null", "none", "n/a", "na", "n.a.", "blank", "", "—", "-"}
YES_SYNONYMS = {"yes", "true", "y", "checked", "marked", "selected", "x", "✓", "✗", "1"}
NO_SYNONYMS  = {"no", "false", "n", "unchecked", "unmarked", "not selected", "blank", "", "0"}


def normalize_value(v: str) -> str:
    """Normalize a single value for fuzzy comparison."""
    if v is None: return "null"
    s = str(v).strip().lower()
    # null synonyms
    if s in NULL_SYNONYMS: return "null"
    # boolean synonyms
    if s in YES_SYNONYMS: return "yes"
    if s in NO_SYNONYMS:  return "no"
    # currency / number normalization
    s_no_curr = re.sub(r"[$£€¥,]", "", s)
    if re.fullmatch(r"-?\d+(\.\d+)?%?", s_no_curr):
        # canonicalize: strip trailing zeros after decimal
        try:
            f = float(s_no_curr.rstrip("%"))
            if f == int(f): return str(int(f)) + ("%" if "%" in s_no_curr else "")
            return f"{f:g}" + ("%" if "%" in s_no_curr else "")
        except ValueError:
            pass
    # date normalization (best-effort)
    for fmt_in, fmt_out in (
        ("%m/%d/%Y", "%Y-%m-%d"),
        ("%m/%d/%y", "%Y-%m-%d"),
        ("%Y-%m-%d", "%Y-%m-%d"),
        ("%B %d, %Y", "%Y-%m-%d"),
        ("%b %d, %Y", "%Y-%m-%d"),
        ("%b %d %Y", "%Y-%m-%d"),
        ("%d %B %Y", "%Y-%m-%d"),
        ("%d-%b-%Y", "%Y-%m-%d"),
    ):
        try:
            from datetime import datetime
            return datetime.strptime(s, fmt_in).strftime(fmt_out)
        except Exception:
            continue
    # strip trailing punctuation, collapse whitespace
    s = re.sub(r"\s+", " ", s).strip(" .,;:!?")
    return s


def parse_kv(s: str) -> dict[str, str] | None:
    """Parse 'k1=v1; k2=v2' into {k1: v1, ...}. None if not key=val format."""
    if "=" not in s: return None
    parts = [p.strip() for p in s.split(";")]
    out = {}
    for p in parts:
        if "=" not in p: return None
        k, v = p.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


def fuzzy_match(answer, gold, fuzzy_threshold: int = 90) -> bool:
    """Per-key fuzzy match. Returns True iff key sets match AND each value
    matches under normalize_value OR rapidfuzz token_set_ratio >= threshold."""
    if answer is None or gold is None:
        # null vs null OK
        return (answer in (None, "null") or normalize_value(str(answer)) == "null") and \
               (gold in (None, "null") or normalize_value(str(gold)) == "null")
    a_norm = normalize_value(str(answer))
    g_norm = normalize_value(str(gold))
    # both single-value
    a_kv = parse_kv(str(answer))
    g_kv = parse_kv(str(gold))
    if g_kv is None and a_kv is None:
        if a_norm == g_norm: return True
        return fuzz.token_set_ratio(a_norm, g_norm) >= fuzzy_threshold
    if g_kv is None or a_kv is None:
        return False  # one is kv, other is scalar → no match
    # both are kv
    if set(a_kv.keys()) != set(g_kv.keys()):
        return False
    for k in g_kv:
        a_v = normalize_value(a_kv[k])
        g_v = normalize_value(g_kv[k])
        if a_v == g_v: continue
        if fuzz.token_set_ratio(a_v, g_v) >= fuzzy_threshold: continue
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────
# Layer 3 — LLM judge (Gemini 3 Flash)
# ──────────────────────────────────────────────────────────────────────────
JUDGE_PROMPT = """You are grading a structured-data extraction benchmark.

Determine if the PARSER's answer is semantically equivalent to the GOLD answer.

The gold format is plain-string:
  "key1=value1; key2=value2; ..."  (paired values)
  "value1; value2; value3"          (ordered values, keys implied)

A MATCH means:
- Same set of keys (or same number of ordered items in same positions)
- Each value semantically equals the gold value, ignoring formatting:
  * Currency symbols ($1,234.56 ≈ 1234.56)
  * Decimal precision when gold is whole (5 ≈ 5.0)
  * Date format (01/15/2024 ≈ Jan 15, 2024 ≈ 2024-01-15)
  * Case (Apple Inc. ≈ apple inc.)
  * Trailing punctuation, extra whitespace
  * Synonymous values: yes/Yes/Y/checked/marked/✓/X; no/No/N/unchecked; null/None/N/A/blank
  * Minor abbreviations (Inc. vs Inc, U.S. vs US)

NOT a match:
- Any key missing or extra
- Different number value (12 vs 13)
- Different proper noun (Apple vs Amazon)
- Different date by more than format (Jan 15 vs Jan 16)
- Answer is "null" but gold has a non-null value (or vice versa)
- Answer omits a value the gold has

Return ONLY JSON: {"match": <true|false>, "reasoning": "<one short sentence>"}"""


def llm_judge(answer, gold, question: str, model: str = "gemini-3-flash-preview") -> dict:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"match": False, "reasoning": "no GEMINI_API_KEY", "error": True}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": JUDGE_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text":
            f"Question: {question}\n\nGOLD: {gold}\nPARSER ANSWER: {answer}"
        }]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 256,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    try:
        with httpx.Client(timeout=60) as client:
            r = client.post(url, params={"key": api_key}, json=body)
            if r.status_code != 200:
                return {"match": False, "reasoning": f"http {r.status_code}", "error": True}
            data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m: return {"match": False, "reasoning": "no JSON in response", "error": True}
        obj = json.loads(m.group(0))
        return {"match": bool(obj.get("match")), "reasoning": obj.get("reasoning", "")}
    except Exception as e:
        return {"match": False, "reasoning": f"{type(e).__name__}: {e}", "error": True}


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=REPO / "runs" / "pilot_50")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--no-llm", action="store_true",
                    help="skip the LLM judge layer (only strict + fuzzy)")
    args = ap.parse_args()
    _set_run_dir(args.run_dir.resolve())

    eval_data = json.loads((PILOT / "eval" / "eval_r1.json").read_text())
    # Cache for LLM judge — per (qid, parser, sub-answer key)
    judge_cache_dir = PILOT / "eval" / "llm_judge_struct"
    judge_cache_dir.mkdir(parents=True, exist_ok=True)

    PARSERS = ["extend_performance_v1_0_0", "extend_performance_v2_0_0_advanced",
               "extend_performance_v2_0_0_agentic",
               "aws_textract", "azure_di", "reducto",
               "llamaparse", "llamaparse_cost_effective"]

    def safe_key(k: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", str(k))[:60] or "_"

    # ── Layers 1 + 2 (strict, fuzzy) — per sub-answer key, free ──────────────
    for dp, page in eval_data.items():
        for qid, q in page["questions"].items():
            gs = q["gold_struct"]
            keys = q["keys"]
            q["sub_correct_strict"] = {k: [] for k in keys}
            q["sub_correct_fuzzy"] = {k: [] for k in keys}
            for p, ans in q["parser_answers"].items():
                for k in keys:
                    av, gv = ans.get(k), gs[k]
                    s = strict_match(av, gv)
                    if s:
                        q["sub_correct_strict"][k].append(p)
                    if s or fuzzy_match(av, gv):
                        q["sub_correct_fuzzy"][k].append(p)

    # ── Layer 3 (LLM judge) — only sub-answers where fuzzy STILL says wrong ──
    llm_jobs = []  # (qid, parser, key, answer_val, gold_val, cache_path)
    for dp, page in eval_data.items():
        for qid, q in page["questions"].items():
            gs = q["gold_struct"]
            for p, ans in q["parser_answers"].items():
                for k in q["keys"]:
                    if p in q["sub_correct_fuzzy"][k]:
                        continue
                    cache_path = judge_cache_dir / f"{qid}__{p}__{safe_key(k)}.json"
                    if cache_path.exists():
                        continue
                    llm_jobs.append((qid, p, k, ans.get(k), gs[k], cache_path))

    if not args.no_llm and llm_jobs:
        bank = json.loads((PILOT / "qa" / "bank_r1.json").read_text())
        bank_by_qid = {it["question_id"]: it for it in bank["items"]}
        print(f"LLM judge: {len(llm_jobs)} (parser, Q, sub-answer) cells "
              f"where fuzzy still says wrong", flush=True)

        def do_judge(job):
            qid, p, k, av, gv, cache_path = job
            q_text = bank_by_qid.get(qid, {}).get("question", "")
            result = llm_judge(av, gv, f"{q_text}\n[Grading only sub-answer key: {k}]")
            cache_path.write_text(json.dumps(result, indent=2))
            return job, result

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {ex.submit(do_judge, j): j for j in llm_jobs}
            done = 0
            for fut in as_completed(futures):
                fut.result()
                done += 1
                if done % 50 == 0 or done == len(llm_jobs):
                    print(f"  [{done}/{len(llm_jobs)}]  ({(time.time()-t0)/60:.1f}min)", flush=True)
    elif args.no_llm:
        print("Skipping LLM judge layer (--no-llm)")

    # ── Aggregate LLM verdicts + roll up to question level ───────────────────
    for dp, page in eval_data.items():
        for qid, q in page["questions"].items():
            keys = q["keys"]
            q["sub_correct_llm"] = {k: list(q["sub_correct_fuzzy"][k]) for k in keys}
            for p in PARSERS:
                if p not in q["parser_answers"]:
                    continue
                for k in keys:
                    if p in q["sub_correct_fuzzy"][k]:
                        continue
                    cache_path = judge_cache_dir / f"{qid}__{p}__{safe_key(k)}.json"
                    if cache_path.exists():
                        verdict = json.loads(cache_path.read_text())
                        if verdict.get("match"):
                            q["sub_correct_llm"][k].append(p)
            # question-level: a parser is correct iff correct on ALL keys
            for layer in ("strict", "fuzzy", "llm"):
                sub = q[f"sub_correct_{layer}"]
                q[f"parsers_correct_{layer}"] = [
                    p for p in q["parser_answers"]
                    if all(p in sub[k] for k in keys)
                ]

    # Write
    out_path = PILOT / "eval" / "eval_r1_rescored.json"
    out_path.write_text(json.dumps(eval_data, indent=2, ensure_ascii=False))

    # ── Summary: sub-answer accuracy (micro-avg) + question-level ────────────
    print(f"\n=== rescored accuracy — per sub-answer (micro-avg) ===")
    print(f"{'parser':<38} {'strict':>13} {'fuzzy':>13} {'llm':>13}")
    for p in PARSERS:
        sc = {"strict": 0, "fuzzy": 0, "llm": 0}
        st = 0
        for page in eval_data.values():
            for q in page["questions"].values():
                if p not in q["parser_answers"]:
                    continue
                for k in q["keys"]:
                    st += 1
                    for layer in sc:
                        if p in q[f"sub_correct_{layer}"][k]:
                            sc[layer] += 1
        if st:
            print(f"  {p:<36} "
                  f"{sc['strict']:>5}/{st} ({sc['strict']/st*100:>3.0f}%) "
                  f"{sc['fuzzy']:>5}/{st} ({sc['fuzzy']/st*100:>3.0f}%) "
                  f"{sc['llm']:>5}/{st} ({sc['llm']/st*100:>3.0f}%)")

    print(f"\n=== rescored accuracy — question-level (all keys correct) ===")
    print(f"{'parser':<38} {'strict':>13} {'fuzzy':>13} {'llm':>13}")
    for p in PARSERS:
        c = {"strict": 0, "fuzzy": 0, "llm": 0}
        t = 0
        for page in eval_data.values():
            for q in page["questions"].values():
                if p not in q["parser_answers"]:
                    continue
                t += 1
                for layer in c:
                    if p in q[f"parsers_correct_{layer}"]:
                        c[layer] += 1
        if t:
            print(f"  {p:<36} "
                  f"{c['strict']:>5}/{t} ({c['strict']/t*100:>3.0f}%) "
                  f"{c['fuzzy']:>5}/{t} ({c['fuzzy']/t*100:>3.0f}%) "
                  f"{c['llm']:>5}/{t} ({c['llm']/t*100:>3.0f}%)")

    print(f"\nwritten: {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
