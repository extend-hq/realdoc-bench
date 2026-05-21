#!/usr/bin/env python3
"""Re-score the eval with 3 layered strategies:

  1. strict   — normalize_exact (the raw eval_r1.json score)
  2. fuzzy    — per-value compare via the shared scoring engine
                (realdoc_bench.extraction.scoring.compare) under a profile
  3. llm_judge — only for cells strict + fuzzy disagree on; Gemini adjudicates
                (realdoc_bench.extraction.scoring.llm_judge)

The fuzzy matcher and the LLM judge both live in the realdoc_bench scoring
library now — this script no longer carries its own copies. Pick the fuzzy
strictness with ``--profile {default,strict,lenient}``.

Output: runs/<run>/eval/eval_r1_rescored.json — adds
  parsers_correct_strict / _fuzzy / _llm  (+ the per-sub-answer sets).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from realdoc_bench.extraction.scoring.compare import compare_text          # noqa: E402
from realdoc_bench.extraction.scoring.llm_judge import llm_judge           # noqa: E402
from realdoc_bench.extraction.scoring.profiles import (                    # noqa: E402
    DEFAULT_PROFILE,
    PROFILES,
    ScoringProfile,
)

PILOT = REPO / "runs" / "pilot_50"


def _set_run_dir(run_dir: Path) -> None:
    global PILOT
    PILOT = run_dir


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


# ──────────────────────────────────────────────────────────────────────────
# Layer 1 — strict
# ──────────────────────────────────────────────────────────────────────────
def normalize_strict(value: str | None) -> str:
    if value is None:
        return "null"
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
# Layer 2 — fuzzy. Value canonicalization (synonyms / currency / dates) is
# benchmark policy and stays here; the actual exact/fuzzy string comparison
# is delegated to the shared scoring engine (scoring.compare.compare_text).
# ──────────────────────────────────────────────────────────────────────────
NULL_SYNONYMS = {"null", "none", "n/a", "na", "n.a.", "blank", "", "—", "-"}
YES_SYNONYMS = {"yes", "true", "y", "checked", "marked", "selected", "x", "✓", "✗", "1"}
NO_SYNONYMS = {"no", "false", "n", "unchecked", "unmarked", "not selected", "blank", "", "0"}


def normalize_value(v: str) -> str:
    """Canonicalize one value for comparison: null / boolean synonyms,
    currency + number, best-effort dates."""
    if v is None:
        return "null"
    s = str(v).strip().lower()
    if s in NULL_SYNONYMS:
        return "null"
    if s in YES_SYNONYMS:
        return "yes"
    if s in NO_SYNONYMS:
        return "no"
    s_no_curr = re.sub(r"[$£€¥,]", "", s)
    if re.fullmatch(r"-?\d+(\.\d+)?%?", s_no_curr):
        try:
            f = float(s_no_curr.rstrip("%"))
            if f == int(f):
                return str(int(f)) + ("%" if "%" in s_no_curr else "")
            return f"{f:g}" + ("%" if "%" in s_no_curr else "")
        except ValueError:
            pass
    for fmt_in in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y",
                   "%b %d %Y", "%d %B %Y", "%d-%b-%Y"):
        try:
            from datetime import datetime
            return datetime.strptime(s, fmt_in).strftime("%Y-%m-%d")
        except Exception:
            continue
    return re.sub(r"\s+", " ", s).strip(" .,;:!?")


def parse_kv(s: str) -> dict[str, str] | None:
    """Parse 'k1=v1; k2=v2' into {k1: v1, ...}. None if not key=val format."""
    if "=" not in s:
        return None
    out = {}
    for p in (p.strip() for p in s.split(";")):
        if "=" not in p:
            return None
        k, v = p.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


def _value_match(answer, gold, profile: ScoringProfile) -> bool:
    a_norm = normalize_value(str(answer))
    g_norm = normalize_value(str(gold))
    if a_norm == g_norm:
        return True
    score, _ = compare_text(a_norm, g_norm, profile=profile)
    return score >= 1.0


def fuzzy_match(answer, gold, *, profile: ScoringProfile = DEFAULT_PROFILE) -> bool:
    """Per-key fuzzy match: key sets must agree and every value must match
    under :func:`_value_match` (canonicalize → shared compare_text)."""
    if answer is None or gold is None:
        return ((answer in (None, "null") or normalize_value(str(answer)) == "null")
                and (gold in (None, "null") or normalize_value(str(gold)) == "null"))
    a_kv = parse_kv(str(answer))
    g_kv = parse_kv(str(gold))
    if g_kv is None and a_kv is None:
        return _value_match(answer, gold, profile)
    if g_kv is None or a_kv is None:
        return False
    if set(a_kv.keys()) != set(g_kv.keys()):
        return False
    return all(_value_match(a_kv[k], g_kv[k], profile) for k in g_kv)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=REPO / "runs" / "pilot_50")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--profile", choices=list(PROFILES), default="default",
                    help="fuzzy-match strictness profile (scoring/profiles.py)")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip the LLM judge layer (only strict + fuzzy)")
    args = ap.parse_args()
    _set_run_dir(args.run_dir.resolve())
    profile = PROFILES[args.profile]
    print(f"fuzzy profile: {args.profile} "
          f"(threshold={profile.fuzzy_threshold}, floor={profile.fuzzy_floor})")

    eval_data = json.loads((PILOT / "eval" / "eval_r1.json").read_text())
    judge_cache_dir = PILOT / "eval" / "llm_judge_struct"
    judge_cache_dir.mkdir(parents=True, exist_ok=True)

    PARSERS = ["extend_performance_v1_0_0", "extend_performance_v2_0_0_advanced",
               "extend_performance_v2_0_0_agentic",
               "aws_textract", "azure_di", "reducto",
               "llamaparse", "llamaparse_cost_effective"]

    def safe_key(k: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", str(k))[:60] or "_"

    # ── Layers 1 + 2 (strict, fuzzy) — per sub-answer key, free ──────────────
    for page in eval_data.values():
        for q in page["questions"].values():
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
                    if s or fuzzy_match(av, gv, profile=profile):
                        q["sub_correct_fuzzy"][k].append(p)

    # ── Layer 3 (LLM judge) — only sub-answers where fuzzy STILL says wrong ──
    llm_jobs = []  # (qid, parser, key, answer_val, gold_val, cache_path)
    for page in eval_data.values():
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
                    print(f"  [{done}/{len(llm_jobs)}]  ({(time.time()-t0)/60:.1f}min)",
                          flush=True)
    elif args.no_llm:
        print("Skipping LLM judge layer (--no-llm)")

    # ── Aggregate LLM verdicts + roll up to question level ───────────────────
    for page in eval_data.values():
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
            for layer in ("strict", "fuzzy", "llm"):
                sub = q[f"sub_correct_{layer}"]
                q[f"parsers_correct_{layer}"] = [
                    p for p in q["parser_answers"]
                    if all(p in sub[k] for k in keys)
                ]

    out_path = PILOT / "eval" / "eval_r1_rescored.json"
    out_path.write_text(json.dumps(eval_data, indent=2, ensure_ascii=False))

    # ── Summary ──────────────────────────────────────────────────────────────
    for scope, title in (("sub", "per sub-answer (micro-avg)"),
                         ("q", "question-level (all keys correct)")):
        print(f"\n=== rescored accuracy — {title} ===")
        print(f"{'parser':<38} {'strict':>13} {'fuzzy':>13} {'llm':>13}")
        for p in PARSERS:
            c = {"strict": 0, "fuzzy": 0, "llm": 0}
            t = 0
            for page in eval_data.values():
                for q in page["questions"].values():
                    if p not in q["parser_answers"]:
                        continue
                    if scope == "sub":
                        for k in q["keys"]:
                            t += 1
                            for layer in c:
                                if p in q[f"sub_correct_{layer}"][k]:
                                    c[layer] += 1
                    else:
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
