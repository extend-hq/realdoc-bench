#!/usr/bin/env python3
"""Score parser extractions with fuzzy normalization + LLM judge for hard cases.

Pipeline per (doc, field, parser):
  1. Apply CHEAP fuzzy normalization to each value:
     - Strings: lowercase, strip punct, collapse whitespace
     - Numbers: parse to float, round to 2 decimals
     - Dates: parse with dateutil, normalize to YYYY-MM-DD
     - Booleans: stay as-is
  2. Compute consensus across parsers using fuzzy-normalized values
  3. For (parser, consensus) pairs that DISAGREE after fuzzy norm:
     - Optionally call LLM judge: "are these semantically equivalent?"
     - Cache decisions per (field_name, val_a, val_b)

Output: same as compare_parser_extractions.py PLUS a fuzzy column.

Usage:
  python -u scripts/score_with_fuzzy_judge.py --sweep-dir runs/parse_sweep/<ts>
  python -u scripts/score_with_fuzzy_judge.py … --source qa   # use _qa_extractions
  python -u scripts/score_with_fuzzy_judge.py … --no-judge    # fuzzy only, no LLM
  python -u scripts/score_with_fuzzy_judge.py … --judge-budget 200  # max LLM judge calls
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
V1_1 = REPO / "datasets" / "extraction" / "v1.1"
SCHEMAS_DIR = V1_1 / "_schemas"


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


_anthropic = None
def get_anthropic():
    global _anthropic
    if _anthropic is None:
        from anthropic import Anthropic
        _anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _anthropic


# =====================================================================
# Fuzzy normalization
# =====================================================================

_PUNCT_RE = re.compile(r"[^\w\s$%./-]")  # keep digits/letters/space/$/%/./-/
_WS_RE = re.compile(r"\s+")
_CURRENCY_RE = re.compile(r"^[\$\s]*([0-9,]+(?:\.\d+)?)\s*(?:usd|dollars?)?$", re.IGNORECASE)
_PERCENT_RE = re.compile(r"^([0-9.]+)\s*%$")

_DATE_FORMATS = [
    "%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y",
    "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d",
    "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
    "%Y%m%d",
]


def _try_parse_date(s: str) -> str | None:
    """Return YYYY-MM-DD or None."""
    from datetime import datetime
    s = s.strip()
    if len(s) < 4 or len(s) > 30:
        return None
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    # last resort: dateutil if available
    try:
        from dateutil import parser as du
        dt = du.parse(s, fuzzy=False)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def _try_parse_number(s: str) -> float | None:
    s = s.strip()
    # currency
    m = _CURRENCY_RE.match(s)
    if m:
        try:
            return round(float(m.group(1).replace(",", "")), 2)
        except ValueError:
            pass
    m = _PERCENT_RE.match(s)
    if m:
        try:
            return round(float(m.group(1)) / 100, 4)
        except ValueError:
            pass
    # plain number with commas
    s2 = s.replace(",", "")
    try:
        return round(float(s2), 2)
    except ValueError:
        return None


def fuzzy_normalize(v):
    """Return a canonical comparable representation.
    Returns:
      None for null/empty
      bool for booleans
      ('num', float) for numbers
      ('date', 'YYYY-MM-DD') for dates
      ('list', n) for arrays (loose — count only)
      ('dict', n) for objects
      ('str', canonical_str) for free strings
    """
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (int, float)):
        return ("num", round(float(v), 2))
    if isinstance(v, list):
        return ("list", len(v))
    if isinstance(v, dict):
        return ("dict", len(v))
    if not isinstance(v, str):
        return ("str", str(v)[:200])

    s = v.strip()
    if not s:
        return None
    if s.lower() in {"null", "none", "n/a", "na", "-", "—", "not specified", "not provided"}:
        return None

    # numeric?
    n = _try_parse_number(s)
    if n is not None:
        return ("num", n)

    # date? only try if it looks date-ish
    if re.search(r"\d", s) and any(c in s for c in "/-,") and not s.startswith("$"):
        d = _try_parse_date(s)
        if d:
            return ("date", d)

    # string: lowercase, strip non-essential punct, collapse ws
    canon = _PUNCT_RE.sub(" ", s.lower())
    canon = _WS_RE.sub(" ", canon).strip()
    return ("str", canon[:300]) if canon else None


# =====================================================================
# Load extractions (tool-use OR QA)
# =====================================================================

def load_extractions(sweep_dir: Path, source: str) -> dict:
    """source ∈ {'tooluse', 'qa'}
       Returns {(domain,type,fname) → {parser → {field → value}}}"""
    out: dict = defaultdict(lambda: defaultdict(dict))
    subdir = "_extractions" if source == "tooluse" else "_qa_extractions"
    parsers = set()
    for parser_dir in sorted(sweep_dir.iterdir()):
        if not (parser_dir.is_dir() or parser_dir.is_symlink()):
            continue
        parser = parser_dir.name
        if parser == "comparison":
            continue
        ext_root = parser_dir / subdir
        if not ext_root.exists():
            continue
        for jpath in ext_root.rglob("*.json"):
            if jpath.name.endswith(".meta.json"):
                continue
            rel = jpath.relative_to(ext_root)
            if len(rel.parts) < 3:
                continue
            domain, typ, fname = rel.parts[0], rel.parts[1], rel.parts[2].replace(".json", "")
            try:
                blob = json.loads(jpath.read_text())
            except Exception:
                continue
            if not isinstance(blob, dict):
                continue
            # QA format wraps each value as {answer, evidence, confidence}; unwrap
            if source == "qa":
                normalized: dict = {}
                for k, v in blob.items():
                    if isinstance(v, dict) and "answer" in v:
                        normalized[k] = v["answer"]
                    else:
                        normalized[k] = v
                blob = normalized
            out[(domain, typ, fname)][parser] = blob
            parsers.add(parser)
    return dict(out), sorted(parsers)


# =====================================================================
# LLM judge (cached)
# =====================================================================

JUDGE_PROMPT = """You're a judge deciding whether two extracted values mean the same thing for the same document field.

Field: {field_name}
Field context: {field_context}

Value A: {value_a}
Value B: {value_b}

Are these SEMANTICALLY equivalent (i.e., they describe the same fact, just differently formatted or phrased)? Tiny formatting differences count as equivalent.

Reply with ONLY one word: "yes" or "no".
"""


def judge_equivalent(client, field_name: str, field_context: str, val_a: str, val_b: str,
                     cache: dict, model: str = "claude-sonnet-4-5") -> bool:
    """Cached LLM judge. Returns True iff val_a ≈ val_b semantically."""
    # Order-insensitive cache key
    key_pair = tuple(sorted([str(val_a)[:200], str(val_b)[:200]]))
    cache_key = (field_name, key_pair)
    if cache_key in cache:
        return cache[cache_key]
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=8,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                field_name=field_name,
                field_context=field_context[:200] or "(none)",
                value_a=str(val_a)[:400],
                value_b=str(val_b)[:400],
            )}],
        )
        out = resp.content[0].text.strip().lower()
        verdict = out.startswith("y")
    except Exception:
        verdict = False
    cache[cache_key] = verdict
    return verdict


# =====================================================================
# Main scoring
# =====================================================================

def load_field_contexts(schemas_dir: Path) -> dict[tuple[str, str], str]:
    """Returns {(type, field) → description string} from schemas."""
    out: dict = {}
    for sp in schemas_dir.glob("*.json"):
        typ = sp.stem
        try:
            sch = json.loads(sp.read_text())
        except Exception:
            continue
        for fname, fdef in (sch.get("properties") or {}).items():
            if isinstance(fdef, dict):
                desc = fdef.get("description", "")
                out[(typ, fname)] = desc[:300]
    return out


def compute(data: dict, parsers: list[str], field_contexts: dict,
            use_judge: bool, judge_budget: int) -> tuple[list[dict], dict, dict]:
    """Returns (field_rows, parser_stats, judge_cache)."""
    judge_cache: dict = {}
    judge_calls = 0
    client = get_anthropic() if use_judge else None

    field_rows: list[dict] = []
    parser_agreement: dict = defaultdict(list)
    parser_nonnull: dict = defaultdict(int)
    parser_total: dict = defaultdict(int)

    for (domain, typ, fname), parser_outputs in data.items():
        all_fields: set = set()
        for p in parsers:
            if p in parser_outputs and isinstance(parser_outputs[p], dict):
                all_fields.update(parser_outputs[p].keys())
        for fld in all_fields:
            # Fuzzy normalize each parser's value
            norm_vals: dict = {}
            raw_vals: dict = {}
            for p in parsers:
                if p not in parser_outputs:
                    continue
                v = parser_outputs[p].get(fld)
                raw_vals[p] = v
                norm_vals[p] = fuzzy_normalize(v)

            # Cluster non-null normalized values
            non_null = [(p, v) for p, v in norm_vals.items() if v is not None]
            tally = Counter(v for _, v in non_null)
            consensus = None
            if non_null:
                top, c = tally.most_common(1)[0]
                if c * 2 > len(non_null):
                    consensus = top

            # Judge pass: for parsers that disagree with consensus, ask LLM if they're equivalent
            # If yes, upgrade them to "agree"
            for p in parsers:
                parser_total[p] += 1
                v = norm_vals.get(p)
                if v is None:
                    parser_agreement[p].append(None)
                    continue
                parser_nonnull[p] += 1
                if consensus is None:
                    parser_agreement[p].append(None)
                    continue
                if v == consensus:
                    parser_agreement[p].append(1)
                    continue
                # Disagrees — try judge if budget remains
                if use_judge and judge_calls < judge_budget and client is not None:
                    judge_calls += 1
                    ctx = field_contexts.get((typ, fld), "")
                    raw_v = raw_vals.get(p)
                    raw_c = None
                    # find raw value of any parser that matches consensus
                    for p2, n2 in norm_vals.items():
                        if n2 == consensus:
                            raw_c = raw_vals.get(p2)
                            break
                    eq = judge_equivalent(client, fld, ctx, raw_v, raw_c, judge_cache)
                    parser_agreement[p].append(1 if eq else 0)
                else:
                    parser_agreement[p].append(0)

            field_rows.append({
                "domain": domain, "type": typ, "filename": fname, "field": fld,
                "n_parsers": len(non_null),
                "n_unique_fuzzy": len(set(v for _, v in non_null)),
                "has_consensus": consensus is not None,
                "consensus_str": str(consensus)[:200] if consensus is not None else "",
            })

    parser_stats: dict = {}
    for p in parsers:
        agr = parser_agreement[p]
        graded = [a for a in agr if a is not None]
        parser_stats[p] = {
            "fields_seen": parser_total[p],
            "fields_non_null": parser_nonnull[p],
            "non_null_rate": (parser_nonnull[p] / parser_total[p]) if parser_total[p] else 0,
            "agreement_rate": (sum(graded) / len(graded)) if graded else 0,
            "n_graded": len(graded),
        }
    return field_rows, parser_stats, {"judge_calls": judge_calls, "cache_size": len(judge_cache)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--source", choices=["tooluse", "qa", "both"], default="both",
                    help="Which extractions to score. 'both' produces side-by-side report.")
    ap.add_argument("--no-judge", action="store_true",
                    help="Skip LLM judge — fuzzy-only scoring")
    ap.add_argument("--judge-budget", type=int, default=500,
                    help="Max LLM judge calls (cached, so unique pairs only)")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    out_dir = sweep_dir / "comparison_fuzzy"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading field contexts from schemas...")
    field_contexts = load_field_contexts(SCHEMAS_DIR)
    print(f"  {len(field_contexts)} (type, field) descriptions loaded")

    sources = ["tooluse", "qa"] if args.source == "both" else [args.source]
    all_results: dict = {}

    for source in sources:
        print(f"\n=== Source: {source} ===")
        data, parsers = load_extractions(sweep_dir, source)
        if not data:
            print(f"  (no extractions found for source={source}, skipping)")
            continue
        print(f"  {len(data)} docs, {len(parsers)} parsers")
        t0 = time.time()
        field_rows, parser_stats, judge_info = compute(
            data, parsers, field_contexts,
            use_judge=not args.no_judge,
            judge_budget=args.judge_budget,
        )
        wall = time.time() - t0
        print(f"  scored in {wall:.0f}s (judge calls: {judge_info['judge_calls']}, "
              f"cached pairs: {judge_info['cache_size']})")
        all_results[source] = {
            "field_rows": field_rows,
            "parser_stats": parser_stats,
            "parsers": parsers,
            "judge_info": judge_info,
        }

        # Per-source CSV
        csv_path = out_dir / f"per_parser_{source}.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["parser", "fields_seen", "fields_non_null",
                                               "non_null_rate", "agreement_rate", "n_graded"])
            w.writeheader()
            for p in parsers:
                s = parser_stats[p]
                w.writerow({
                    "parser": p,
                    "fields_seen": s["fields_seen"],
                    "fields_non_null": s["fields_non_null"],
                    "non_null_rate": round(s["non_null_rate"], 4),
                    "agreement_rate": round(s["agreement_rate"], 4),
                    "n_graded": s["n_graded"],
                })

    # Combined report
    idx = out_dir / "INDEX.md"
    with idx.open("w") as f:
        f.write(f"# Cross-parser comparison — fuzzy + judge — {sweep_dir.name}\n\n")
        f.write(f"Judge: {'enabled' if not args.no_judge else 'DISABLED'}, "
                f"budget: {args.judge_budget} per source.\n\n")
        for source in sources:
            if source not in all_results:
                continue
            r = all_results[source]
            f.write(f"## Source: `{source}` extractions\n\n")
            f.write(f"_{len(r['parsers'])} parsers · "
                    f"{r['judge_info']['judge_calls']} judge calls · "
                    f"{r['judge_info']['cache_size']} unique pairs cached_\n\n")
            f.write("| Parser | Fields seen | Non-null % | Agreement (fuzzy+judge) |\n")
            f.write("|---|---:|---:|---:|\n")
            rows = sorted(r["parsers"], key=lambda p: -r["parser_stats"][p]["agreement_rate"])
            for p in rows:
                s = r["parser_stats"][p]
                f.write(f"| `{p}` | {s['fields_seen']:,} | "
                        f"{s['non_null_rate']:.1%} | "
                        f"{s['agreement_rate']:.1%} ({s['n_graded']:,}) |\n")
            f.write("\n")

        # Side-by-side delta if both sources
        if len(all_results) == 2:
            f.write("## Tool-use vs QA — agreement delta per parser\n\n")
            f.write("| Parser | Tool-use agree | QA agree | Δ |\n|---|---:|---:|---:|\n")
            tu = all_results["tooluse"]["parser_stats"]
            qa = all_results["qa"]["parser_stats"]
            for p in sorted(set(tu) | set(qa)):
                t = tu.get(p, {}).get("agreement_rate", 0)
                q = qa.get(p, {}).get("agreement_rate", 0)
                delta = q - t
                sign = "+" if delta >= 0 else ""
                f.write(f"| `{p}` | {t:.1%} | {q:.1%} | {sign}{delta * 100:.1f}pp |\n")

    print(f"\nWrote {idx.relative_to(REPO)}")
    # Console headline
    for source in sources:
        if source not in all_results:
            continue
        r = all_results[source]
        print(f"\n=== HEADLINE — {source} ===")
        print(f"  {'parser':32} {'nonnull':>8} {'agree':>8}")
        rows = sorted(r["parsers"], key=lambda p: -r["parser_stats"][p]["agreement_rate"])
        for p in rows:
            s = r["parser_stats"][p]
            print(f"  {p:32} {s['non_null_rate']:>7.1%}  {s['agreement_rate']:>7.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
