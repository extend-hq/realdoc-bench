"""Score stage — one Gemini call per (question, parser); typed deep_equal.

For every (item, parser) pair: load `<run-dir>/parses/<parser>/<stem>.md`,
send Gemini the question + the item's typed template + the markdown, parse the
typed JSON answer back, score per-field with `deep_equal` (with `fuzzy_equal`
on plain `<string>` fields), write
`<run-dir>/eval/cache/<qid>__<parser>.json`.

Cache key: `(qid, parser)`. On hit, the cached answer is **re-scored** with
the current `score_typed` and the cache file is rewritten if the verdict
changed — so editing scoring code + re-running `score` updates verdicts for
free (no model calls). `--force` re-calls Gemini.

When `parsers` is empty, defaults to every parser that has parses under
`<run-dir>/parses/` (the natural chain after `parse`).
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from rapidfuzz import fuzz

from realdoc_bench.evaluate.jsonify import (
    _placeholder_to_type_hint,
    render_template,
    widen_types_from_gold,
)
from realdoc_bench.evaluate.runs import RunLayout


# ── Constants — tunable scoring knobs ──────────────────────────────────────
DEFAULT_MODEL = "gemini-3-flash-preview"
FUZZ_THRESHOLD = 92      # rapidfuzz ratio above which strings count as equal
FUZZ_MIN_WORDS = 5       # min word count on EITHER side for fuzzy to apply

# ── Extractor system prompt — the canonical typed-extraction prompt ────────
SYSTEM_PROMPT = (
    "You are extracting structured answers from document-parser markdown. "
    "Read the markdown only -- never use outside knowledge. "
    "Return ONE JSON object whose keys and value types exactly match the answer "
    "template provided. The TYPE HINT inside `< >` is the source of truth for "
    "each field; the KEY NAME does not change typing or formatting. Rules: "
    "(1) numbers must be JSON numbers (no commas, no '$', no '%', no quotes); "
    "(2) only fields whose template says `<boolean>` may be JSON true / false; "
    "(3) fields whose template says `<one of: a | b | ...>` MUST return one of "
    "the listed tokens as a JSON string, copied verbatim including case; "
    "(4) ONLY fields whose type hint is `<date string YYYY-MM-DD>` get "
    "normalized to ISO YYYY-MM-DD; "
    "(5) all other fields -- including any `<string>` -- copy the page text "
    "verbatim, trimmed (do not normalize dates, numbers, capitalization or "
    "separators), EXCEPT collapse spurious OCR character-spacing: if a value "
    "is printed with stray spaces between its characters (e.g. '0 . 0 0', "
    "'1 2 3 4', 'C A T'), return its natural un-spaced form; "
    "(6) for any field whose value is absent, blank, illegible, or genuinely "
    "ambiguous in the markdown, use JSON null; "
    "(6a) treat redaction markers like `[Redacted]` as ABSENT -- return JSON "
    "null if a cell holds only a marker; drop the marker if mixed with real "
    "content. "
    "Do not include keys outside the template. Do not wrap in an 'answer' "
    "parent key beyond what the template shows. Output the JSON object only, "
    "no prose or fences."
)


# ── String normalization for deep_equal ────────────────────────────────────
_WS_AROUND_SEP = re.compile(r"\s*([/\-,:])\s*")
_WS_RUN = re.compile(r"\s+")
_DASH_RE = re.compile(r"[‐‑‒–—―−]")
_SMART_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"'})
_TAG_RE = re.compile(r"<br\s*/?>|</?[a-z][^>]*>", re.I)
_STYLE_RE = re.compile(r"\*\*|__|`")


def _norm_str(s: str) -> str:
    s = s.translate(_SMART_QUOTES)
    s = _DASH_RE.sub("-", s)
    s = _TAG_RE.sub(" ", s)
    s = _STYLE_RE.sub("", s)
    s = _WS_AROUND_SEP.sub(r"\1", s)
    return _WS_RUN.sub(" ", s).strip()


def deep_equal(a: Any, b: Any) -> bool:
    """Type-tolerant equality used as the base comparison for every key.

    Tolerates: parser-style 1-element list wrapping a dict, smart-quote /
    dash variants, parser markdown styling (`<br>`, `**`, ``), OCR
    character-spacing on numeric strings, and number-vs-numeric-string with
    currency / percent / comma formatting (`"$1,234" == 1234`).
    """
    if isinstance(a, list) and len(a) == 1 and isinstance(b, dict) and isinstance(a[0], dict):
        return deep_equal(a[0], b)
    if isinstance(b, list) and len(b) == 1 and isinstance(a, dict) and isinstance(b[0], dict):
        return deep_equal(a, b[0])
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, str) and isinstance(b, str):
        na, nb = _norm_str(a), _norm_str(b)
        if na == nb:
            return True
        # OCR character-spacing — equal once ALL whitespace removed.
        return re.sub(r"\s+", "", na) == re.sub(r"\s+", "", nb)
    if isinstance(a, (int, float)) and isinstance(b, str):
        num, s = a, b
    elif isinstance(b, (int, float)) and isinstance(a, str):
        num, s = b, a
    else:
        return a == b
    try:
        return float(re.sub(r"[,$%\s]", "", s)) == float(num)
    except ValueError:
        return False


def fuzzy_equal(a: Any, b: Any) -> bool:
    """Conservative rapidfuzz comparison — ONLY for plain <string> fields.
    Falls back to deep_equal first; then punctuation-insensitive equality;
    then rapidfuzz ratio above FUZZ_THRESHOLD, but only when both sides are
    multi-word (short strings are IDs/amounts where 90+ ratio would forgive
    a single wrong digit)."""
    if deep_equal(a, b):
        return True
    if not (isinstance(a, str) and isinstance(b, str)):
        return False
    ca = re.sub(r"[^0-9a-z]+", " ", a.lower()).strip()
    cb = re.sub(r"[^0-9a-z]+", " ", b.lower()).strip()
    if ca and ca == cb:
        return True
    if min(len(ca.split()), len(cb.split())) < FUZZ_MIN_WORDS:
        return False
    return bool(ca) and fuzz.ratio(ca, cb) >= FUZZ_THRESHOLD


_TYPE_RE = re.compile(r'"([^"]+)":\s*<([^>]+)>')


def string_keys(template: str) -> set[str]:
    """Keys whose template type is plain <string> — the fuzzy-eligible ones.
    Enums, numbers, booleans and dates stay exact."""
    return {k for k, h in _TYPE_RE.findall(template)
            if h.replace("| null", "").strip() == "string"}


def score_typed(answer: Any, gold_dict: dict,
                str_keys: set[str]) -> tuple[dict, bool]:
    """Per-field score: fuzzy_equal for <string> fields, deep_equal otherwise."""
    fm = {}
    for k, gv in gold_dict.items():
        av = answer.get(k) if isinstance(answer, dict) else None
        fm[k] = fuzzy_equal(av, gv) if k in str_keys else deep_equal(av, gv)
    return fm, (bool(fm) and all(fm.values()))


# ── Template building ──────────────────────────────────────────────────────
_RF_PREFIX = re.compile(r"^\s*Return exactly[^:]*:\s*")
_PAIR_RE = re.compile(r"([^=<>;|]+)=<([^>]+)>")


def build_template(rf: str, gold_dict: dict) -> str:
    """Turn a freeform `response_format` into a typed JSON template.

    Handles three dialects observed in the bank:
      'Return exactly: k=<tok>; k2=<tok>'            semicolon-separated
      'Return exactly (... \" || \"): k=<tok> || ...'   pipe-pair-separated
      'Return ... separated by semicolon-space'      list-style; gold is list

    If neither pattern matches, derives the shape from the gold (list → list of
    strings, scalar → string).
    """
    body = _RF_PREFIX.sub("", rf or "")
    pairs = [(k.strip(), tok) for k, tok in _PAIR_RE.findall(body)]
    if not pairs:
        kt = {k: ("list of strings" if isinstance(v, list) else "string")
              for k, v in (gold_dict or {}).items()}
        return render_template(kt or {"answer": "string"})
    kt = {k: _placeholder_to_type_hint(tok) for k, tok in pairs}
    try:
        widen_types_from_gold(kt, gold_dict or {})
    except Exception:  # noqa: BLE001 — keep going on a bad widen
        pass
    return render_template(kt)


# ── Gemini ─────────────────────────────────────────────────────────────────
def require_api_key() -> str:
    k = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not k:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY not set")
    return k


def gemini_extract(question: str, template: str, markdown: str,
                   *, model: str = DEFAULT_MODEL,
                   client: httpx.Client | None = None) -> Any:
    """One Gemini call. Returns the parsed JSON answer or None on failure."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    prompt = (f"Question:\n{question}\n\n"
              f"Answer template (return JSON with exactly these keys and value "
              f"types):\n{template}\n\n"
              f"Parser markdown:\n```markdown\n{markdown}\n```")
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 8192,
                             "responseMimeType": "application/json"},
    }
    own_client = client is None
    c = client or httpx.Client(timeout=300)
    try:
        r = c.post(url, params={"key": require_api_key()}, json=body)
        r.raise_for_status()
        raw = r.json()
    finally:
        if own_client:
            c.close()
    cands = raw.get("candidates") or []
    parts = ((cands[0].get("content") or {}).get("parts")) if cands else []
    text = "\n".join(p.get("text", "") for p in (parts or []) if p.get("text")).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if 0 <= s < e:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                return None
    return None


# ── Worker ─────────────────────────────────────────────────────────────────
@dataclass
class ScoreRecord:
    qid: str
    parser: str
    source_file: str
    domain: str
    cached: bool
    ok: bool                  # answer is present (no error)
    match: bool               # all fields correct
    field_matches: dict
    error: str = ""


def _ensure_template(item: dict) -> None:
    if "template" not in item:
        item["template"] = build_template(item.get("response_format", ""),
                                          item.get("gold_dict") or {})
    if "str_keys" not in item:
        item["str_keys"] = string_keys(item["template"])


def _worker(item: dict, parser: str, layout: RunLayout, *,
            force: bool) -> ScoreRecord:
    qid = item["question_id"]
    sf = item["source_file"]
    cache = layout.cache_path(qid, parser)
    base = ScoreRecord(qid, parser, sf, item.get("domain", ""),
                       False, False, False, {})

    if cache.exists() and not force:
        try:
            rec = json.loads(cache.read_text())
        except Exception:
            rec = None
        if rec is not None:
            if "answer" in rec:
                fm, allc = score_typed(rec["answer"], item["gold_dict"],
                                       item["str_keys"])
                changed = (fm != rec.get("field_matches")
                           or allc != rec.get("match"))
                if changed:
                    rec["field_matches"], rec["match"] = fm, allc
                    cache.write_text(json.dumps(rec, ensure_ascii=False))
                return ScoreRecord(qid, parser, sf, item.get("domain", ""),
                                   True, True, allc, fm, rec.get("error", ""))
            if "error" in rec and "no answer" not in rec["error"]:
                # cached terminal error (e.g. "markdown missing") — re-attempt
                # only when the markdown later appears, never automatically
                return ScoreRecord(qid, parser, sf, item.get("domain", ""),
                                   True, False, False, {}, rec["error"])

    md_path = layout.parse_md(parser, sf)
    if not md_path.exists():
        rec = {"qid": qid, "parser": parser, "source_file": sf,
               "domain": item.get("domain", ""),
               "error": "markdown missing"}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rec, ensure_ascii=False))
        base.error = "markdown missing"
        return base

    md = md_path.read_text()
    for attempt in range(3):
        try:
            ans = gemini_extract(item["question"], item["template"], md)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                rec = {"qid": qid, "parser": parser, "source_file": sf,
                       "domain": item.get("domain", ""),
                       "error": str(e)[:200]}
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(rec, ensure_ascii=False))
                base.error = str(e)[:200]
                return base
            time.sleep(2 + attempt * 3)
    fm, allc = score_typed(ans, item["gold_dict"], item["str_keys"])
    rec = {"qid": qid, "parser": parser, "source_file": sf,
           "domain": item.get("domain", ""), "answer": ans,
           "field_matches": fm, "match": allc}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rec, ensure_ascii=False))
    return ScoreRecord(qid, parser, sf, item.get("domain", ""),
                       False, True, allc, fm)


# ── Public entry ───────────────────────────────────────────────────────────
def load_bank(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    items = data["items"] if isinstance(data, dict) else data
    for it in items:
        if "gold_dict" not in it:
            raise ValueError(
                f"bank item {it.get('question_id', '?')} missing gold_dict — "
                f"run jsonify on the bank first")
        _ensure_template(it)
    return items


def docs_for_items(items: list[dict]) -> list[str]:
    """Ordered unique `source_file` stems referenced by these bank items.

    All `--limit N` flows ultimately resolve to "the source_files referenced
    by the first N bank items" — this helper is the single source of truth
    so download, parse, and score stay aligned for smoke runs.
    """
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        sf = it.get("source_file")
        if sf and sf not in seen:
            seen.add(sf)
            out.append(sf)
    return out


def run_score(layout: RunLayout, parsers: list[str] | None = None, *,
              bank_path: Path | None = None, force: bool = False,
              workers: int = 16, limit: int | None = None,
              progress: bool = True) -> list[ScoreRecord]:
    """Score every (item × parser) pair against the bank. When `parsers` is
    falsy, scores every parser found under `<run-dir>/parses/`.

    Tests stub the model call by monkeypatching :func:`gemini_extract`
    (and setting ``GEMINI_API_KEY`` in the test env to satisfy the preflight).
    """
    layout.ensure_dirs()
    require_api_key()  # fail fast on missing GEMINI_API_KEY before spinning up the worker pool
    bank_path = bank_path or layout.bank_path
    if not bank_path.exists():
        raise FileNotFoundError(f"no bank at {bank_path}")
    items = load_bank(bank_path)
    if limit:
        items = items[:limit]

    parsers = list(parsers or layout.parsed_parsers())
    if not parsers:
        raise ValueError(
            f"no parsers given and no parses under {layout.parses_root} — "
            f"run `evaluate parse` first or pass -p")

    jobs = [(it, p) for it in items for p in parsers]
    if progress:
        print(f"{len(items)} questions × {len(parsers)} parsers = "
              f"{len(jobs)} (question, parser) cells", flush=True)

    records: list[ScoreRecord] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, it, p, layout,
                          force=force) for it, p in jobs]
        n_cached = 0
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            records.append(rec)
            if rec.cached:
                n_cached += 1
            if progress and (i % 500 == 0 or i == len(jobs)):
                print(f"  [{i}/{len(jobs)}]  cached={n_cached}  "
                      f"({(time.time()-t0)/60:.1f}min)", flush=True)

    if progress:
        n_ok = sum(1 for r in records if r.ok)
        n_match = sum(1 for r in records if r.match)
        print(f"score complete — {n_ok} answered, {n_match} fully correct, "
              f"{len(records) - n_ok} errors", flush=True)
    return records


def aggregate_results(layout: RunLayout) -> dict:
    """Flatten the cache dir into `eval/results.json` for downstream tools."""
    out: list[dict] = []
    for f in sorted(layout.cache_dir.glob("*__*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:  # noqa: BLE001
            continue
    layout.results_path.parent.mkdir(parents=True, exist_ok=True)
    layout.results_path.write_text(
        json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    return {"n_results": len(out), "path": str(layout.results_path)}


def summarize(records: list[ScoreRecord], parsers: list[str]) -> dict:
    """Per-parser per-field + per-question accuracy."""
    out = {}
    for p in parsers:
        rs = [r for r in records if r.parser == p and r.ok]
        f_ok = sum(sum(1 for v in r.field_matches.values() if v) for r in rs)
        f_tot = sum(len(r.field_matches) for r in rs)
        q_ok = sum(1 for r in rs if r.match)
        out[p] = {
            "n_results": len(rs),
            "field_pct": (100 * f_ok / f_tot) if f_tot else 0.0,
            "field_ok": f_ok, "field_tot": f_tot,
            "q_pct": (100 * q_ok / len(rs)) if rs else 0.0,
            "q_ok": q_ok, "q_tot": len(rs),
        }
    return out
