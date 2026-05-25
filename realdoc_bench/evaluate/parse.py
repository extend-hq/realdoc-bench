"""Parse stage — run each parser over every document in `<run-dir>/docs/`.

Cached by `(parser, doc_stem)`: a `(parser, doc)` is skipped when its `.md` and
`.json` both exist and the json's `ok=true`. `--force` busts the cache for the
listed parsers; `--retry-errors` retries only the previously-errored entries.

Per-parser scope: this only ever writes under `parses/<parser>/` for the
parsers the caller asked for. Other parsers' artifacts are never touched.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from realdoc_bench.evaluate.runs import RunLayout
from realdoc_bench.evaluate.parsers.base import ParseProvider, registry
from realdoc_bench.evaluate.score import docs_for_items, load_bank


@dataclass
class ParseRecord:
    parser: str
    stem: str
    ok: bool
    cached: bool
    page_count: int = 0
    latency_sec: float = 0.0
    cost_usd: float = 0.0
    md_length: int = 0
    error: str = ""


def _parse_one(parser_name: str, parser: ParseProvider, pdf: Path,
               layout: RunLayout) -> ParseRecord:
    stem = pdf.stem
    md_path = layout.parse_md(parser_name, stem)
    meta_path = layout.parse_meta(parser_name, stem)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        result = parser.parse(pdf)
        md = result.markdown or ""
        md_path.write_text(md)
        meta = {
            "parser": parser_name, "stem": stem, "ok": True,
            "page_count": result.page_count or 0,
            "latency_sec": result.latency_sec,
            "cost_usd": result.cost_estimate_usd or 0.0,
            "md_length": len(md),
            "elapsed_total": time.time() - t0,
            "error": "",
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        return ParseRecord(parser_name, stem, True, False, meta["page_count"],
                           meta["latency_sec"], meta["cost_usd"], len(md))
    except Exception as e:  # noqa: BLE001 — any parser failure → recorded, not raised
        err = f"{type(e).__name__}: {str(e)[:300]}"
        meta = {
            "parser": parser_name, "stem": stem, "ok": False,
            "page_count": 0, "latency_sec": 0.0, "cost_usd": 0.0,
            "md_length": 0, "elapsed_total": time.time() - t0,
            "error": err,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        return ParseRecord(parser_name, stem, False, False, error=err)


def _is_done(layout: RunLayout, parser: str, stem: str,
             force: bool, retry_errors: bool) -> bool:
    if force:
        return False
    md = layout.parse_md(parser, stem)
    meta = layout.parse_meta(parser, stem)
    if not (md.exists() and meta.exists()):
        return False
    try:
        ok = json.loads(meta.read_text()).get("ok", False)
    except Exception:
        return False
    if not ok and retry_errors:
        return False
    return ok or not retry_errors


def _cached_record(layout: RunLayout, parser: str, stem: str) -> ParseRecord:
    """Synthesize a record from existing cache files so the summary still
    reflects skipped work."""
    try:
        meta = json.loads(layout.parse_meta(parser, stem).read_text())
    except Exception:
        return ParseRecord(parser, stem, True, True)
    return ParseRecord(
        parser, stem, meta.get("ok", False), True,
        meta.get("page_count", 0), meta.get("latency_sec", 0.0),
        meta.get("cost_usd", 0.0), meta.get("md_length", 0),
        meta.get("error", ""),
    )


def run_parse(layout: RunLayout, parsers: list[str], *,
              bank_path: Path | None = None,
              force: bool = False, retry_errors: bool = False,
              workers: int = 8, limit: int | None = None,
              progress: bool = True) -> list[ParseRecord]:
    """Parse every PDF under `<run-dir>/docs/` with each parser.

    When `limit` is set, the scope is restricted to the source_files referenced
    by the first `limit` items of the bank (same selection rule as `score`
    and `download`, keeping all three flows consistent). The bank is loaded
    from `bank_path` or `layout.bank_path` if unset.
    """
    layout.ensure_dirs()
    pdfs = sorted(layout.docs_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDFs under {layout.docs_dir}")
    if limit:
        bank_path = bank_path or layout.bank_path
        if not bank_path.exists():
            raise FileNotFoundError(
                f"--limit needs the bank to pick which docs to parse, but "
                f"no bank at {bank_path}")
        items = load_bank(bank_path)
        wanted = set(docs_for_items(items[:limit]))
        pdfs = [p for p in pdfs if p.stem in wanted]
        if not pdfs:
            raise FileNotFoundError(
                f"--limit {limit} → {len(wanted)} source_files, but none of "
                f"them are present in {layout.docs_dir} as *.pdf")

    unknown = [n for n in parsers if n not in registry]
    if unknown:
        raise ValueError(
            f"unknown parser name(s): {unknown}. Available: {registry.names()}"
        )
    instances: dict[str, ParseProvider] = {n: registry.get(n)() for n in parsers}

    all_records: list[ParseRecord] = []
    t0 = time.time()

    for pname, pinst in instances.items():
        todo: list[Path] = []
        for pdf in pdfs:
            if _is_done(layout, pname, pdf.stem, force, retry_errors):
                all_records.append(_cached_record(layout, pname, pdf.stem))
            else:
                todo.append(pdf)
        if progress:
            print(f"=== {pname}: {len(todo)} to parse "
                  f"({len(pdfs) - len(todo)} cached) ===", flush=True)
        if not todo:
            continue
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_parse_one, pname, pinst, pdf, layout): pdf
                    for pdf in todo}
            ok = fail = 0
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                all_records.append(rec)
                ok, fail = ok + int(rec.ok), fail + int(not rec.ok)
                if progress and (i % 25 == 0 or i == len(todo)):
                    print(f"  {pname}: {i}/{len(todo)}  ok={ok} fail={fail}  "
                          f"({(time.time()-t0)/60:.1f}min)", flush=True)

    if progress:
        wall = (time.time() - t0) / 60
        n_ok = sum(1 for r in all_records if r.ok)
        n_fail = sum(1 for r in all_records if not r.ok)
        print(f"\nparse complete in {wall:.1f}min — {n_ok} ok, {n_fail} fail",
              flush=True)
    return all_records
