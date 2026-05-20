#!/usr/bin/env python3
"""Recover per-page Extend chunks from past parse runs WITHOUT re-parsing.

Uses client.parse_runs.list(file_name_contains=...) + .retrieve(id) to pull
the chunks we already paid for. Each chunk corresponds to one page
(chunkingStrategy="page"), and each chunk has typed blocks (table, key_value,
heading, text, etc.).

For each v1.1 source PDF:
  1. List past runs matching file name
  2. Pick the most recent PROCESSED run with engineVersion='2.0.0'
  3. Retrieve the full run → save chunks JSON
  4. Compute per-page feature counts from block types + content regex

Output:
  runs/extend_chunks/<ts>/raw/<dom>/<typ>/<doc>.json       — full chunks per doc
  runs/extend_chunks/<ts>/per_page_features.csv            — per-page features
  runs/extend_chunks/<ts>/summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
V1_1 = REPO / "datasets" / "extraction" / "v1.1"


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


_client = None
def get_client():
    global _client
    if _client is None:
        from extend_ai import Extend
        token = os.environ["EXTEND_API_KEY"]
        url = "https://api.staging.extend.ai" if token.startswith("sk_test_") else "https://api.extend.ai"
        _client = Extend(token=token, base_url=url)
    return _client


def discover_docs(v1_1: Path) -> list[tuple[str, str, str, Path, int]]:
    out = []
    for domain_dir in sorted(d for d in v1_1.iterdir() if d.is_dir() and not d.name.startswith("_")):
        for type_dir in sorted(t for t in domain_dir.iterdir() if t.is_dir()):
            sources = type_dir / "sources"
            if not sources.exists():
                continue
            for pdf in sorted(sources.glob("*.pdf")):
                try:
                    with fitz.open(pdf) as d:
                        pc = d.page_count
                except Exception:
                    continue
                out.append((domain_dir.name, type_dir.name, pdf.name, pdf, pc))
    return out


MD_CHECKBOX_RE = re.compile(r"\[[ xX✓]\]")
UNICODE_CHECKBOX_RE = re.compile(r"[☐☑□■⬜⬛✓✗✔]")
SIGNATURE_RE = re.compile(r"signature|signed by|/s/", re.IGNORECASE)


def block_type_str(b) -> str:
    t = b.get("type")
    if hasattr(t, "value"):
        return t.value
    return str(t).lower() if t else ""


def features_from_chunk(chunk: dict) -> dict:
    """Per-page features from one Extend chunk."""
    blocks = chunk.get("blocks") or []
    content = chunk.get("content") or ""
    n_tables = 0
    n_kv = 0
    n_figures = 0
    n_headings = 0
    n_text_blocks = 0
    block_types_seen: dict = defaultdict(int)
    for b in blocks:
        t = block_type_str(b)
        block_types_seen[t] += 1
        if t == "table":
            n_tables += 1
        elif t == "key_value":
            n_kv += 1
        elif t == "figure":
            n_figures += 1
        elif t in ("heading", "section_heading"):
            n_headings += 1
        elif t == "text":
            n_text_blocks += 1
    # Checkboxes via regex on chunk content
    n_checkboxes = len(MD_CHECKBOX_RE.findall(content)) + len(UNICODE_CHECKBOX_RE.findall(content))
    has_signature = bool(SIGNATURE_RE.search(content))
    # Primary category — heuristic
    if n_checkboxes >= 3 and n_kv == 0:
        category = "checkboxes"
    elif n_tables >= 2:
        category = "tables"
    elif n_kv >= 5:
        category = "key_values"
    elif n_checkboxes >= 1 and n_kv >= 1:
        category = "form"
    elif n_tables >= 1 and n_kv >= 1:
        category = "mixed"
    elif n_tables >= 1:
        category = "tables"
    elif n_kv >= 1 or n_checkboxes >= 1:
        category = "form"
    elif n_text_blocks > 0 or n_headings > 0:
        category = "prose"
    else:
        category = "blank"
    # Richness score (1-10)
    structure = n_tables * 3 + n_checkboxes * 0.5 + n_kv * 0.4 + n_figures
    if structure >= 20: rich = 10
    elif structure >= 12: rich = 9
    elif structure >= 6: rich = 7
    elif structure >= 3: rich = 5
    elif structure >= 1: rich = 3
    elif n_text_blocks > 0: rich = 2
    else: rich = 1
    return {
        "n_tables": n_tables,
        "n_checkboxes": n_checkboxes,
        "n_kv": n_kv,
        "n_figures": n_figures,
        "n_headings": n_headings,
        "n_text_blocks": n_text_blocks,
        "n_blocks_total": len(blocks),
        "content_chars": len(content),
        "has_signature": has_signature,
        "primary_category": category,
        "richness_score": rich,
        "block_types": dict(block_types_seen),
    }


def fetch_doc(client, domain: str, type_name: str, fname: str,
              expected_engine_version: str, target_pc: int,
              raw_dir: Path) -> dict:
    """List + retrieve the best matching parse run for this file. Returns
    {'ok': bool, 'chunks': [...], 'run_id': ...} or error."""
    out_path = raw_dir / domain / type_name / fname.replace(".pdf", ".json")
    if out_path.exists():
        # cached
        try:
            data = json.loads(out_path.read_text())
            return {"ok": True, "chunks": data.get("chunks", []),
                    "run_id": data.get("run_id"), "cached": True}
        except Exception:
            pass

    try:
        # Step 1: list matching by filename
        # Some filenames are very long — only use first 60 chars to avoid issues
        search = fname[:60].replace(".pdf", "")
        resp = client.parse_runs.list(file_name_contains=search, max_page_size=20)
        candidates = [r for r in resp.data
                       if (getattr(r, "file", None) and
                           getattr(r.file, "name", "") == fname)]
        # Filter to PROCESSED with right engine version
        good = []
        for r in candidates:
            cfg = getattr(r, "config", None)
            ev = ""
            if cfg is not None:
                ev = (getattr(cfg, "engine_version", "") or
                      (cfg.get("engine_version", "") if isinstance(cfg, dict) else ""))
            status = str(getattr(r, "status", "")).upper()
            if status == "PROCESSED" and ev == expected_engine_version:
                good.append(r)
        if not good:
            # Looser fallback: accept any PROCESSED match
            good = [r for r in candidates if str(getattr(r, "status", "")).upper() == "PROCESSED"]
        if not good:
            return {"ok": False, "error": f"no matching parse runs (found {len(candidates)} for filename, none PROCESSED with engine_version={expected_engine_version})"}

        # Step 2: retrieve the most recent
        # Sort by created_at if available, else just take first
        good.sort(key=lambda r: str(getattr(r, "created_at", "")), reverse=True)
        best_id = good[0].id
        run = client.parse_runs.retrieve(best_id)
        d = run.model_dump()
        output = d.get("output") or {}
        chunks = output.get("chunks") or []
        if len(chunks) != target_pc:
            # Mismatch — warn but accept
            pass
        # Save raw chunks
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "run_id": best_id,
            "engine_version": d.get("config", {}).get("engine_version") if isinstance(d.get("config"), dict) else getattr(getattr(run, "config", None), "engine_version", ""),
            "expected_page_count": target_pc,
            "actual_chunk_count": len(chunks),
            "chunks": chunks,
        }, indent=2, default=str))
        return {"ok": True, "chunks": chunks, "run_id": best_id, "cached": False}

    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--engine-version", default="2.0.0")
    args = ap.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO / "runs" / "extend_chunks" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    docs = discover_docs(V1_1)
    print(f"{len(docs)} docs, {sum(d[4] for d in docs)} pages")

    client = get_client()

    page_rows: list[dict] = []
    doc_rows: list[dict] = []
    fail_rows: list[dict] = []
    n_cached = 0; n_fetched = 0; n_fail = 0

    def _job(domain, typ, fname, pdf, pc):
        return (domain, typ, fname, pc, fetch_doc(client, domain, typ, fname,
                                                    args.engine_version, pc, raw_dir))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_job, *d) for d in docs]
        for i, fut in enumerate(as_completed(futures), 1):
            domain, typ, fname, pc, r = fut.result()
            if not r["ok"]:
                n_fail += 1
                fail_rows.append({"domain": domain, "type": typ, "filename": fname,
                                  "expected_pc": pc, "error": r["error"]})
                print(f"  [{i}/{len(docs)}] ✗ {domain}/{typ}/{fname[:40]}: {r['error']}", flush=True)
                continue
            if r.get("cached"): n_cached += 1
            else: n_fetched += 1
            chunks = r["chunks"]
            doc_features: dict = defaultdict(int)
            for pi, chunk in enumerate(chunks):
                f = features_from_chunk(chunk)
                page_rows.append({
                    "domain": domain, "type": typ, "filename": fname,
                    "page_idx": pi, "page_count_total": pc,
                    "primary_category": f["primary_category"],
                    "richness_score": f["richness_score"],
                    "n_tables": f["n_tables"],
                    "n_checkboxes": f["n_checkboxes"],
                    "n_kv": f["n_kv"],
                    "n_figures": f["n_figures"],
                    "n_headings": f["n_headings"],
                    "n_text_blocks": f["n_text_blocks"],
                    "n_blocks_total": f["n_blocks_total"],
                    "content_chars": f["content_chars"],
                    "has_signature": f["has_signature"],
                })
                for k in ("n_tables", "n_checkboxes", "n_kv", "n_figures", "n_headings"):
                    doc_features[k] += f[k]
            doc_rows.append({
                "domain": domain, "type": typ, "filename": fname,
                "page_count": pc, "chunks_found": len(chunks),
                "run_id": r["run_id"], "cached": r.get("cached", False),
                **{f"total_{k}": v for k, v in doc_features.items()},
            })
            if i % 20 == 0 or i == len(docs):
                tag = "cached" if r.get("cached") else "fetched"
                print(f"  [{i}/{len(docs)}] ✓ {tag} {domain}/{typ}/{fname[:40]} ({len(chunks)} pages)", flush=True)

    wall = time.time() - t0
    print(f"\nFetch done in {wall:.0f}s — {n_fetched} fetched, {n_cached} cached, {n_fail} failed")

    # Write per-page features
    pp_path = out_dir / "per_page_features.csv"
    pp_fields = ["domain", "type", "filename", "page_idx", "page_count_total",
                  "primary_category", "richness_score",
                  "n_tables", "n_checkboxes", "n_kv", "n_figures", "n_headings",
                  "n_text_blocks", "n_blocks_total", "content_chars", "has_signature"]
    with pp_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pp_fields)
        w.writeheader()
        for r in sorted(page_rows, key=lambda x: (x["domain"], x["type"], x["filename"], x["page_idx"])):
            w.writerow(r)

    pd_path = out_dir / "per_doc_features.csv"
    pd_fields = ["domain", "type", "filename", "page_count", "chunks_found",
                  "run_id", "cached",
                  "total_n_tables", "total_n_checkboxes", "total_n_kv",
                  "total_n_figures", "total_n_headings"]
    with pd_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pd_fields)
        w.writeheader()
        for r in sorted(doc_rows, key=lambda x: (x["domain"], x["type"], x["filename"])):
            w.writerow(r)

    if fail_rows:
        fail_path = out_dir / "_failures.csv"
        with fail_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["domain", "type", "filename", "expected_pc", "error"])
            w.writeheader()
            for r in fail_rows: w.writerow(r)

    # Summary
    summary = out_dir / "summary.md"
    cat_pages = defaultdict(int)
    for r in page_rows:
        cat_pages[r["primary_category"]] += 1
    rich_pages = sum(1 for r in page_rows
                       if r["primary_category"] in {"tables", "checkboxes", "form", "key_values", "mixed"})
    with summary.open("w") as f:
        f.write(f"# Per-page features from Extend chunks — {out_dir.name}\n\n")
        f.write(f"**{len(doc_rows)} docs, {len(page_rows)} pages**  ·  "
                f"fetched {n_fetched} from Extend, {n_cached} cached locally, {n_fail} failed\n\n")
        f.write("## Per-page primary category\n\n")
        f.write("| Category | Pages |\n|---|---:|\n")
        for c, n in sorted(cat_pages.items(), key=lambda x: -x[1]):
            f.write(f"| `{c}` | {n:,} |\n")
        f.write(f"\n**Rich pages** (tables/checkboxes/form/key_values/mixed): "
                f"**{rich_pages:,} of {len(page_rows):,}** ({rich_pages/max(len(page_rows),1)*100:.0f}%)\n\n")

    print(f"\n=== SUMMARY ===")
    print(f"  rich pages: {rich_pages:,} / {len(page_rows):,} ({rich_pages/max(len(page_rows),1)*100:.0f}%)")
    for c, n in sorted(cat_pages.items(), key=lambda x: -x[1]):
        print(f"  {c:15} {n:>5}")
    print(f"\nWrote: {pp_path.relative_to(REPO)}, {pd_path.relative_to(REPO)}, {summary.relative_to(REPO)}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
