#!/usr/bin/env python3
"""Parse every source PDF in v1.1/*/<type>/sources/ with Extend Performance v2.

Output:
  runs/extend_v2_parse/<timestamp>/
    manifest.csv         — per-PDF: path, pages, latency, cost, md_length, error
    <domain>/<type>/<filename>.md   — extracted markdown

Idempotent: skips PDFs whose .md already exists. Re-run to resume after
interruption.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
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


def discover_sources(v1_1_root: Path) -> list[tuple[str, str, Path]]:
    """Yield (domain, type, pdf_path) for every PDF under v1.1/*/<type>/sources/."""
    out = []
    for domain_dir in sorted(v1_1_root.iterdir()):
        if not domain_dir.is_dir():
            continue
        for type_dir in sorted(domain_dir.iterdir()):
            if not type_dir.is_dir():
                continue
            sources = type_dir / "sources"
            if not sources.exists():
                continue
            for pdf in sorted(sources.glob("*.pdf")):
                out.append((domain_dir.name, type_dir.name, pdf))
    return out


def parse_one(parser, pdf: Path) -> dict:
    t0 = time.time()
    try:
        result = parser.parse(pdf)
        return {
            "ok": True,
            "markdown": result.markdown or "",
            "page_count": result.page_count or 0,
            "latency_sec": result.latency_sec,
            "cost_usd": result.cost_estimate_usd or 0.0,
            "elapsed_total": time.time() - t0,
            "error": "",
        }
    except Exception as e:
        return {
            "ok": False, "markdown": "", "page_count": 0,
            "latency_sec": 0.0, "cost_usd": 0.0,
            "elapsed_total": time.time() - t0,
            "error": f"{type(e).__name__}: {str(e)[:300]}",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="Smoke: only N PDFs")
    ap.add_argument("--resume-dir", default=None,
                    help="Resume into an existing runs/extend_v2_parse/<ts>/ dir")
    ap.add_argument("--out-dir", default=None,
                    help="Explicit output dir (overrides timestamp-based default). "
                         "Created if needed; outputs go directly here.")
    ap.add_argument("--parser-name", default="extend_performance_v2_0_0",
                    help="Registered parser name in realdoc_bench")
    args = ap.parse_args()

    # Trigger parser registration via package import
    import realdoc_bench.extraction.parsers  # noqa: F401
    from realdoc_bench.extraction.parsers import extend  # noqa: F401  registers extend_*
    from realdoc_bench.extraction.parsers.base import registry as parser_registry
    parser_cls = parser_registry.get(args.parser_name)
    parser = parser_cls()

    pdfs = discover_sources(V1_1)
    if args.limit:
        pdfs = pdfs[:args.limit]
    print(f"Found {len(pdfs)} source PDFs in v1.1/")

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    elif args.resume_dir:
        out_dir = Path(args.resume_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO / "runs" / "extend_v2_parse" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir.relative_to(REPO)}/")

    manifest_path = out_dir / "manifest.csv"
    # Build skip-set from existing .md files
    existing: dict[tuple[str, str, str], Path] = {}
    for md in out_dir.rglob("*.md"):
        rel = md.relative_to(out_dir)
        parts = rel.parts
        if len(parts) >= 3:
            domain, typ, fname = parts[0], parts[1], parts[2]
            existing[(domain, typ, fname.replace(".md", ".pdf"))] = md

    todo = [(d, t, p) for d, t, p in pdfs if (d, t, p.name) not in existing]
    print(f"  skip {len(pdfs) - len(todo)} already done, todo {len(todo)} with {args.workers} workers")
    print(f"  parser: {args.parser_name}")

    # Open manifest append-mode
    fields = ["domain", "type", "filename", "page_count", "latency_sec",
              "cost_usd", "md_length", "elapsed_total", "ok", "error"]
    manifest_exists = manifest_path.exists()
    f_manifest = manifest_path.open("a", newline="")
    writer = csv.DictWriter(f_manifest, fieldnames=fields)
    if not manifest_exists:
        writer.writeheader()
        f_manifest.flush()

    t_start = time.time()
    n_ok = n_fail = 0
    total_cost = 0.0
    total_pages = 0

    def _job(domain: str, typ: str, pdf: Path):
        r = parse_one(parser, pdf)
        # Write markdown
        md_dir = out_dir / domain / typ
        md_dir.mkdir(parents=True, exist_ok=True)
        md_path = md_dir / (pdf.stem + ".md")
        if r["ok"]:
            md_path.write_text(r["markdown"])
        return (domain, typ, pdf, r, md_path)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_job, d, t, p) for d, t, p in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            domain, typ, pdf, r, md_path = fut.result()
            writer.writerow({
                "domain": domain, "type": typ, "filename": pdf.name,
                "page_count": r["page_count"], "latency_sec": f"{r['latency_sec']:.1f}",
                "cost_usd": f"{r['cost_usd']:.4f}", "md_length": len(r["markdown"]),
                "elapsed_total": f"{r['elapsed_total']:.1f}",
                "ok": "1" if r["ok"] else "0", "error": r["error"],
            })
            f_manifest.flush()
            if r["ok"]:
                n_ok += 1
                total_cost += r["cost_usd"] or 0
                total_pages += r["page_count"] or 0
                print(f"  [{i}/{len(todo)}] ✓ {domain}/{typ}/{pdf.name}  "
                      f"{r['page_count']}pp ${r['cost_usd']:.3f} {r['latency_sec']:.1f}s "
                      f"md={len(r['markdown'])}ch", flush=True)
            else:
                n_fail += 1
                print(f"  [{i}/{len(todo)}] ✗ {domain}/{typ}/{pdf.name}  {r['error']}", flush=True)

    f_manifest.close()
    wall = time.time() - t_start
    print(f"\nDone — {n_ok} ok / {n_fail} fail / {len(todo)} todo in {wall:.0f}s")
    print(f"  total pages parsed: {total_pages}, total cost: ${total_cost:.2f}")
    print(f"  manifest: {manifest_path.relative_to(REPO)}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
