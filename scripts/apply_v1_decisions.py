#!/usr/bin/env python3
"""Apply accept/reject decisions exported from the v1_review HTML.

Reads a JSON of shape:
    {
      "schema_version": 1,
      "exported_at": "...",
      "decisions": {
        "v1_med_autopsy": {
          "ewscripps_debose_autopsy.pdf": "accept",
          "lada_ramirez_ois_autopsy_court_filing.pdf": "reject",
          ...
        },
        ...
      }
    }

Actions:
  - For each "reject": delete the PDF from disk + remove the row from the manifest.
  - For each "accept": no-op (kept; useful as a positive paper trail).
  - "undecided" (i.e. absent from JSON) is left untouched.

Use --dry-run to preview without modifying anything.
Use --reset to clear the existing rejected-archive folder before running.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets" / "extraction"
ARCHIVE = REPO / "datasets" / "extraction" / "_rejected"
QUARANTINE_DIRS = [
    DATASETS / "_quarantine_confidentiality_stamp",
    DATASETS / "_quarantine_wrong_doctype",
]


def find_pdf_anywhere(doc_type: str, fname: str) -> Path | None:
    """Locate a PDF that may be in sources/ or any _quarantine_*/ folder."""
    primary = DATASETS / doc_type / "sources" / fname
    if primary.exists():
        return primary
    for q in QUARANTINE_DIRS:
        candidate = q / doc_type / fname
        if candidate.exists():
            return candidate
    return None


def load_decisions(path: Path) -> dict[str, dict[str, str]]:
    raw = json.loads(path.read_text())
    return raw.get("decisions", raw)


def _ensure_manifest_row(doc_type: str, fname: str) -> None:
    """If a restored file isn't in the manifest, append a stub row.
    The user can rerun write_v1_manifests.py to refresh richer metadata."""
    manifest = DATASETS / doc_type / "_sources_manifest.csv"
    if not manifest.exists():
        return
    import fitz
    pdf = DATASETS / doc_type / "sources" / fname
    pages = ""
    try:
        with fitz.open(pdf) as d:
            pages = str(d.page_count)
    except Exception:
        pass
    with manifest.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if any(r.get("local_filename") == fname for r in rows):
        return
    new_row = {fn: "" for fn in fieldnames}
    new_row.update({
        "local_filename": fname,
        "source_url": "",
        "source_site": "",
        "is_filled": "unknown",
        "doc_pages": pages,
        "downloaded_on": "2026-05-17",
        "notes": "[restored from quarantine via apply_v1_decisions.py — refresh provenance via write_v1_manifests.py]",
    })
    rows.append(new_row)
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


def trim_manifest(folder: Path, rejected_files: set[str], dry_run: bool) -> tuple[int, int]:
    """Drop manifest rows whose local_filename is in rejected_files.
    Returns (rows_kept, rows_dropped)."""
    manifest = folder / "_sources_manifest.csv"
    if not manifest.exists():
        return (0, 0)
    with manifest.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    kept = [r for r in rows if r["local_filename"] not in rejected_files]
    dropped = [r for r in rows if r["local_filename"] in rejected_files]
    if not dry_run and dropped:
        with manifest.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(kept)
    return (len(kept), len(dropped))


def archive_pdf(pdf: Path, doc_type: str, dry_run: bool) -> Path:
    archive_dir = ARCHIVE / doc_type
    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(pdf), str(archive_dir / pdf.name))
    return archive_dir / pdf.name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decisions_json", type=Path, help="path to v1_review_decisions.json")
    ap.add_argument("--dry-run", action="store_true", help="preview without changes")
    ap.add_argument("--reset", action="store_true", help="empty _rejected/ archive before running")
    ap.add_argument("--delete", action="store_true",
                    help="delete rejected PDFs outright instead of archiving to _rejected/")
    args = ap.parse_args()

    if not args.decisions_json.exists():
        print(f"error: {args.decisions_json} not found", file=sys.stderr)
        return 1

    decisions = load_decisions(args.decisions_json)
    if not decisions:
        print("no decisions in file")
        return 0

    if args.reset and not args.dry_run and ARCHIVE.exists():
        shutil.rmtree(ARCHIVE)
        print(f"[reset] cleared {ARCHIVE.relative_to(REPO)}")

    print(f"\nApplying decisions from {args.decisions_json.relative_to(REPO) if args.decisions_json.is_relative_to(REPO) else args.decisions_json}")
    if args.dry_run:
        print("(DRY RUN — no changes)\n")

    summary = {"accepted": 0, "rejected": 0, "missing": 0, "types_touched": 0}
    for doc_type, files in sorted(decisions.items()):
        folder = DATASETS / doc_type
        if not folder.exists():
            print(f"[skip] {doc_type}: folder not found")
            continue
        accepted = sorted(f for f, v in files.items() if v == "accept")
        rejected = sorted(f for f, v in files.items() if v == "reject")
        if not accepted and not rejected:
            continue
        summary["types_touched"] += 1
        summary["accepted"] += len(accepted)
        summary["rejected"] += len(rejected)

        print(f"\n=== {doc_type} ===")
        if accepted:
            print(f"  ✓ keep ({len(accepted)}):")
            for f in accepted:
                # Auto-restore if currently in quarantine
                primary = DATASETS / doc_type / "sources" / f
                if not primary.exists():
                    restored = False
                    for q in QUARANTINE_DIRS:
                        candidate = q / doc_type / f
                        if candidate.exists():
                            if not args.dry_run:
                                primary.parent.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(candidate), str(primary))
                            print(f"      {f}  ← RESTORED from {q.name}/{doc_type}/")
                            restored = True
                            # Add row back to manifest if missing
                            if not args.dry_run:
                                _ensure_manifest_row(doc_type, f)
                            break
                    if not restored:
                        summary["missing"] += 1
                        print(f"      {f}  [accepted but not found on disk or quarantine]")
                else:
                    print(f"      {f}")
        if rejected:
            print(f"  ✗ remove ({len(rejected)}):")
            for f in rejected:
                pdf = folder / "sources" / f
                if not pdf.exists():
                    summary["missing"] += 1
                    print(f"      {f}  [already missing]")
                    continue
                if args.delete:
                    if not args.dry_run:
                        pdf.unlink()
                    print(f"      {f}  → DELETED")
                else:
                    dest = archive_pdf(pdf, doc_type, args.dry_run)
                    print(f"      {f}  → archived to {dest.relative_to(REPO)}")

        # Trim manifest
        kept, dropped = trim_manifest(folder, set(rejected), args.dry_run)
        if dropped:
            print(f"  manifest: {dropped} rows dropped → {kept} kept")

    print(f"\n=== SUMMARY ===")
    print(f"  types touched : {summary['types_touched']}")
    print(f"  ✓ accepted    : {summary['accepted']}")
    print(f"  ✗ rejected    : {summary['rejected']}")
    print(f"  missing       : {summary['missing']}")
    if args.dry_run:
        print("\n(dry-run — rerun without --dry-run to apply)")
    elif summary["rejected"] and not args.delete:
        print(f"\nRejected PDFs archived under {ARCHIVE.relative_to(REPO)}/")
        print(f"To recover a PDF: mv {ARCHIVE.name}/<type>/<file>.pdf <type>/sources/")
        print(f"To delete archive permanently: rm -rf {ARCHIVE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
