"""Fetch the realdoc-layout dataset from the ocr-bench S3 cache.

Pulls only the pageIds present in ``datasets/layout/manifest.csv`` (1,500 rows)
rather than the full 48k chunkr_human cache, into a snapshot dir matching
the layout loader's expected structure (``images/<id>.png`` +
``annotations/<id>.json``).

Defaults assume the ocr-bench staging cache:

    s3://ocr-bench-datasets/datasets/chunkr_human/{images,ground_truth}/

Output dir defaults to ``~/.cache/realdoc-bench/local-snapshots/realdoc-layout``,
matching the convention the loader's REALDOC_BENCH_DATASET_* env-var uses.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_BUCKET = "ocr-bench-datasets"
DEFAULT_PREFIX = "datasets/chunkr_human"
DEFAULT_REGION = "us-east-2"
DEFAULT_PROFILE = "staging-admin"
DEFAULT_OUT = Path.home() / ".cache" / "realdoc-bench" / "local-snapshots" / "realdoc-layout"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "datasets" / "layout" / "manifest.csv"


def _s3_client(profile: str, region: str):
    import boto3

    session = boto3.Session(profile_name=profile)
    return session.client("s3", region_name=region)


def _download_one(
    client,
    bucket: str,
    key: str,
    dst: Path,
    skip_existing: bool,
) -> tuple[str, str]:
    if skip_existing and dst.exists() and dst.stat().st_size > 0:
        return ("skip", str(dst))
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(bucket, key, str(dst))
        return ("ok", str(dst))
    except Exception as e:  # pragma: no cover - best-effort
        return ("err", f"{key}: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE") or DEFAULT_PROFILE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Only fetch first N pageIds")
    parser.add_argument("--image-ext", default="png", help="Image extension under images/")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    images_dir = args.out / "images"
    anno_dir = args.out / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    anno_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.open() as f:
        rows = list(csv.DictReader(f))
    page_ids = [r["pageId"] for r in rows]
    if args.limit:
        page_ids = page_ids[: args.limit]

    print(f"manifest: {len(page_ids)} pageIds", file=sys.stderr)
    print(f"source:   s3://{args.bucket}/{args.prefix}/", file=sys.stderr)
    print(f"out:      {args.out}", file=sys.stderr)
    print(f"profile:  {args.profile} ({args.region})", file=sys.stderr)

    client = _s3_client(args.profile, args.region)
    skip = not args.no_skip_existing

    tasks: list[tuple[str, Path]] = []
    for pid in page_ids:
        img_key = f"{args.prefix}/images/{pid}.{args.image_ext}"
        img_dst = images_dir / f"{pid}.{args.image_ext}"
        ann_key = f"{args.prefix}/ground_truth/{pid}.json"
        ann_dst = anno_dir / f"{pid}.json"
        tasks.append((img_key, img_dst))
        tasks.append((ann_key, ann_dst))

    ok = skipped = err = 0
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download_one, client, args.bucket, key, dst, skip): (key, dst)
            for key, dst in tasks
        }
        for i, fut in enumerate(as_completed(futures), 1):
            status, msg = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                err += 1
                if len(errors) < 10:
                    errors.append(msg)
            if i % 200 == 0:
                print(f"  ... {i}/{len(tasks)} (ok={ok}, skip={skipped}, err={err})", file=sys.stderr)

    print(f"\nDone. ok={ok}, skipped={skipped}, err={err}", file=sys.stderr)
    if errors:
        print("First errors:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
