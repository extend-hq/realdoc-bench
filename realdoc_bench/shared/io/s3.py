"""Optional S3 mirror loader.

Useful for engineers inside the VPC who want to bypass HF; HF remains the source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path

from realdoc_bench.shared.io.cache import default_cache_root, ensure_dir


def _client(profile: str | None, region: str | None):
    import boto3

    session_kwargs: dict[str, str] = {}
    if profile:
        session_kwargs["profile_name"] = profile
    elif env_profile := os.environ.get("REALDOC_BENCH_S3_PROFILE"):
        session_kwargs["profile_name"] = env_profile
    session = boto3.Session(**session_kwargs)
    return session.client("s3", region_name=region or os.environ.get("AWS_REGION"))


def fetch_prefix(
    bucket: str,
    prefix: str,
    *,
    cache_dir: Path | None = None,
    profile: str | None = None,
    region: str | None = None,
) -> Path:
    """Sync ``s3://bucket/prefix/`` into the local cache. Returns the local prefix root."""
    root = ensure_dir(cache_dir or (default_cache_root() / "s3" / bucket / prefix))
    client = _client(profile, region)
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            rel = key[len(prefix) :].lstrip("/")
            if not rel:
                continue
            local = root / rel
            if local.exists() and local.stat().st_size == obj.get("Size"):
                continue
            ensure_dir(local.parent)
            client.download_file(bucket, key, str(local))
    return root
