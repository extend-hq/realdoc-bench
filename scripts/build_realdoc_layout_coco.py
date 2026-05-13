"""Build a COCO eval file containing only the realdoc-layout manifest pageIds.

Reads val.json + test.json from the v1.2-human-prepared dataset on awsl4,
filters images by file_name matching the manifest's 1500 pageIds, and writes
a unified COCO file at the requested output path.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: build_realdoc_layout_coco.py <data_root> <manifest_csv> <out_json>",
            file=sys.stderr,
        )
        return 2

    data_root = Path(sys.argv[1])
    manifest_csv = Path(sys.argv[2])
    out_json = Path(sys.argv[3])

    with manifest_csv.open() as f:
        manifest_rows = list(csv.DictReader(f))
    want = {f"human/{r['pageId']}.png" for r in manifest_rows}
    print(f"manifest: {len(want)} pageIds (file_name keys)", file=sys.stderr)

    merged_images: dict[int, dict] = {}
    merged_annotations: list[dict] = []
    categories = None
    next_image_id = 1
    next_ann_id = 1
    seen_file_names: set[str] = set()

    for split in ("val", "test"):
        path = data_root / "COCO" / f"{split}.json"
        with path.open() as f:
            data = json.load(f)
        if categories is None:
            categories = data["categories"]
        kept_image_ids: dict[int, int] = {}
        kept_in_split = 0
        for img in data["images"]:
            if img["file_name"] not in want:
                continue
            if img["file_name"] in seen_file_names:
                # appears in both val and test — keep first occurrence only
                continue
            new_id = next_image_id
            next_image_id += 1
            new_img = dict(img)
            new_img["id"] = new_id
            new_img["original_split"] = split
            merged_images[new_id] = new_img
            kept_image_ids[img["id"]] = new_id
            seen_file_names.add(img["file_name"])
            kept_in_split += 1
        for ann in data["annotations"]:
            if ann["image_id"] not in kept_image_ids:
                continue
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            next_ann_id += 1
            new_ann["image_id"] = kept_image_ids[ann["image_id"]]
            merged_annotations.append(new_ann)
        print(f"  {split}: kept {kept_in_split} images, running ann total {len(merged_annotations)}", file=sys.stderr)

    missing = sorted(want - seen_file_names)
    print(
        f"summary: {len(seen_file_names)}/{len(want)} manifest pageIds found in val+test; "
        f"{len(missing)} missing",
        file=sys.stderr,
    )
    if missing:
        print(f"  first 5 missing: {missing[:5]}", file=sys.stderr)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "info": {"description": "realdoc-layout subset of v1.2-human-prepared (val+test)"},
        "images": list(merged_images.values()),
        "annotations": merged_annotations,
        "categories": categories,
    }
    with out_json.open("w") as f:
        json.dump(out, f)
    print(f"wrote {out_json} ({out_json.stat().st_size:,} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
