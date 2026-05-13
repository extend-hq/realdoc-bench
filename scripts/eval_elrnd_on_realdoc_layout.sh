#!/usr/bin/env bash
# Run the elrnd evaluator on the RealDoc-Layout 1500-pageId subset using
# the production ELRND checkpoint, with the original benchmark's 9-class
# collapse applied (Form merged into Key Value rather than into Text).
#
# Designed to be run on awsl4 (or any host with a CUDA GPU + the elrnd repo
# cloned at $ELRND_DIR and the v1.2-human-prepared dataset present).
#
# Inputs (env or args):
#   ELRND_DIR      default ~/elrnd
#   CHECKPOINT     default
#     $ELRND_DIR/checkpoints/chunkr_v1_2_kv_split_human_alignment/qvmdexq/checkpoint_step_00002100.pt
#   COCO_DIR       default $ELRND_DIR/data/v1.2-human-prepared/COCO
#   MANIFEST_CSV   default /tmp/realdoc-layout-manifest.csv (copy with
#                  `scp datasets/layout/manifest.csv awsl4:/tmp/realdoc-layout-manifest.csv`)
#   LOG            default /tmp/eval_realdoc_9class.log

set -e
ELRND_DIR="${ELRND_DIR:-$HOME/elrnd}"
CHECKPOINT="${CHECKPOINT:-$ELRND_DIR/checkpoints/chunkr_v1_2_kv_split_human_alignment/qvmdexq/checkpoint_step_00002100.pt}"
COCO_DIR="${COCO_DIR:-$ELRND_DIR/data/v1.2-human-prepared/COCO}"
MANIFEST_CSV="${MANIFEST_CSV:-/tmp/realdoc-layout-manifest.csv}"
LOG="${LOG:-/tmp/eval_realdoc_9class.log}"

# Step 1 — build the realdoc-layout COCO subset if it doesn't exist yet.
if [ ! -f "$COCO_DIR/realdoc_layout.json" ]; then
    python3 "$(dirname "$0")/build_realdoc_layout_coco.py" \
        "$(dirname "$COCO_DIR")" \
        "$MANIFEST_CSV" \
        "$COCO_DIR/realdoc_layout.json"
fi

# Step 2 — run evaluate.py with the 9-class remap.  Caption/Footnote/List
# Item/Line Number/Formula collapse into Text; Bar/QR Codes folds into the
# Picture/Figure/Image/Chart class (=Figure in the original report); Form
# folds into Key Value (this is the realdoc-layout twist on the original,
# which folded Key Value itself into Text).
cd "$ELRND_DIR"
source .venv/bin/activate
python scripts/evaluate.py \
    +experiment=chunkr_v1_2_kv_split_human_alignment \
    "data.train_sources.0.val_annotation=COCO/realdoc_layout.json" \
    "+eval.checkpoint=$CHECKPOINT" \
    +eval.split=val \
    +eval.batch_size=4 \
    "+eval.class_remap={Caption: Text, Line Number: Text, List Item: Text, Formula: Text, Bar/QR Codes: Picture/Figure/Image/Chart, Form: Key Value}" \
    2>&1 | tee "$LOG"
