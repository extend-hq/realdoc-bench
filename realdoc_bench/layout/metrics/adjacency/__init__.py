"""Layout-detection metric with adjacency split/merge recovery.

Hungarian 1:1 matching with a +0.25 type-disagreement cost penalty,
``IoU ≥ 0.5`` gate, and an adjacency-merge pass that lets the matcher
recover from over-/under-segmentation. Reports both a *strict* and an
*adjusted* F1 per cell. No confidence sweep — every prediction is
counted positive at the IoU gate.

Layout:
- ``layout.match_blocks_with_adjacency`` — the per-cell matcher + adjacency
  recovery. Strict + adjusted PRF in one pass.
- ``aggregate.score_pairs`` — per-class / micro / macro roll-up over an
  iterable of per-page ``(preds, gts, page_w, page_h)`` tuples.
- ``ignore.apply_ignore_filter`` — optional configurable filter for block
  types you want excluded from a particular study. Default ignore set is
  empty under the public 9-class vocabulary.
- ``rescore`` — module entrypoint to re-score an existing ``runs/`` tree
  without re-running the processors.

``geometry.polygon_iou`` (Shapely) is intentionally omitted — this metric
operates on axis-aligned boxes, and dropping it avoids a Shapely dep.
"""

from realdoc_bench.layout.metrics.adjacency.aggregate import (
    AggregateResult,
    ClassScore,
    MetricBreakdown,
    score_pairs,
)
from realdoc_bench.layout.metrics.adjacency.ignore import (
    IGNORABLE_BLOCK_TYPES,
    apply_ignore_filter,
)
from realdoc_bench.layout.metrics.adjacency.layout import (
    LayoutEvaluationResult,
    MergeRecord,
    evaluate_layout_detection,
    get_adjusted_blocks,
    match_blocks,
    match_blocks_with_adjacency,
    precision_recall_f1,
)

__all__ = [
    "IGNORABLE_BLOCK_TYPES",
    "AggregateResult",
    "ClassScore",
    "LayoutEvaluationResult",
    "MergeRecord",
    "MetricBreakdown",
    "apply_ignore_filter",
    "evaluate_layout_detection",
    "get_adjusted_blocks",
    "match_blocks",
    "match_blocks_with_adjacency",
    "precision_recall_f1",
    "score_pairs",
]
