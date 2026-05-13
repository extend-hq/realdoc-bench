from __future__ import annotations

from realdoc_bench.layout.metrics import ap, structural_errors
from realdoc_bench.layout.metrics.f1 import evaluate, per_class
from realdoc_bench.layout.normalizers.base import BBox, LayoutBlock


def _b(x, y, w, h, t="text", conf=1.0):
    return LayoutBlock(block_type=t, bbox=BBox(x=x, y=y, w=w, h=h), confidence=conf)


def test_evaluate_perfect_match():
    preds = [_b(0, 0, 100, 100), _b(0, 200, 100, 100, t="table")]
    gts = [_b(0, 0, 100, 100), _b(0, 200, 100, 100, t="table")]
    r = evaluate(preds, gts)
    assert r.f1 == 1.0
    assert r.tp == 2 and r.fp == 0 and r.fn == 0


def test_evaluate_iou_below_threshold_is_fp_fn():
    preds = [_b(0, 0, 100, 100)]
    gts = [_b(80, 80, 100, 100)]  # IoU < 0.5
    r = evaluate(preds, gts)
    assert r.tp == 0 and r.fp == 1 and r.fn == 1


def test_misclassification_in_strict_eval_is_fp_plus_fn():
    preds = [_b(0, 0, 100, 100, t="table")]
    gts = [_b(0, 0, 100, 100, t="text")]
    r = evaluate(preds, gts, type_sensitive=True)
    # High IoU pair matched by matcher but rejected by strict type check.
    assert r.tp == 0 and r.fp == 1 and r.fn == 1


def test_per_class_breakdown():
    preds = [
        _b(0, 0, 100, 100, t="text"),
        _b(0, 200, 100, 100, t="table"),
    ]
    gts = [
        _b(0, 0, 100, 100, t="text"),
        _b(0, 200, 100, 100, t="text"),  # misclass: pred is table, gt is text
    ]
    out = per_class(preds, gts)
    assert out["text"].tp == 1
    # The "table" class has 1 pred and 0 gts → 1 FP.
    assert out["table"].fp == 1
    assert out["table"].fn == 0


def test_structural_errors_counts_misclassification():
    preds = [_b(0, 0, 100, 100, t="table")]
    gts = [_b(0, 0, 100, 100, t="text")]
    e = structural_errors.analyze(preds, gts)
    assert e.misclassifications == 1
    assert e.confusion[("table", "text")] == 1


def test_map_perfect_predictions():
    preds_by_doc = {"d": [_b(0, 0, 100, 100, conf=0.99)]}
    gts_by_doc = {"d": [_b(0, 0, 100, 100)]}
    per_cls, mean = ap.compute_map(preds_by_doc, gts_by_doc, iou_thresholds=[0.5, 0.75])
    assert per_cls["text"].ap50 == 1.0


def test_map_ignores_classes_without_gt():
    preds_by_doc = {"d": [_b(0, 0, 100, 100, t="figure", conf=0.99)]}
    gts_by_doc = {"d": [_b(0, 0, 100, 100, t="text")]}
    # Figure has predictions but no GT → contributes 0 to its bucket, mean drops.
    _, mean = ap.compute_map(preds_by_doc, gts_by_doc, iou_thresholds=[0.5])
    assert mean.ap50 == 0.0
