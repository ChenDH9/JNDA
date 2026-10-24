"""Deterministic semantic-segmentation metrics for S3 predeployment."""

from __future__ import annotations

import numpy as np


def _arrays(prediction, target):
    pred = np.asarray(prediction)
    truth = np.asarray(target)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target shapes differ")
    if pred.ndim < 1:
        raise ValueError("masks must be at least one-dimensional")
    return pred, truth


def confusion_matrix(
    prediction, target, num_classes: int, ignore_index: int | None = None
) -> np.ndarray:
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    pred, truth = _arrays(prediction, target)
    keep = np.ones(truth.shape, dtype=bool) if ignore_index is None else truth != ignore_index
    p = pred[keep].astype(np.int64, copy=False)
    t = truth[keep].astype(np.int64, copy=False)
    if np.any((p < 0) | (p >= num_classes)) or np.any((t < 0) | (t >= num_classes)):
        raise ValueError("class value outside configured range")
    return np.bincount(num_classes * t + p, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def _absent(values: np.ndarray, absent: np.ndarray, policy: str) -> np.ndarray:
    if policy not in {"nan", "zero", "one"}:
        raise ValueError("absent policy must be nan, zero, or one")
    result = values.astype(float, copy=True)
    result[absent] = {"nan": np.nan, "zero": 0.0, "one": 1.0}[policy]
    return result


def per_class_iou(confusion: np.ndarray, absent_policy: str = "nan") -> np.ndarray:
    cm = np.asarray(confusion, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("confusion matrix must be square")
    tp = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - tp
    return _absent(
        np.divide(tp, union, out=np.zeros_like(tp), where=union != 0), union == 0, absent_policy
    )


def mean_iou(confusion: np.ndarray, absent_policy: str = "nan") -> float:
    values = per_class_iou(confusion, absent_policy)
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")


def per_class_dice(confusion: np.ndarray, absent_policy: str = "nan") -> np.ndarray:
    cm = np.asarray(confusion, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("confusion matrix must be square")
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return _absent(
        np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom != 0), denom == 0, absent_policy
    )


def pixel_accuracy(confusion: np.ndarray) -> float:
    cm = np.asarray(confusion, dtype=np.int64)
    total = int(cm.sum())
    return float(np.trace(cm) / total) if total else float("nan")


def evaluate_segmentation(
    prediction,
    target,
    num_classes: int,
    ignore_index: int | None = None,
    absent_policy: str = "nan",
) -> dict:
    cm = confusion_matrix(prediction, target, num_classes, ignore_index)
    return {
        "confusion": cm.tolist(),
        "per_class_iou": per_class_iou(cm, absent_policy).tolist(),
        "miou": mean_iou(cm, absent_policy),
        "per_class_dice": per_class_dice(cm, absent_policy).tolist(),
        "pixel_accuracy": pixel_accuracy(cm),
        "ignore_index": ignore_index,
        "absent_policy": absent_policy,
    }
