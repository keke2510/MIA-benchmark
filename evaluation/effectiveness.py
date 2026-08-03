from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _safe_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return value


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(tpr[valid[-1]])


def summarize_curve_metrics(fpr: np.ndarray, tpr: np.ndarray) -> dict[str, float]:
    auc = float(np.trapz(tpr, fpr))
    return {
        "auc": auc,
        "TPR@0.1%FPR": tpr_at_fpr(fpr, tpr, 0.001),
        "TPR@1%FPR": tpr_at_fpr(fpr, tpr, 0.01),
        "TPR@10%FPR": tpr_at_fpr(fpr, tpr, 0.10),
    }


def infer_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_idx = int(np.argmax(tpr - fpr))
    return float(thresholds[best_idx])


def summarize_binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float | None]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if threshold is None:
        threshold = infer_threshold(labels, scores)
    preds = (scores >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(labels, scores)
    metrics = summarize_curve_metrics(fpr=fpr, tpr=tpr)
    metrics.update(
        {
            "accuracy": _safe_float(accuracy_score(labels, preds)),
            "balanced_accuracy": _safe_float(balanced_accuracy_score(labels, preds)),
            "precision": _safe_float(precision_score(labels, preds, zero_division=0)),
            "recall": _safe_float(recall_score(labels, preds, zero_division=0)),
            "auc": _safe_float(roc_auc_score(labels, scores)),
            "threshold": _safe_float(threshold),
        }
    )
    return metrics
