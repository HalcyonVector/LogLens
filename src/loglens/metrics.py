"""Session-level evaluation helpers shared by every model.

Thresholds are always chosen on the validation period and then applied
unchanged to the test period.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def best_f1_threshold(y_val: np.ndarray, s_val: np.ndarray) -> float:
    """Score threshold that maximises F1 on validation (flag when score >= threshold)."""
    precision, recall, thresholds = precision_recall_curve(y_val, s_val)
    # The last precision/recall pair has no threshold.
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[int(np.argmax(f1))])


def prf(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "flagged": tp + fp,
    }


def evaluate(y_val, s_val, y_test, s_test) -> dict:
    """Pick the threshold on validation, report everything on test."""
    thr = best_f1_threshold(y_val, s_val)
    out = prf(y_test, (s_test >= thr).astype(int))
    out["pr_auc"] = round(float(average_precision_score(y_test, s_test)), 6) if y_test.sum() else None
    out["threshold"] = thr
    out["val"] = prf(y_val, (s_val >= thr).astype(int))
    return out


def pr_curve_points(y: np.ndarray, s: np.ndarray, n: int = 60) -> list[dict]:
    """A PR curve thinned to about n points, small enough for the dashboard."""
    precision, recall, _ = precision_recall_curve(y, s)
    idx = np.unique(np.linspace(0, len(recall) - 1, n).round().astype(int))
    return [{"recall": round(float(recall[i]), 4), "precision": round(float(precision[i]), 4)} for i in idx]
