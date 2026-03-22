"""Competition-aligned metrics (ISIC 2024 pAUC)."""

from __future__ import annotations

import numpy as np
import sklearn.metrics as sklearn_metrics
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_pauc(
    y_true,
    y_pred,
    min_tpr: float = 0.88,
    sample_weight=None,
) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(np.unique(y_true)) != 2:
        return float("nan")

    v_gt = np.abs(y_true.astype(float) - 1.0)
    v_pred = np.abs(y_pred - 1.0)
    max_fpr = abs(1.0 - min_tpr)

    fpr, tpr, _ = sklearn_metrics.roc_curve(v_gt, v_pred, sample_weight=sample_weight)

    if max_fpr is None or max_fpr >= 1:
        return float(sklearn_metrics.auc(fpr, tpr))
    if max_fpr <= 0:
        raise ValueError(f"Expected min_tpr in [0, 1), got min_tpr={min_tpr!r}")

    stop = np.searchsorted(fpr, max_fpr, side="right")
    if stop < 1:
        return 0.0
    if stop >= len(fpr):
        return float(sklearn_metrics.auc(fpr, tpr))

    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr_c = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr_c = np.append(fpr[:stop], max_fpr)
    return float(sklearn_metrics.auc(fpr_c, tpr_c))


def compute_all_metrics(y_true, y_pred, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)
    y_binary = (y_pred >= threshold).astype(int)
    pauc = compute_pauc(y_true, y_pred)
    return {
        "ROC-AUC": roc_auc_score(y_true, y_pred),
        "pAUC (≥88% TPR, ISIC 24)": pauc,
        "Average Precision": average_precision_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_binary, zero_division=0),
        "Recall (Sensitivity)": recall_score(y_true, y_binary, zero_division=0),
        "F1": f1_score(y_true, y_binary, zero_division=0),
    }
