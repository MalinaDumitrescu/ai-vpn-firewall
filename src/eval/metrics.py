# src/eval/metrics.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


_EPS = 1e-12


def _to_numpy_1d(x) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 1:
        a = a.reshape(-1)
    return a


def _safe_probs(p: np.ndarray) -> np.ndarray:
    # clamp to avoid inf in logloss, etc.
    return np.clip(p, _EPS, 1.0 - _EPS)


def confusion_at_threshold(y_true: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, Any]:
    y_true = _to_numpy_1d(y_true).astype(int)
    p = _to_numpy_1d(p).astype(float)
    y_hat = (p >= float(thr)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_hat, average="binary", zero_division=0
    )

    fpr = fp / max(fp + tn, 1)
    tpr = tp / max(tp + fn, 1)

    return {
        "threshold": float(thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def pick_threshold_for_fpr(y_true: np.ndarray, p: np.ndarray, target_fpr: float) -> float:
    """
    Choose the highest threshold that achieves fpr <= target_fpr.
    If no threshold achieves it, returns the max threshold (most conservative).
    """
    y_true = _to_numpy_1d(y_true).astype(int)
    p = _safe_probs(_to_numpy_1d(p).astype(float))

    fpr, tpr, thr = roc_curve(y_true, p)
    # roc_curve returns thr sorted descending (usually), with an extra inf threshold at start.
    # We'll scan all finite thresholds.
    finite = np.isfinite(thr)
    fpr = fpr[finite]
    thr = thr[finite]

    if thr.size == 0:
        return 1.0

    ok = np.where(fpr <= float(target_fpr))[0]
    if ok.size == 0:
        return float(np.max(thr))

    # pick the largest threshold among ok => lowest FPR, most conservative
    return float(np.max(thr[ok]))


def binary_metrics(
    y_true,
    p,
    *,
    threshold: float = 0.5,
    compute_policy_thresholds: bool = True,
    policy_fprs: Tuple[float, ...] = (0.01, 0.05),
) -> Dict[str, Any]:
    """
    Main one-stop metric function for binary classification.
    Returns ROC-AUC, PR-AUC, logloss, brier, and threshold-based metrics.
    """
    y_true = _to_numpy_1d(y_true).astype(int)
    p = _safe_probs(_to_numpy_1d(p).astype(float))

    n = int(y_true.size)
    pos = int(y_true.sum())
    neg = int(n - pos)

    out: Dict[str, Any] = {"n": n, "pos": pos, "neg": neg}

    # Some metrics are undefined if only one class appears
    if pos == 0 or neg == 0:
        out.update(
            {
                "roc_auc": None,
                "pr_auc": None,
                "logloss": float(log_loss(y_true, p, labels=[0, 1])),
                "brier": float(brier_score_loss(y_true, p)),
                "threshold_0.5": confusion_at_threshold(y_true, p, threshold),
                "note": "Only one class present in y_true; ROC/PR AUC undefined.",
            }
        )
        return out

    out["roc_auc"] = float(roc_auc_score(y_true, p))
    out["pr_auc"] = float(average_precision_score(y_true, p))
    out["logloss"] = float(log_loss(y_true, p, labels=[0, 1]))
    out["brier"] = float(brier_score_loss(y_true, p))

    out["threshold_0.5"] = confusion_at_threshold(y_true, p, threshold)

    if compute_policy_thresholds:
        pol: Dict[str, Any] = {}
        for f in policy_fprs:
            thr = pick_threshold_for_fpr(y_true, p, target_fpr=float(f))
            rec = confusion_at_threshold(y_true, p, thr)
            rec["fpr_target"] = float(f)
            pol[f"fpr_{int(round(100*f))}pct"] = rec
        out["policy_thresholds"] = pol

    return out


def group_metrics_by_split(
    df,
    *,
    label_col: str = "label",
    prob_col: str = "p",
    split_col: str = "split",
    threshold: float = 0.5,
    policy_fprs: Tuple[float, ...] = (0.01, 0.05),
) -> Dict[str, Any]:
    """
    Compute metrics for each split in a single dataframe.
    Expects columns: split, label, prob.
    """
    import pandas as pd  # local import to keep this file lightweight

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    needed = {label_col, prob_col, split_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out: Dict[str, Any] = {"splits": {}}
    for split_name, g in df.groupby(split_col, sort=True):
        y = g[label_col].to_numpy()
        p = g[prob_col].to_numpy()
        out["splits"][str(split_name)] = binary_metrics(
            y, p, threshold=threshold, policy_fprs=policy_fprs
        )

    return out
