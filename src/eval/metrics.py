# src/eval/metrics.py

from __future__ import annotations

from typing import Any, Dict, Tuple, Literal, Optional, List

import numpy as np
import pandas as pd

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


def _policy_key_from_fpr(fpr: float) -> str:
    """
    Make stable keys for FPR targets.
    Examples:
      0.001 -> fpr_0_1pct
      0.01  -> fpr_1pct
      0.05  -> fpr_5pct
    """
    pct = float(fpr) * 100.0

    # If it's basically an integer percent, keep it clean
    if abs(pct - round(pct)) < 1e-12:
        return f"fpr_{int(round(pct))}pct"

    # Otherwise keep one decimal (0.1%) and swap '.' -> '_'
    s = f"{pct:.1f}".rstrip("0").rstrip(".")
    s = s.replace(".", "_")
    return f"fpr_{s}pct"


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


def threshold_at_fpr(y_true: np.ndarray, p: np.ndarray, target_fpr: float) -> float:
    """
    Determines threshold t such that FPR <= target_fpr on the provided data.
    Uses the quantile of negative scores method.
    """
    y_true = _to_numpy_1d(y_true).astype(int)
    p = _safe_probs(_to_numpy_1d(p).astype(float))
    
    neg_scores = p[y_true == 0]
    if neg_scores.size == 0:
        return 1.000001
        
    # We want the threshold such that only target_fpr fraction of negatives are >= threshold.
    # This is the (1 - target_fpr) quantile.
    t = np.quantile(neg_scores, 1.0 - target_fpr)
    return float(t)


def pick_threshold_for_fpr(
    y_true: np.ndarray,
    p: np.ndarray,
    target_fpr: float,
    mode: Literal["max_recall_under_fpr", "most_conservative"] = "max_recall_under_fpr",
) -> float:
    """
    Wrapper for threshold_at_fpr to maintain compatibility.
    """
    return threshold_at_fpr(y_true, p, target_fpr)


def compute_policy_thresholds(
    y_true: np.ndarray, 
    p_calib: np.ndarray, 
    fpr_list: Tuple[float, ...]
) -> Dict[str, float]:
    """
    Computes thresholds for a list of target FPRs.
    """
    thresholds = {}
    for fpr in fpr_list:
        t = threshold_at_fpr(y_true, p_calib, fpr)
        key = _policy_key_from_fpr(fpr)
        thresholds[key] = t
    return thresholds


def select_policy_thresholds(
    df,
    *,
    label_col: str = "label",
    prob_col: str = "p",
    split_col: str = "split",
    split_name: str = "val",
    policy_fprs: Tuple[float, ...] = (0.01, 0.05),
    policy_mode: Literal["max_recall_under_fpr", "most_conservative"] = "max_recall_under_fpr",
) -> Dict[str, float]:
    """
    Selects thresholds based on a specific split (usually 'val').
    Returns a dict mapping policy key (e.g. 'fpr_1pct') to the chosen threshold.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    subset = df[df[split_col] == split_name]
    if len(subset) == 0:
        raise ValueError(f"Split '{split_name}' not found or empty in dataframe.")

    y = subset[label_col].to_numpy()
    p = subset[prob_col].to_numpy()

    return compute_policy_thresholds(y, p, policy_fprs)


def binary_metrics(
    y_true,
    p,
    *,
    threshold: float = 0.5,
    compute_policy_thresholds: bool = True,
    policy_fprs: Tuple[float, ...] = (0.01, 0.05),
    policy_mode: Literal["max_recall_under_fpr", "most_conservative"] = "max_recall_under_fpr",
    fixed_policy_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Main one-stop metric function for binary classification.
    Returns ROC-AUC, PR-AUC, logloss, brier, and threshold-based metrics.

    If fixed_policy_thresholds is provided (dict of key->thr), we use those thresholds
    instead of computing new ones from the data.
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
                "firewall_policy": confusion_at_threshold(y_true, p, threshold),
                "note": "Only one class present in y_true; ROC/PR AUC undefined.",
            }
        )
        # Even if one class is missing, we can still compute confusion metrics for fixed thresholds
        if fixed_policy_thresholds:
            pol: Dict[str, Any] = {}
            for key, thr in fixed_policy_thresholds.items():
                rec = confusion_at_threshold(y_true, p, thr)
                rec["fixed_threshold_used"] = True
                pol[key] = rec
            out["policy_thresholds"] = pol
        return out

    out["roc_auc"] = float(roc_auc_score(y_true, p))
    out["pr_auc"] = float(average_precision_score(y_true, p))
    out["logloss"] = float(log_loss(y_true, p, labels=[0, 1]))
    out["brier"] = float(brier_score_loss(y_true, p))

    # CHANGED: Use "firewall_policy" key instead of "threshold_0.5"
    out["firewall_policy"] = confusion_at_threshold(y_true, p, threshold)

    if fixed_policy_thresholds:
        # Use pre-computed thresholds (e.g. from val split)
        pol: Dict[str, Any] = {}
        for key, thr in fixed_policy_thresholds.items():
            rec = confusion_at_threshold(y_true, p, thr)
            rec["fixed_threshold_used"] = True
            pol[key] = rec
        out["policy_thresholds"] = pol

    elif compute_policy_thresholds:
        # Compute thresholds on THIS data (potentially optimistic if this is test set)
        pol: Dict[str, Any] = {}
        for f in policy_fprs:
            thr = threshold_at_fpr(y_true, p, target_fpr=float(f))
            rec = confusion_at_threshold(y_true, p, thr)
            rec["fpr_target"] = float(f)
            pol[_policy_key_from_fpr(float(f))] = rec
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
    policy_mode: Literal["max_recall_under_fpr", "most_conservative"] = "max_recall_under_fpr",
    fixed_policy_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute metrics for each split in a single dataframe.
    Expects columns: split, label, prob.
    """
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
            y,
            p,
            threshold=threshold,
            policy_fprs=policy_fprs,
            policy_mode=policy_mode,
            fixed_policy_thresholds=fixed_policy_thresholds,
        )

    return out
