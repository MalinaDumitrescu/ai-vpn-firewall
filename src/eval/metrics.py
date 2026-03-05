# src/eval/metrics.py

from __future__ import annotations

from typing import Any, Dict, Tuple, Literal, Optional

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


def pick_threshold_for_fpr(
    y_true: np.ndarray,
    p: np.ndarray,
    target_fpr: float,
    mode: Literal["max_recall_under_fpr", "most_conservative"] = "max_recall_under_fpr",
) -> float:
    """
    Choose a threshold that achieves fpr <= target_fpr.

    Modes:
      - "max_recall_under_fpr" (default): Pick the LOWEST threshold that keeps FPR <= target.
        This maximizes Recall (TPR) while respecting the FPR constraint.
      - "most_conservative": Pick the HIGHEST threshold that keeps FPR <= target.
        This minimizes False Positives as much as possible, potentially sacrificing Recall.

    If no threshold achieves FPR <= target (e.g. even max threshold has high FPR?),
    returns a very high threshold (1.0 or slightly above max score) to force 0 predictions.
    """
    y_true = _to_numpy_1d(y_true).astype(int)
    p = _safe_probs(_to_numpy_1d(p).astype(float))

    fpr, _tpr, thr = roc_curve(y_true, p)

    # roc_curve returns an inf threshold at start; filter to finite
    # However, we might need that 'inf' if we want to predict all-zeros.
    # Let's handle finite thresholds for selection.
    finite_mask = np.isfinite(thr)
    fpr_finite = fpr[finite_mask]
    thr_finite = thr[finite_mask]

    if thr_finite.size == 0:
        # No finite thresholds? Just return something > 1.0
        return 1.000001

    # Find indices where FPR constraint is met
    ok_indices = np.where(fpr_finite <= float(target_fpr))[0]

    if ok_indices.size == 0:
        # Impossible to satisfy FPR constraint with any finite threshold?
        # Return a threshold > max(p) to predict all 0s (FPR=0).
        return float(np.max(p) + 1e-6)

    valid_thresholds = thr_finite[ok_indices]

    if mode == "max_recall_under_fpr":
        # Lowest threshold => Highest Recall
        return float(np.min(valid_thresholds))
    elif mode == "most_conservative":
        # Highest threshold => Lowest FPR (safest)
        return float(np.max(valid_thresholds))
    else:
        raise ValueError(f"Unknown mode: {mode}")


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
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    subset = df[df[split_col] == split_name]
    if len(subset) == 0:
        raise ValueError(f"Split '{split_name}' not found or empty in dataframe.")

    y = subset[label_col].to_numpy()
    p = subset[prob_col].to_numpy()

    thresholds = {}
    for f in policy_fprs:
        thr = pick_threshold_for_fpr(y, p, target_fpr=float(f), mode=policy_mode)
        key = _policy_key_from_fpr(float(f))
        thresholds[key] = thr

    return thresholds


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
            thr = pick_threshold_for_fpr(y_true, p, target_fpr=float(f), mode=policy_mode)
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
            y,
            p,
            threshold=threshold,
            policy_fprs=policy_fprs,
            policy_mode=policy_mode,
            fixed_policy_thresholds=fixed_policy_thresholds,
        )

    return out
