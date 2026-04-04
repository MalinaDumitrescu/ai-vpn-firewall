# src/eval/bootstrap.py
"""
Bootstrap confidence interval utilities for session-level VPN detection metrics.

All bootstrapping is done at the SESSION level (not flow level) to respect
the hierarchical structure of the data. This avoids inflated sample sizes
and gives honest confidence intervals.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from src.eval.metrics import threshold_at_fpr, confusion_at_threshold


def _p90_agg(x):
    return float(np.percentile(x, 90))


def _wt5_agg(x):
    s = np.sort(x)[::-1][:5]
    if len(s) == 0:
        return 0.0
    w = np.array([0.40, 0.25, 0.15, 0.10, 0.10])[:len(s)]
    w = w / w.sum()
    return float(np.sum(s * w))


def _wt7_agg(x):
    s = np.sort(x)[::-1][:7]
    if len(s) == 0:
        return 0.0
    w = np.array([0.30, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05])[:len(s)]
    w = w / w.sum()
    return float(np.sum(s * w))


def _trimmed_mean_agg(x):
    if len(x) < 5:
        return float(np.mean(x)) if len(x) > 0 else 0.0
    s = np.sort(x)
    n = len(s)
    lo = max(1, int(n * 0.1))
    hi = n - lo
    return float(np.mean(s[lo:hi]))


AGG_FUNCTIONS = {
    "p90": _p90_agg,
    "wt5": _wt5_agg,
    "wt7": _wt7_agg,
    "p80": lambda x: float(np.percentile(x, 80)),
    "p85": lambda x: float(np.percentile(x, 85)),
    "median": lambda x: float(np.median(x)),
    "trimmed_mean": _trimmed_mean_agg,
}


def _aggregate_to_sessions(
    df: pd.DataFrame,
    prob_col: str,
    agg_fn: Callable,
    session_col: str = "capture_id",
    label_col: str = "label",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate flows to sessions, return (session_ids, labels, scores)."""
    sess_labels = df.groupby(session_col)[label_col].max()
    sess_scores = df.groupby(session_col)[prob_col].agg(agg_fn)
    common = sess_labels.index.intersection(sess_scores.index)
    return common.values, sess_labels.loc[common].values, sess_scores.loc[common].values


def _compute_session_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute all standard metrics at a given threshold."""
    out = {}
    if len(np.unique(y_true)) > 1:
        out["session_roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["session_pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["session_roc_auc"] = float("nan")
        out["session_pr_auc"] = float("nan")

    cm = confusion_at_threshold(y_true, y_score, threshold)
    out["block_recall"] = cm["recall"]
    out["block_fpr"] = cm["fpr"]
    out["precision"] = cm["precision"]
    out["f1"] = cm["f1"]
    return out


def bootstrap_session_metrics(
    preds_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    prob_col: str = "prob_iso",
    agg_name: str = "p90",
    session_col: str = "capture_id",
    label_col: str = "label",
    dataset_col: str = "dataset",
    split: str = "test",
    val_target_fpr: float = 0.0,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ds_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Bootstrap session-level metrics with 95% CIs.

    Bootstraps over SESSIONS (not flows) to respect data hierarchy.
    Threshold is derived from val_df (not bootstrapped — same threshold for all).

    Returns dict with metric -> {mean, ci_lower, ci_upper, std}.
    """
    agg_fn = AGG_FUNCTIONS[agg_name]

    # Get test sessions
    test = preds_df[preds_df["split"] == split].copy()
    if ds_filter is not None:
        test = test[test[dataset_col] == ds_filter]

    if len(test) == 0:
        return {"error": f"No data for split={split}, ds_filter={ds_filter}"}

    # Compute threshold from val
    if val_df is not None:
        val = val_df
    else:
        val = preds_df[preds_df["split"] == "val"]

    _, val_y, val_s = _aggregate_to_sessions(val, prob_col, agg_fn, session_col, label_col)
    if len(val_y) > 0 and len(np.unique(val_y)) > 1:
        thr = threshold_at_fpr(val_y, val_s, val_target_fpr, warn_resolution=False)
    else:
        thr = 0.5

    # Get test sessions
    sess_ids, sess_y, sess_s = _aggregate_to_sessions(test, prob_col, agg_fn, session_col, label_col)
    n_sessions = len(sess_ids)

    if n_sessions < 5 or len(np.unique(sess_y)) < 2:
        return {"error": f"Insufficient sessions ({n_sessions}) or single class"}

    # Point estimate
    point = _compute_session_metrics(sess_y, sess_s, thr)
    point["threshold"] = float(thr)
    point["n_sessions"] = n_sessions

    # Bootstrap
    rng = np.random.RandomState(seed)
    boot_results = {k: [] for k in point if k not in ("threshold", "n_sessions")}

    for _ in range(n_bootstrap):
        idx = rng.choice(n_sessions, size=n_sessions, replace=True)
        by = sess_y[idx]
        bs = sess_s[idx]
        if len(np.unique(by)) < 2:
            continue
        bm = _compute_session_metrics(by, bs, thr)
        for k in boot_results:
            boot_results[k].append(bm[k])

    # Compute CIs
    results = {"threshold": float(thr), "n_sessions": n_sessions}
    for k, vals in boot_results.items():
        arr = np.array(vals)
        results[k] = {
            "mean": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
            "std": float(np.std(arr)),
            "point_estimate": point[k],
        }

    return results


def bootstrap_per_dataset(
    preds_df: pd.DataFrame,
    prob_col: str = "prob_iso",
    agg_name: str = "p90",
    session_col: str = "capture_id",
    label_col: str = "label",
    dataset_col: str = "dataset",
    val_target_fpr: float = 0.0,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Run bootstrap for pooled test AND each dataset separately.
    Returns {pooled: {...}, iscx: {...}, vnat: {...}, usbvpn: {...}}.
    """
    results = {}

    # Pooled
    results["pooled"] = bootstrap_session_metrics(
        preds_df, prob_col=prob_col, agg_name=agg_name,
        session_col=session_col, label_col=label_col,
        dataset_col=dataset_col, val_target_fpr=val_target_fpr,
        n_bootstrap=n_bootstrap, seed=seed,
    )

    # Per dataset
    test = preds_df[preds_df["split"] == "test"]
    for ds in sorted(test[dataset_col].unique()):
        results[ds] = bootstrap_session_metrics(
            preds_df, prob_col=prob_col, agg_name=agg_name,
            session_col=session_col, label_col=label_col,
            dataset_col=dataset_col, val_target_fpr=val_target_fpr,
            n_bootstrap=n_bootstrap, seed=seed, ds_filter=ds,
        )

    return results


def threshold_robustness_sweep(
    preds_df: pd.DataFrame,
    base_threshold: float,
    prob_col: str = "prob_iso",
    agg_name: str = "p90",
    session_col: str = "capture_id",
    label_col: str = "label",
    dataset_col: str = "dataset",
    deltas: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Sweep threshold around base_threshold and report metric sensitivity.

    Returns DataFrame with columns: delta, threshold, pooled_recall, pooled_fpr,
    iscx_fpr, usbvpn_recall, vnat_recall, etc.
    """
    if deltas is None:
        deltas = [-0.10, -0.05, -0.03, -0.02, -0.01, -0.005,
                  0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]

    agg_fn = AGG_FUNCTIONS[agg_name]
    test = preds_df[preds_df["split"] == "test"]

    # Get pooled sessions
    _, pooled_y, pooled_s = _aggregate_to_sessions(test, prob_col, agg_fn, session_col, label_col)

    # Per-dataset sessions
    ds_sessions = {}
    for ds in sorted(test[dataset_col].unique()):
        ds_test = test[test[dataset_col] == ds]
        _, dy, ds_score = _aggregate_to_sessions(ds_test, prob_col, agg_fn, session_col, label_col)
        ds_sessions[ds] = (dy, ds_score)

    rows = []
    for d in deltas:
        thr = base_threshold + d
        row = {"delta": d, "threshold": thr}

        if len(pooled_y) > 0 and len(np.unique(pooled_y)) > 1:
            cm = confusion_at_threshold(pooled_y, pooled_s, thr)
            row["pooled_recall"] = cm["recall"]
            row["pooled_fpr"] = cm["fpr"]
            row["pooled_precision"] = cm["precision"]

        for ds, (dy, ds_score) in ds_sessions.items():
            if len(dy) > 0:
                dcm = confusion_at_threshold(dy, ds_score, thr)
                row[f"{ds}_recall"] = dcm["recall"]
                row[f"{ds}_fpr"] = dcm["fpr"]

        rows.append(row)

    return pd.DataFrame(rows)


def fpr_resolution_report(
    preds_df: pd.DataFrame,
    prob_col: str = "prob_iso",
    agg_name: str = "p90",
    session_col: str = "capture_id",
    label_col: str = "label",
    dataset_col: str = "dataset",
) -> pd.DataFrame:
    """
    Report achievable FPR resolution for val split, pooled and per-dataset.
    """
    agg_fn = AGG_FUNCTIONS[agg_name]
    val = preds_df[preds_df["split"] == "val"]

    rows = []
    for scope in ["pooled"] + sorted(val[dataset_col].unique()):
        if scope == "pooled":
            subset = val
        else:
            subset = val[val[dataset_col] == scope]

        _, vy, vs = _aggregate_to_sessions(subset, prob_col, agg_fn, session_col, label_col)
        n_benign = int((vy == 0).sum())
        fpr_resolution = 1.0 / max(n_benign, 1)

        rows.append({
            "scope": scope,
            "n_sessions": len(vy),
            "n_benign_sessions": n_benign,
            "n_vpn_sessions": int((vy == 1).sum()),
            "fpr_resolution": fpr_resolution,
            "min_achievable_fpr_pct": fpr_resolution * 100,
            "note": (
                f"Cannot distinguish FPR targets below {fpr_resolution:.4f} "
                f"(1/{n_benign} benign sessions)"
            ),
        })

    return pd.DataFrame(rows)


