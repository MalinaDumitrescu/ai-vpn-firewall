# src/eval/calibration_diagnostics.py
"""
Calibration diagnostic utilities for thesis finalization.

Provides:
- Expected Calibration Error (ECE)
- Brier score (wraps sklearn)
- Reliability curve data generation
- Cross-domain calibration shift detection
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import brier_score_loss


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE) and per-bin reliability data.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Returns dict with:
        ece: float
        bin_edges: list of bin edges
        bin_accs: list of per-bin accuracies
        bin_confs: list of per-bin mean confidences
        bin_counts: list of per-bin sample counts
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        count = mask.sum()
        bin_counts.append(int(count))

        if count == 0:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            continue

        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        bin_accs.append(acc)
        bin_confs.append(conf)
        ece += (count / n) * abs(acc - conf)

    return {
        "ece": float(ece),
        "n_bins": n_bins,
        "bin_edges": bin_edges.tolist(),
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
    }


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score (lower is better)."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    return float(brier_score_loss(y_true, y_prob))


def calibration_summary(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_iso: Optional[np.ndarray] = None,
    y_prob_platt: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Full calibration summary for one set of predictions.

    Returns dict with ECE, Brier, and reliability data for each
    calibration variant (raw, isotonic, platt).
    """
    out: Dict[str, Any] = {}

    for name, probs in [("raw", y_prob_raw), ("isotonic", y_prob_iso), ("platt", y_prob_platt)]:
        if probs is None:
            continue
        probs = np.asarray(probs, dtype=float).ravel()
        ece_data = expected_calibration_error(y_true, probs, n_bins=n_bins)
        out[name] = {
            "ece": ece_data["ece"],
            "brier": brier_score(y_true, probs),
            "reliability": ece_data,
        }

    return out


def cross_domain_calibration_shift(
    predictions_df,
    prob_col: str = "prob_iso",
    label_col: str = "label",
    dataset_col: str = "dataset",
    split: str = "test",
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compare calibration quality across datasets.

    Returns per-dataset ECE/Brier and a shift summary.
    """
    df = predictions_df
    if split:
        df = df[df["split"] == split]

    results = {}
    ece_values = []

    for ds in sorted(df[dataset_col].unique()):
        ds_df = df[df[dataset_col] == ds]
        if len(ds_df) < 10 or ds_df[label_col].nunique() < 2:
            continue
        y = ds_df[label_col].values
        p = ds_df[prob_col].values
        ece_data = expected_calibration_error(y, p, n_bins=n_bins)
        bs = brier_score(y, p)
        results[ds] = {"ece": ece_data["ece"], "brier": bs, "n": len(ds_df)}
        ece_values.append(ece_data["ece"])

    shift_summary = {}
    if len(ece_values) >= 2:
        shift_summary["ece_range"] = float(max(ece_values) - min(ece_values))
        shift_summary["ece_max"] = float(max(ece_values))
        shift_summary["ece_min"] = float(min(ece_values))
        if shift_summary["ece_range"] > 0.10:
            shift_summary["interpretation"] = "HIGH calibration shift across domains"
        elif shift_summary["ece_range"] > 0.05:
            shift_summary["interpretation"] = "MODERATE calibration shift across domains"
        else:
            shift_summary["interpretation"] = "LOW calibration shift across domains"

    return {"per_dataset": results, "shift_summary": shift_summary}


def interpret_calibration(ece: float, brier: float) -> Dict[str, str]:
    """
    Produce machine-readable calibration interpretation fields.
    """
    if ece < 0.05:
        cq = "well-calibrated"
    elif ece < 0.10:
        cq = "moderately-calibrated"
    elif ece < 0.20:
        cq = "poorly-calibrated"
    else:
        cq = "severely-miscalibrated"

    if brier < 0.10:
        ts = "low"
    elif brier < 0.20:
        ts = "moderate"
    else:
        ts = "high"

    return {
        "calibration_quality": cq,
        "threshold_stability_risk": ts,
    }



