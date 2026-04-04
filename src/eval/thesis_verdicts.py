# src/eval/thesis_verdicts.py
"""
Revised thesis-safe verdict framework for NB30 domain-robustness experiments.

Replaces the overly strict LEAKY/INVALID verdicts with a nuanced labelling
system that supports wording like:
    "domain-aware but deployable with adaptive thresholds"

Labels:
    deployment-positive   — strong metrics, acceptable domain awareness
    domain-sensitive      — domain detector AUC > 0.95
    calibration-sensitive — threshold shift > 0.10 across domains
    research-only         — interesting but not deployment-ready
    underfit              — test AUC < 0.80
    overfit               — train-test AUC gap > 0.05

An experiment can carry MULTIPLE labels (e.g., deployment-positive + domain-sensitive).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import math


def _safe_get(d: dict, key: str, default=float("nan")) -> float:
    """Safely get a numeric value from a dict."""
    v = d.get(key, default)
    if v is None:
        return float("nan")
    try:
        f = float(v)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def thesis_verdict(
    experiment_summary: Dict[str, Any],
    nb29_reference: Optional[Dict[str, Any]] = None,
    *,
    domain_detector_auc: Optional[float] = None,
    threshold_shift: Optional[float] = None,
    # Configurable thresholds
    domain_auc_warn: float = 0.95,
    threshold_shift_warn: float = 0.10,
    test_auc_min: float = 0.80,
    overfit_gap_max: float = 0.05,
    session_auc_strong: float = 0.90,
    block_recall_acceptable: float = 0.50,
) -> Dict[str, Any]:
    """
    Compute a nuanced thesis-safe verdict for one experiment.

    Returns:
        dict with:
            experiment: str
            labels: list[str]  — one or more verdict labels
            primary_verdict: str — the single most important label
            reasons: list[str] — human-readable explanation per label
            metrics_used: dict — the metric values used for the decision
            deployment_recommendation: str — one-sentence recommendation
    """
    s = experiment_summary
    labels: List[str] = []
    reasons: List[str] = []

    # ── Extract metrics ──
    test_auc = _safe_get(s, "test_auc")
    train_auc = _safe_get(s, "train_auc")
    session_auc = _safe_get(s, "session_roc_auc_p90",
                            _safe_get(s, "session_roc_auc"))
    block_recall = _safe_get(s, "block_recall_p90",
                             _safe_get(s, "block_recall"))
    block_fpr = _safe_get(s, "block_fpr_p90",
                          _safe_get(s, "block_fpr"))

    dd_auc = domain_detector_auc
    if dd_auc is None:
        dd_auc = _safe_get(s, "domain_detector_auc")

    thr_shift = threshold_shift
    if thr_shift is None:
        thr_shift = _safe_get(s, "threshold_shift")

    train_test_gap = (train_auc - test_auc) if (
        math.isfinite(train_auc) and math.isfinite(test_auc)
    ) else float("nan")

    metrics_used = {
        "test_auc": test_auc,
        "train_auc": train_auc,
        "session_auc_p90": session_auc,
        "block_recall_p90": block_recall,
        "block_fpr_p90": block_fpr,
        "domain_detector_auc": dd_auc,
        "threshold_shift": thr_shift,
        "train_test_gap": train_test_gap,
    }

    # ── Rule 1: Underfit ──
    if math.isfinite(test_auc) and test_auc < test_auc_min:
        labels.append("underfit")
        reasons.append(f"test AUC = {test_auc:.4f} < {test_auc_min}")

    # ── Rule 2: Overfit ──
    if math.isfinite(train_test_gap) and train_test_gap > overfit_gap_max:
        labels.append("overfit")
        reasons.append(
            f"train-test AUC gap = {train_test_gap:.4f} > {overfit_gap_max}"
        )

    # ── Rule 3: Domain-sensitive ──
    if math.isfinite(dd_auc) and dd_auc > domain_auc_warn:
        labels.append("domain-sensitive")
        reasons.append(
            f"domain detector AUC = {dd_auc:.4f} > {domain_auc_warn}"
        )

    # ── Rule 4: Calibration-sensitive ──
    if math.isfinite(thr_shift) and thr_shift > threshold_shift_warn:
        labels.append("calibration-sensitive")
        reasons.append(
            f"threshold shift = {thr_shift:.4f} > {threshold_shift_warn}"
        )

    # ── Rule 5: Deployment-positive ──
    session_ok = math.isfinite(session_auc) and session_auc >= session_auc_strong
    recall_ok = math.isfinite(block_recall) and block_recall >= block_recall_acceptable
    auc_ok = math.isfinite(test_auc) and test_auc >= test_auc_min

    if session_ok and recall_ok and auc_ok:
        labels.append("deployment-positive")
        reasons.append(
            f"session AUC = {session_auc:.4f} >= {session_auc_strong}, "
            f"block recall = {block_recall:.4f} >= {block_recall_acceptable}, "
            f"test AUC = {test_auc:.4f} >= {test_auc_min}"
        )

    # ── Rule 5b: Requires-local-calibration ──
    # If deployment-positive but calibration-sensitive or domain-sensitive,
    # the model needs adaptive thresholds — deployment is conditional.
    if "deployment-positive" in labels and (
        "calibration-sensitive" in labels or "domain-sensitive" in labels
    ):
        labels.append("requires-local-calibration")
        reasons.append(
            "deployment-positive detector with domain/calibration sensitivity "
            "requires local threshold calibration for safe deployment"
        )

    # ── Rule 6: Research-only (fallback) ──
    if not labels:
        labels.append("research-only")
        reasons.append("does not meet any deployment-positive criteria")
    elif "deployment-positive" not in labels:
        # Has issues but not deployment-positive
        if "underfit" not in labels and "overfit" not in labels:
            labels.append("research-only")
            reasons.append(
                "has domain/calibration sensitivity without meeting deployment criteria"
            )

    # ── Primary verdict ──
    # Priority: underfit > overfit > research-only > calibration-sensitive > domain-sensitive > deployment-positive
    priority = [
        "underfit", "overfit", "research-only",
        "calibration-sensitive", "domain-sensitive", "deployment-positive",
    ]
    primary = "research-only"
    for p in priority:
        if p in labels:
            primary = p
            break

    # If deployment-positive is present alongside sensitivities, upgrade primary
    if "deployment-positive" in labels:
        if primary in ("domain-sensitive", "calibration-sensitive"):
            primary = "deployment-positive"

    # ── Deployment recommendation ──
    if "deployment-positive" in labels and not {"underfit", "overfit"} & set(labels):
        sensitivities = [l for l in labels if l.endswith("-sensitive")]
        if sensitivities:
            rec = (
                f"Conditional deployment readiness. "
                f"Strong detector with sensitivities: {', '.join(sensitivities)}. "
                f"Requires local threshold calibration or adaptive thresholds. "
                f"Monitor domain drift and recalibrate periodically."
            )
        else:
            rec = "Deployment-ready. Standard monitoring recommended."
    elif "underfit" in labels:
        rec = "Not deployment-ready: model underperforms on test data."
    elif "overfit" in labels:
        rec = "Not deployment-ready: significant overfitting detected."
    else:
        rec = "Research-only: does not meet deployment criteria."

    return {
        "experiment": s.get("experiment", "unknown"),
        "labels": labels,
        "primary_verdict": primary,
        "reasons": reasons,
        "metrics_used": metrics_used,
        "deployment_recommendation": rec,
    }


def compare_verdicts(
    old_verdict: Dict[str, Any],
    new_verdict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a comparison row between old (NB30) and new (thesis) verdicts.
    """
    return {
        "experiment": new_verdict.get("experiment", "unknown"),
        "old_verdict": old_verdict.get("verdict", "N/A"),
        "new_primary_verdict": new_verdict["primary_verdict"],
        "new_labels": ", ".join(new_verdict["labels"]),
        "old_reasons": "; ".join(old_verdict.get("reasons", [])),
        "new_reasons": "; ".join(new_verdict["reasons"]),
        "deployment_recommendation": new_verdict["deployment_recommendation"],
    }


def verdict_table(
    experiments: List[Dict[str, Any]],
    nb29_reference: Optional[Dict[str, Any]] = None,
    old_verdicts: Optional[List[Dict[str, Any]]] = None,
    threshold_shifts: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Produce a full comparison table for all experiments.

    Args:
        experiments: list of experiment summary dicts
        nb29_reference: NB29 reference metrics (optional)
        old_verdicts: list of old verdict dicts from NB30 (optional)
        threshold_shifts: map experiment_name -> threshold_shift (optional)

    Returns:
        list of comparison rows
    """
    old_map = {}
    if old_verdicts:
        for v in old_verdicts:
            old_map[v.get("experiment", "")] = v

    rows = []
    for exp in experiments:
        name = exp.get("experiment", "")
        thr_shift = None
        if threshold_shifts and name in threshold_shifts:
            thr_shift = threshold_shifts[name]

        new_v = thesis_verdict(
            exp, nb29_reference,
            threshold_shift=thr_shift,
        )
        old_v = old_map.get(name, {"verdict": "N/A", "reasons": []})
        rows.append(compare_verdicts(old_v, new_v))

    return rows

