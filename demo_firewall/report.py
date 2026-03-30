# demo_firewall/report.py
"""
Robustness reporting and evaluation metrics.

Generates structured reports for firewall inference results,
including session-level metrics, calibration diagnostics,
and provenance metadata.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils.logging import setup_logger

from demo_firewall.policy import SessionDecision, Decision

logger = setup_logger(name="firewall.report")


def compute_evaluation_metrics(
    session_decisions: List[SessionDecision],
    include_ground_truth: bool = True,
) -> Dict[str, Any]:
    """
    Compute evaluation metrics from session decisions.

    Parameters
    ----------
    session_decisions : list of SessionDecision
        Results from FirewallPolicy.
    include_ground_truth : bool
        If False, skip metrics requiring labels (live deployment mode).

    Returns
    -------
    dict
        Structured metrics report.
    """
    if not session_decisions:
        return {"error": "No session decisions to evaluate."}

    n_sessions = len(session_decisions)
    n_blocked = sum(1 for d in session_decisions if d.decision == Decision.BLOCK)
    n_flagged = sum(1 for d in session_decisions if d.decision == Decision.FLAG)
    n_allowed = sum(1 for d in session_decisions if d.decision == Decision.ALLOW)

    scores = np.array([d.session_score for d in session_decisions])

    report: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_sessions": n_sessions,
        "n_blocked": n_blocked,
        "n_flagged": n_flagged,
        "n_allowed": n_allowed,
        "block_rate": n_blocked / n_sessions if n_sessions > 0 else 0.0,
        "flag_rate": n_flagged / n_sessions if n_sessions > 0 else 0.0,
        "score_stats": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "p25": float(np.percentile(scores, 25)),
            "median": float(np.percentile(scores, 50)),
            "p75": float(np.percentile(scores, 75)),
            "p90": float(np.percentile(scores, 90)),
            "max": float(np.max(scores)),
        },
        "deployment_mode": session_decisions[0].deployment_mode,
        "aggregation_rule": session_decisions[0].aggregation_rule,
        "block_threshold": session_decisions[0].block_threshold,
        "flag_threshold": session_decisions[0].flag_threshold,
    }

    # Ground-truth metrics (only if labels are available and meaningful)
    if not include_ground_truth:
        return report

    # Try to extract labels from flow decisions
    # Labels come from the capture-level max label
    labels = []
    has_labels = True
    for d in session_decisions:
        if d.flow_decisions:
            # Label -1 means unlabeled
            flow_labels = [fd.probability for fd in d.flow_decisions]
            # We need actual labels — check if they were propagated
            pass

    # If we have ground truth labels from predictions
    # (they would need to be passed separately for proper evaluation)
    report["ground_truth_available"] = False
    report["note"] = (
        "For ground-truth metrics, use evaluate_with_labels() "
        "with a labeled predictions DataFrame."
    )

    return report


def evaluate_with_labels(
    flow_preds: pd.DataFrame,
    session_decisions: List[SessionDecision],
    prob_col: str = "prob_cal",
    label_col: str = "label",
    session_col: str = "capture_id",
    dataset_col: str = "dataset",
    include_flow_metrics: bool = True,
    include_per_dataset: bool = True,
    include_fpr_sweep: bool = True,
) -> Dict[str, Any]:
    """
    Full evaluation with ground-truth labels.

    This is the primary evaluation function for research/validation.

    Parameters
    ----------
    flow_preds : pd.DataFrame
        Flow-level predictions with labels.
    session_decisions : list of SessionDecision
        Results from policy engine.
    prob_col : str
        Probability column name.
    label_col : str
        Label column name.
    session_col : str
        Session identifier column name.
    dataset_col : str
        Dataset identifier column name (for per-dataset breakdown).
    include_flow_metrics : bool
        If True, compute flow-level ROC-AUC and PR-AUC.
    include_per_dataset : bool
        If True, compute per-dataset session metrics.
    include_fpr_sweep : bool
        If True, compute recall-vs-FPR budget sweep.

    Returns
    -------
    dict
        Complete metrics report with flow AUC, session AUC, PR-AUC,
        block recall, block FPR, flagged FPR, per-dataset metrics,
        confusion matrices, and recall-vs-FPR sweep.
    """
    # ── Flow-level metrics ──────────────────────────────────
    metrics: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
    }

    if include_flow_metrics and prob_col in flow_preds.columns and label_col in flow_preds.columns:
        flow_y = flow_preds[label_col].values.astype(int)
        flow_p = flow_preds[prob_col].values.astype(float)
        valid_flow = flow_y >= 0  # exclude unlabeled flows (label == -1)
        flow_y = flow_y[valid_flow]
        flow_p = flow_p[valid_flow]

        if len(np.unique(flow_y)) > 1 and len(flow_y) > 0:
            metrics["flow_roc_auc"] = float(roc_auc_score(flow_y, flow_p))
            metrics["flow_pr_auc"] = float(average_precision_score(flow_y, flow_p))
        else:
            metrics["flow_roc_auc"] = None
            metrics["flow_pr_auc"] = None
        metrics["n_flows_evaluated"] = int(len(flow_y))
    else:
        metrics["flow_roc_auc"] = None
        metrics["flow_pr_auc"] = None

    # ── Session-level ground truth ──────────────────────────
    session_labels = (
        flow_preds.groupby(session_col)[label_col]
        .max()
        .to_dict()
    )

    y_true = []
    y_score = []
    y_pred_block = []
    y_pred_flag = []
    session_ids = []

    for d in session_decisions:
        label = session_labels.get(d.capture_id, -1)
        if label < 0:
            continue
        y_true.append(int(label))
        y_score.append(d.session_score)
        y_pred_block.append(1 if d.decision == Decision.BLOCK else 0)
        y_pred_flag.append(1 if d.decision in (Decision.BLOCK, Decision.FLAG) else 0)
        session_ids.append(d.capture_id)

    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)
    y_pred_block = np.array(y_pred_block, dtype=int)
    y_pred_flag = np.array(y_pred_flag, dtype=int)

    metrics["n_sessions_evaluated"] = len(y_true)
    metrics["n_positive"] = int(y_true.sum())
    metrics["n_negative"] = int((1 - y_true).sum())

    # Session AUC
    if len(np.unique(y_true)) > 1:
        metrics["session_roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["session_pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["session_roc_auc"] = None
        metrics["session_pr_auc"] = None

    # Block metrics (zero-FPR operating point)
    block_tp = int(np.sum((y_pred_block == 1) & (y_true == 1)))
    block_fp = int(np.sum((y_pred_block == 1) & (y_true == 0)))
    block_fn = int(np.sum((y_pred_block == 0) & (y_true == 1)))
    block_tn = int(np.sum((y_pred_block == 0) & (y_true == 0)))

    block_recall = block_tp / max(block_tp + block_fn, 1)
    block_fpr = block_fp / max(block_fp + block_tn, 1)
    block_precision = block_tp / max(block_tp + block_fp, 1)

    metrics["block_recall"] = float(block_recall)
    metrics["block_fpr"] = float(block_fpr)
    metrics["block_precision"] = float(block_precision)
    metrics["block_confusion"] = {
        "tp": block_tp, "fp": block_fp, "fn": block_fn, "tn": block_tn,
    }

    # Flagged metrics (block + flag)
    flag_tp = int(np.sum((y_pred_flag == 1) & (y_true == 1)))
    flag_fp = int(np.sum((y_pred_flag == 1) & (y_true == 0)))
    flag_fn = int(np.sum((y_pred_flag == 0) & (y_true == 1)))
    flag_tn = int(np.sum((y_pred_flag == 0) & (y_true == 0)))

    flagged_recall = flag_tp / max(flag_tp + flag_fn, 1)
    flagged_fpr = flag_fp / max(flag_fp + flag_tn, 1)

    metrics["flagged_recall"] = float(flagged_recall)
    metrics["flagged_fpr"] = float(flagged_fpr)
    metrics["flagged_confusion"] = {
        "tp": flag_tp, "fp": flag_fp, "fn": flag_fn, "tn": flag_tn,
    }

    # Provenance
    if session_decisions:
        d0 = session_decisions[0]
        metrics["deployment_mode"] = d0.deployment_mode
        metrics["aggregation_rule"] = d0.aggregation_rule
        metrics["block_threshold"] = d0.block_threshold
        metrics["flag_threshold"] = d0.flag_threshold

    # ── Per-dataset session metrics ─────────────────────────
    if include_per_dataset and dataset_col in flow_preds.columns:
        # Map capture_id → dataset
        capture_dataset = (
            flow_preds.groupby(session_col)[dataset_col]
            .first()
            .to_dict()
        )

        per_ds: Dict[str, Any] = {}
        datasets = sorted(set(capture_dataset.values()))

        for ds in datasets:
            ds_mask = np.array([
                capture_dataset.get(sid, "") == ds
                for sid in session_ids
            ], dtype=bool)

            if ds_mask.sum() == 0:
                continue

            ds_y = y_true[ds_mask]
            ds_s = y_score[ds_mask]
            ds_block = y_pred_block[ds_mask]

            ds_n_pos = int(ds_y.sum())
            ds_n_neg = int((1 - ds_y).sum())

            ds_metrics: Dict[str, Any] = {
                "n_sessions": int(ds_mask.sum()),
                "n_positive": ds_n_pos,
                "n_negative": ds_n_neg,
            }

            if len(np.unique(ds_y)) > 1:
                ds_metrics["session_roc_auc"] = float(roc_auc_score(ds_y, ds_s))
                ds_metrics["session_pr_auc"] = float(average_precision_score(ds_y, ds_s))
            else:
                ds_metrics["session_roc_auc"] = None
                ds_metrics["session_pr_auc"] = None

            ds_block_tp = int(np.sum((ds_block == 1) & (ds_y == 1)))
            ds_block_fp = int(np.sum((ds_block == 1) & (ds_y == 0)))
            ds_block_fn = int(np.sum((ds_block == 0) & (ds_y == 1)))
            ds_block_tn = int(np.sum((ds_block == 0) & (ds_y == 0)))

            ds_metrics["block_recall"] = ds_block_tp / max(ds_block_tp + ds_block_fn, 1)
            ds_metrics["block_fpr"] = ds_block_fp / max(ds_block_fp + ds_block_tn, 1)
            ds_metrics["block_confusion"] = {
                "tp": ds_block_tp, "fp": ds_block_fp,
                "fn": ds_block_fn, "tn": ds_block_tn,
            }

            per_ds[ds] = ds_metrics

        metrics["per_dataset"] = per_ds

    # ── Recall vs FPR budget sweep ──────────────────────────
    if include_fpr_sweep and len(np.unique(y_true)) > 1:
        benign_scores = y_score[y_true == 0]
        vpn_scores = y_score[y_true == 1]

        fpr_budgets = np.arange(0.0, 0.155, 0.005)
        sweep: List[Dict[str, Any]] = []

        for fpr_budget in fpr_budgets:
            if fpr_budget == 0.0:
                thr = float(np.max(benign_scores))
            else:
                thr = float(np.quantile(benign_scores, 1.0 - fpr_budget))

            n_blocked_vpn = int(np.sum(vpn_scores > thr))
            n_blocked_benign = int(np.sum(benign_scores > thr))
            recall = n_blocked_vpn / max(len(vpn_scores), 1)
            actual_fpr = n_blocked_benign / max(len(benign_scores), 1)

            sweep.append({
                "fpr_budget": round(float(fpr_budget), 4),
                "threshold": round(thr, 6),
                "block_recall": round(recall, 4),
                "actual_fpr": round(actual_fpr, 4),
                "n_blocked_vpn": n_blocked_vpn,
                "n_blocked_benign": n_blocked_benign,
            })

        metrics["recall_vs_fpr_sweep"] = sweep

    return metrics


def format_report(
    metrics: Dict[str, Any],
    predictor_diagnostics: Optional[Dict[str, Any]] = None,
    policy_diagnostics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format a human-readable report from metrics.

    Returns
    -------
    str
        Formatted report text.
    """
    lines = [
        "=" * 70,
        "  VPN FIREWALL — EVALUATION REPORT",
        "=" * 70,
        "",
    ]

    # Header
    lines.append(f"  Timestamp:        {metrics.get('timestamp', 'N/A')}")
    lines.append(f"  Deployment Mode:  {metrics.get('deployment_mode', 'N/A')}")
    lines.append(f"  Aggregation Rule: {metrics.get('aggregation_rule', 'N/A')}")
    lines.append("")

    # Session counts
    lines.append("  SESSION SUMMARY")
    lines.append("  " + "-" * 40)
    lines.append(f"  Sessions evaluated: {metrics.get('n_sessions_evaluated', metrics.get('n_sessions', 'N/A'))}")
    lines.append(f"  Positive (VPN):     {metrics.get('n_positive', 'N/A')}")
    lines.append(f"  Negative (Benign):  {metrics.get('n_negative', 'N/A')}")
    if metrics.get("n_flows_evaluated") is not None:
        lines.append(f"  Flows evaluated:    {metrics['n_flows_evaluated']}")
    lines.append("")

    # Thresholds
    lines.append("  THRESHOLDS")
    lines.append("  " + "-" * 40)
    lines.append(f"  Block threshold:  {metrics.get('block_threshold', 'N/A'):.6f}" if isinstance(metrics.get('block_threshold'), float) else f"  Block threshold:  {metrics.get('block_threshold', 'N/A')}")
    lines.append(f"  Flag threshold:   {metrics.get('flag_threshold', 'N/A'):.6f}" if isinstance(metrics.get('flag_threshold'), float) else f"  Flag threshold:   {metrics.get('flag_threshold', 'N/A')}")
    lines.append("")

    # Flow-level metrics
    if metrics.get("flow_roc_auc") is not None or metrics.get("flow_pr_auc") is not None:
        lines.append("  FLOW-LEVEL METRICS")
        lines.append("  " + "-" * 40)
        f_auc = metrics.get("flow_roc_auc")
        f_pr = metrics.get("flow_pr_auc")
        lines.append(f"  Flow ROC-AUC:      {f_auc:.4f}" if f_auc is not None else "  Flow ROC-AUC:      N/A")
        lines.append(f"  Flow PR-AUC:       {f_pr:.4f}" if f_pr is not None else "  Flow PR-AUC:       N/A")
        lines.append("")

    # Key session metrics
    if "session_roc_auc" in metrics:
        lines.append("  SESSION-LEVEL METRICS")
        lines.append("  " + "-" * 40)
        auc = metrics.get("session_roc_auc")
        pr_auc = metrics.get("session_pr_auc")
        lines.append(f"  Session ROC-AUC:   {auc:.4f}" if auc is not None else "  Session ROC-AUC:   N/A")
        lines.append(f"  Session PR-AUC:    {pr_auc:.4f}" if pr_auc is not None else "  Session PR-AUC:    N/A")
        lines.append(f"  Block Recall:      {metrics.get('block_recall', 'N/A'):.4f}" if isinstance(metrics.get('block_recall'), float) else "")
        lines.append(f"  Block FPR:         {metrics.get('block_fpr', 'N/A'):.4f}" if isinstance(metrics.get('block_fpr'), float) else "")
        lines.append(f"  Block Precision:   {metrics.get('block_precision', 'N/A'):.4f}" if isinstance(metrics.get('block_precision'), float) else "")
        lines.append(f"  Flagged Recall:    {metrics.get('flagged_recall', 'N/A'):.4f}" if isinstance(metrics.get('flagged_recall'), float) else "")
        lines.append(f"  Flagged FPR:       {metrics.get('flagged_fpr', 'N/A'):.4f}" if isinstance(metrics.get('flagged_fpr'), float) else "")
        lines.append("")

    # Confusion matrices
    block_cm = metrics.get("block_confusion")
    if block_cm:
        lines.append("  BLOCK CONFUSION MATRIX")
        lines.append("  " + "-" * 40)
        lines.append(f"  TP={block_cm['tp']}  FP={block_cm['fp']}  FN={block_cm['fn']}  TN={block_cm['tn']}")
        lines.append("")

    flag_cm = metrics.get("flagged_confusion")
    if flag_cm:
        lines.append("  FLAGGED (BLOCK+FLAG) CONFUSION MATRIX")
        lines.append("  " + "-" * 40)
        lines.append(f"  TP={flag_cm['tp']}  FP={flag_cm['fp']}  FN={flag_cm['fn']}  TN={flag_cm['tn']}")
        lines.append("")

    # Per-dataset metrics
    per_ds = metrics.get("per_dataset")
    if per_ds:
        lines.append("  PER-DATASET BREAKDOWN")
        lines.append("  " + "-" * 40)
        header = f"  {'Dataset':<12s} {'Sessions':>8s} {'AUC':>8s} {'BlkRecall':>10s} {'BlkFPR':>8s}"
        lines.append(header)
        for ds_name, ds_m in sorted(per_ds.items()):
            n = ds_m.get("n_sessions", "?")
            auc_v = ds_m.get("session_roc_auc")
            br_v = ds_m.get("block_recall")
            bf_v = ds_m.get("block_fpr")
            auc_s = f"{auc_v:.4f}" if auc_v is not None else "N/A"
            br_s = f"{br_v:.4f}" if br_v is not None else "N/A"
            bf_s = f"{bf_v:.4f}" if bf_v is not None else "N/A"
            lines.append(f"  {ds_name:<12s} {str(n):>8s} {auc_s:>8s} {br_s:>10s} {bf_s:>8s}")
        lines.append("")

    # Recall vs FPR sweep (compact view — first 7 rows)
    sweep = metrics.get("recall_vs_fpr_sweep")
    if sweep:
        lines.append("  RECALL vs FPR BUDGET SWEEP (sample)")
        lines.append("  " + "-" * 40)
        lines.append(f"  {'FPR Budget':>10s} {'Threshold':>10s} {'Recall':>8s} {'ActualFPR':>10s}")
        for row in sweep[:7]:
            lines.append(
                f"  {row['fpr_budget']:>10.4f} {row['threshold']:>10.6f} "
                f"{row['block_recall']:>8.4f} {row['actual_fpr']:>10.4f}"
            )
        if len(sweep) > 7:
            lines.append(f"  ... ({len(sweep)} total operating points)")
        lines.append("")

    # Predictor diagnostics
    if predictor_diagnostics:
        lines.append("  PREDICTOR DIAGNOSTICS")
        lines.append("  " + "-" * 40)
        lines.append(f"  Models loaded:     {predictor_diagnostics.get('n_models_total', 'N/A')}")
        lines.append(f"  Families:          {predictor_diagnostics.get('n_families', 'N/A')}")
        lines.append(f"  Calibration:       {predictor_diagnostics.get('calibration_method', 'N/A')}")
        lines.append(f"  Features:          {predictor_diagnostics.get('n_features', 'N/A')}")
        lines.append("")

    # Policy diagnostics
    if policy_diagnostics:
        lines.append("  POLICY DIAGNOSTICS")
        lines.append("  " + "-" * 40)
        lines.append(f"  Mode:              {policy_diagnostics.get('mode', 'N/A')}")
        lines.append(f"  Zero-FPR enforced: {policy_diagnostics.get('enforce_zero_block_fpr', 'N/A')}")
        lines.append(f"  Target FPR:        {policy_diagnostics.get('target_fpr', 'N/A')}")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def save_report(
    report_dict: Dict[str, Any],
    output_dir: Path,
    prefix: str = "firewall_report",
) -> Path:
    """Save report as JSON to the output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{prefix}_{ts}.json"

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with path.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, default=_convert)

    logger.info(f"Report saved to {path}")
    return path


