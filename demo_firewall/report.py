# demo_firewall/report.py
"""
Robustness reporting and evaluation metrics.

Generates structured reports for firewall inference results,
including session-level metrics, calibration diagnostics,
provenance metadata, and open-set policy dashboard cards.
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

    Returns
    -------
    dict
        Complete metrics report with session AUC, PR-AUC,
        block recall, block FPR, flagged FPR.
    """
    # Session-level ground truth
    session_labels = (
        flow_preds.groupby(session_col)[label_col]
        .max()
        .to_dict()
    )

    y_true = []
    y_score = []
    y_pred_block = []
    y_pred_flag = []

    for d in session_decisions:
        label = session_labels.get(d.capture_id, -1)
        if label < 0:
            continue
        y_true.append(int(label))
        y_score.append(d.session_score)
        y_pred_block.append(1 if d.decision == Decision.BLOCK else 0)
        y_pred_flag.append(1 if d.decision in (Decision.BLOCK, Decision.FLAG) else 0)

    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)
    y_pred_block = np.array(y_pred_block, dtype=int)
    y_pred_flag = np.array(y_pred_flag, dtype=int)

    metrics: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_sessions_evaluated": len(y_true),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
    }

    # Flow-level AUC (computed from the original flow predictions)
    if label_col in flow_preds.columns and prob_col in flow_preds.columns:
        flow_labels = flow_preds[label_col].values
        flow_scores = flow_preds[prob_col].values
        if len(np.unique(flow_labels)) > 1:
            metrics["flow_roc_auc"] = float(roc_auc_score(flow_labels, flow_scores))
            metrics["flow_pr_auc"] = float(average_precision_score(flow_labels, flow_scores))
        else:
            metrics["flow_roc_auc"] = None
            metrics["flow_pr_auc"] = None
    else:
        metrics["flow_roc_auc"] = None
        metrics["flow_pr_auc"] = None

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
    lines.append("")

    # Thresholds
    lines.append("  THRESHOLDS")
    lines.append("  " + "-" * 40)
    lines.append(f"  Block threshold:  {metrics.get('block_threshold', 'N/A'):.6f}" if isinstance(metrics.get('block_threshold'), float) else f"  Block threshold:  {metrics.get('block_threshold', 'N/A')}")
    lines.append(f"  Flag threshold:   {metrics.get('flag_threshold', 'N/A'):.6f}" if isinstance(metrics.get('flag_threshold'), float) else f"  Flag threshold:   {metrics.get('flag_threshold', 'N/A')}")
    lines.append("")

    # Key metrics
    if "session_roc_auc" in metrics:
        lines.append("  KEY METRICS")
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


# ─────────────────────────────────────────────────────────────────────────────
# Open-set three-tier dashboard helpers
# ─────────────────────────────────────────────────────────────────────────────

_ACTION_COLORS = {
    "PASS": "\033[32m",            # green
    "FLAG_REVIEW": "\033[33m",     # yellow/orange
    "SIMULATED_BLOCK": "\033[31m", # red
    "BLOCK": "\033[31m",
    "FLAG": "\033[33m",
    "ALLOW": "\033[32m",
}
_RESET = "\033[0m"


def render_status_cards(
    status_cards: List[Dict[str, Any]],
    use_color: bool = False,
) -> str:
    """
    Render dashboard KPI status cards as a text block.

    Parameters
    ----------
    status_cards : list of dicts
        Produced by OpenSetFirewallPolicy.dashboard_report()["status_cards"].
    use_color : bool
        Emit ANSI color codes for terminal rendering.

    Returns
    -------
    str
        Formatted status card block.
    """
    lines = ["  ┌─ STATUS CARDS ──────────────────────────────────────────────┐"]
    for card in status_cards:
        icon = card.get("icon", "")
        title = card.get("title", "")
        value = str(card.get("value", ""))
        rate = card.get("rate", "")
        desc = card.get("description", "")
        color = ""
        reset = ""
        if use_color:
            action_key = card.get("id", "").upper().replace("_CARD", "")
            color = _ACTION_COLORS.get(action_key, "")
            reset = _RESET if color else ""
        lines.append(
            f"  │  {icon}  {color}{title:30s}{reset}  "
            f"{value:>10}  ({rate:>7})  {desc}"
        )
    lines.append("  └─────────────────────────────────────────────────────────────┘")
    return "\n".join(lines)


def render_events_table(
    events: List[Dict[str, Any]],
    max_rows: int = 20,
    use_color: bool = False,
) -> str:
    """
    Render the recent events table as formatted text.

    Parameters
    ----------
    events : list of dicts
        From OpenSetFirewallPolicy.dashboard_report()["recent_events"].
    max_rows : int
        Maximum number of rows to render.
    use_color : bool
        Emit ANSI color codes.

    Returns
    -------
    str
    """
    rows = events[:max_rows]
    if not rows:
        return "  (no events)"

    header = (
        f"  {'Timestamp':19s}  "
        f"{'Capture':35s}  "
        f"{'Score':>7}  "
        f"{'Action':20s}  "
        f"{'Margin':>7}  "
        f"{'Flows':>6}  "
        f"{'Dataset':12s}  "
        f"{'OK?':>4}"
    )
    sep = "  " + "-" * 118

    lines = [
        "  ┌─ RECENT EVENTS ─────────────────────────────────────────────────┐",
        header,
        sep,
    ]
    for ev in rows:
        action = ev.get("action", "")
        color = ""
        reset = ""
        if use_color:
            color = _ACTION_COLORS.get(action, "")
            reset = _RESET if color else ""

        correct = ev.get("correct")
        ok_marker = "✓" if correct is True else ("✗" if correct is False else " ")

        ts = ev.get("timestamp", "")[:19]  # trim to seconds
        lines.append(
            f"  {ts:19s}  "
            f"{ev.get('capture_id', '')[:35]:35s}  "
            f"{ev.get('score', 0.0):7.4f}  "
            f"{color}{action:20s}{reset}  "
            f"{ev.get('confidence_margin', 0.0):7.4f}  "
            f"{ev.get('n_flows', 0):6d}  "
            f"{str(ev.get('dataset', ''))[:12]:12s}  "
            f"{ok_marker:>4}"
        )
    lines.append("  └─────────────────────────────────────────────────────────────┘")
    return "\n".join(lines)


def render_open_set_dashboard(
    report: Dict[str, Any],
    use_color: bool = False,
) -> str:
    """
    Render a full open-set policy dashboard from a report dict.

    Parameters
    ----------
    report : dict
        Produced by OpenSetFirewallPolicy.dashboard_report().
    use_color : bool
        Emit ANSI terminal colours.

    Returns
    -------
    str
    """
    disclaimer = report.get("disclaimer", "[SIMULATION ONLY]")
    generated = report.get("generated_at", "")
    pi = report.get("policy_info", {})
    m = report.get("metrics", {})

    lines = [
        "=" * 72,
        "  VPN FIREWALL — OPEN-SET POLICY DASHBOARD",
        f"  ⚠  {disclaimer}",
        f"  Generated: {generated}",
        "=" * 72,
        "",
        render_status_cards(report.get("status_cards", []), use_color=use_color),
        "",
        "  ── POLICY CONFIGURATION ─────────────────────────────────────────",
        f"  Model:               {pi.get('model_id', 'N/A')}",
        f"  Policy type:         {pi.get('policy_type', 'N/A')}",
        f"  Source split:        {pi.get('source_split', 'val')}",
        f"  review_threshold:    {pi.get('review_threshold', 'N/A'):.6f}  ← {pi.get('review_threshold_basis', '')}",
        f"  block_threshold:     {pi.get('block_threshold', 'N/A'):.6f}  ← {pi.get('block_threshold_basis', '')}",
        "",
        "  Action rules:",
        *[
            f"    {k:20s}  →  {v}"
            for k, v in pi.get("actions", {}).items()
        ],
        "",
        "  ── AGGREGATE METRICS ────────────────────────────────────────────",
        f"  Total sessions:        {m.get('n_sessions', 0)}",
        f"  ✅  PASS:               {m.get('n_pass', 0)}  ({m.get('pass_rate', 0)*100:.1f}%)",
        f"    FLAG_REVIEW:        {m.get('n_flag_review', 0)}  ({m.get('flag_rate', 0)*100:.1f}%)",
        f"    SIMULATED_BLOCK:    {m.get('n_simulated_block', 0)}  ({m.get('block_rate', 0)*100:.1f}%)",
    ]

    if "vpn_detected_recall" in m:
        lines += [
            "",
            f"  VPN recall (flag+block):  {m['vpn_detected_recall']*100:.1f}%",
            f"  VPN recall (block only):  {m.get('vpn_block_recall', 0)*100:.1f}%",
        ]
    if "benign_block_fpr" in m:
        lines += [
            f"  Benign block FPR:         {m['benign_block_fpr']*100:.2f}%",
            f"  Benign review FPR:        {m.get('benign_review_fpr', 0)*100:.1f}%",
        ]

    lines += [
        "",
        render_events_table(
            report.get("recent_events", []),
            max_rows=20,
            use_color=use_color,
        ),
        "",
        "=" * 72,
    ]
    return "\n".join(lines)
