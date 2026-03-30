from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.eval.metrics import threshold_at_fpr, confusion_at_threshold


def _weighted_top_k_mean(x: pd.Series, k: int = 5) -> float:
    """
    Computes the weighted mean of the top k values in a series.
    """
    vals = np.sort(np.asarray(x, dtype=float))[::-1][:k]
    if len(vals) == 0:
        return np.nan
    
    weights = np.array([0.40, 0.25, 0.15, 0.10, 0.10])[:len(vals)]
    weights = weights / weights.sum()
    
    return float(np.sum(vals * weights))


def aggregate_to_session(
    df: pd.DataFrame,
    prob_col: str = "prob",
    label_col: str = "label",
    session_col: str = "capture_id",
    rule: str = "weighted_top5_mean",
) -> pd.DataFrame:
    """
    Aggregates flow-level predictions to the session level.

    Args:
        df: DataFrame with flow-level predictions.
        prob_col: Name of the probability column.
        label_col: Name of the label column.
        session_col: Name of the session identifier column.
        rule: The aggregation rule to use ("mean" or "weighted_top5_mean").

    Returns:
        A DataFrame with session-level predictions.
    """
    grouper = df.groupby(session_col)
    
    if rule == "mean":
        session_df = grouper[prob_col].mean().reset_index()
    elif rule == "weighted_top5_mean":
        session_df = grouper[prob_col].apply(_weighted_top_k_mean).reset_index()
    else:
        raise ValueError(f"Unknown aggregation rule: {rule}")
        
    labels = grouper[label_col].max()
    session_df = session_df.merge(labels, on=session_col)

    return session_df


def session_metrics(
    session_df: pd.DataFrame,
    prob_col: str = "prob",
    label_col: str = "label",
    low_fpr_targets: Tuple[float, ...] = (0.001, 0.005, 0.01),
) -> Dict[str, Any]:
    """
    Computes session-level evaluation metrics.

    Args:
        session_df: DataFrame with session-level predictions.
        prob_col: Name of the probability column.
        label_col: Name of the label column.
        low_fpr_targets: A tuple of low FPR values to compute recall for.

    Returns:
        A dictionary of session-level metrics.
    """
    y_true = session_df[label_col].values
    y_prob = session_df[prob_col].values

    metrics = {}

    # Session ROC AUC
    if len(np.unique(y_true)) > 1:
        metrics["session_roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["session_roc_auc"] = None

    # Recall at zero FP (block recall)
    block_threshold = threshold_at_fpr(y_true, y_prob, target_fpr=0.0)
    block_metrics = confusion_at_threshold(y_true, y_prob, block_threshold)
    metrics["block_recall_at_zero_fp"] = block_metrics["recall"]
    metrics["block_threshold"] = block_threshold

    # Recall at low FP (flagged recall)
    for fpr in low_fpr_targets:
        flag_threshold = threshold_at_fpr(y_true, y_prob, target_fpr=fpr)
        flag_metrics = confusion_at_threshold(y_true, y_prob, flag_threshold)
        metrics[f"flagged_recall_at_{fpr}_fpr"] = flag_metrics["recall"]
        metrics[f"flagged_threshold_at_{fpr}_fpr"] = flag_threshold

    return metrics
