#!/usr/bin/env python3
"""Shared policy utilities for model policy packaging workflows.

This module is read-only with respect to model artifacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Reused from src/deployment/decision_engine.py:_wt5_agg and src/eval/bootstrap.py:_wt5_agg
WT5_VERIFIED = True
WT5_SOURCE = "src/deployment/decision_engine.py::_wt5_agg"

_KEEP_COLS = [
    "split",
    "dataset",
    "label",
    "flow_id",
    "capture_id",
    "session_id",
    "prob_raw",
    "prob_iso",
    "prob_platt",
]


def _resolve_prediction_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        for name in ("predictions.csv", "predictions.parquet", "test_predictions.parquet", "val_predictions.parquet"):
            cand = p / name
            if cand.exists():
                return cand
        preds = sorted(list(p.glob("*predictions*.csv")) + list(p.glob("*predictions*.parquet")))
        if preds:
            return preds[0]
    raise FileNotFoundError(f"No prediction file found at: {p}")


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Load .csv/.parquet predictions and preserve standard policy columns when present."""
    p = _resolve_prediction_path(path)
    if p.suffix.lower() == ".csv":
        cols = pd.read_csv(p, nrows=0).columns.tolist()
        keep = [c for c in _KEEP_COLS if c in cols]
        return pd.read_csv(p, usecols=keep)
    if p.suffix.lower() == ".parquet":
        cols = pd.read_parquet(p).columns.tolist()
        keep = [c for c in _KEEP_COLS if c in cols]
        return pd.read_parquet(p, columns=keep)
    raise ValueError(f"Unsupported prediction format: {p}")


def get_available_probability_columns(df: pd.DataFrame) -> list[str]:
    """Return available probability columns among prob_raw/prob_iso/prob_platt."""
    return [c for c in ["prob_raw", "prob_iso", "prob_platt"] if c in df.columns]


def choose_probability_column(
    df: pd.DataFrame, preferred_order: Iterable[str] = ("prob_raw", "prob_iso", "prob_platt")
) -> str:
    """Choose the first available probability column in preferred order, with safe fallback."""
    available = set(get_available_probability_columns(df))
    for c in preferred_order:
        if c in available:
            return c
    if available:
        return sorted(available)[0]
    raise ValueError("No supported probability column found in DataFrame")


def infer_session_group_column(df: pd.DataFrame) -> str | None:
    """Prefer session_id, otherwise capture_id, else None."""
    if "session_id" in df.columns:
        return "session_id"
    if "capture_id" in df.columns:
        return "capture_id"
    return None


def aggregate_p80(scores: np.ndarray | pd.Series) -> float:
    """80th percentile aggregation."""
    x = np.asarray(scores, dtype=float)
    if len(x) == 0:
        return 0.0
    return float(np.percentile(x, 80))


def aggregate_wt5(scores: np.ndarray | pd.Series) -> float:
    """Verified wt5 formula reused exactly from the project implementation."""
    x = np.asarray(scores, dtype=float)
    vals = np.sort(x)[::-1][:5]
    if len(vals) == 0:
        return 0.0
    w = np.array([0.40, 0.25, 0.15, 0.10, 0.10])[: len(vals)]
    w = w / w.sum()
    return float(np.sum(vals * w))


def aggregate_wt5_unverified(scores: np.ndarray | pd.Series) -> float:
    """Fallback helper for workflows where verified wt5 is unavailable."""
    # This intentionally mirrors the same top-5 weighted structure but is marked unverified by name.
    x = np.asarray(scores, dtype=float)
    vals = np.sort(x)[::-1][:5]
    if len(vals) == 0:
        return 0.0
    w = np.array([0.40, 0.25, 0.15, 0.10, 0.10])[: len(vals)]
    w = w / w.sum()
    return float(np.sum(vals * w))


def build_session_table(
    df: pd.DataFrame,
    probability_column: str,
    aggregation,
    session_col: str,
) -> pd.DataFrame:
    """Aggregate flows to sessions and return session table.

    Output columns: session_id, label, dataset, n_flows, session_score.
    """
    if session_col is None or session_col not in df.columns:
        raise ValueError("Session grouping column is not available")
    if probability_column not in df.columns:
        raise ValueError(f"Missing probability column: {probability_column}")
    if "label" not in df.columns:
        raise ValueError("Missing label column")

    out_rows = []
    for sid, g in df.groupby(session_col):
        labels = g["label"].astype(int)
        score = aggregation(g[probability_column].to_numpy(dtype=float))
        ds = "mixed"
        if "dataset" in g.columns:
            uniq = g["dataset"].dropna().astype(str).unique().tolist()
            ds = uniq[0] if len(uniq) == 1 else "mixed"
        out_rows.append(
            {
                "session_id": sid,
                "label": int(labels.max()),
                "dataset": ds,
                "n_flows": int(len(g)),
                "session_score": float(score),
            }
        )
    return pd.DataFrame(out_rows)


def compute_strict_threshold(val_scores: np.ndarray | pd.Series, val_labels: np.ndarray | pd.Series) -> float:
    """Validation-only strict threshold with benign FPR=0 and max recall."""
    s = np.asarray(val_scores, dtype=float)
    y = np.asarray(val_labels, dtype=int)
    benign = s[y == 0]
    if len(benign) == 0:
        # If no benign in validation, return maximal score threshold.
        return float(np.nextafter(np.max(s), np.inf)) if len(s) else 1.0
    # Smallest threshold that keeps all benign below threshold (zero FP): nextafter(max_benign,+inf)
    return float(np.nextafter(np.max(benign), np.inf))


def compute_balanced_threshold(
    val_scores: np.ndarray | pd.Series,
    val_labels: np.ndarray | pd.Series,
    max_fpr: float = 0.01,
) -> float:
    """Validation-only threshold maximizing recall subject to benign FPR <= max_fpr."""
    s = np.asarray(val_scores, dtype=float)
    y = np.asarray(val_labels, dtype=int)
    if len(s) == 0:
        return 1.0

    benign_n = max(int((y == 0).sum()), 1)
    vpn_n = max(int((y == 1).sum()), 1)

    # Candidate thresholds include strict edge and all score cut points.
    candidates = [float(np.nextafter(np.max(s), np.inf))] + sorted(set(float(x) for x in s), reverse=True)

    best_thr = candidates[0]
    best_rec = -1.0
    best_fpr = 1.0

    for thr in candidates:
        pred = s >= thr
        fp = int(((pred == 1) & (y == 0)).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        fpr = fp / benign_n
        rec = tp / vpn_n
        if fpr <= max_fpr:
            # Primary: maximize recall. Secondary: choose smallest threshold among ties.
            if (rec > best_rec) or (np.isclose(rec, best_rec) and thr < best_thr):
                best_rec = rec
                best_fpr = fpr
                best_thr = thr

    if best_rec < 0:
        # No feasible threshold under constraint (rare with degenerate labels) -> strict fallback.
        return compute_strict_threshold(s, y)
    return float(best_thr)


def evaluate_threshold(
    test_scores: np.ndarray | pd.Series,
    test_labels: np.ndarray | pd.Series,
    threshold: float,
) -> dict:
    """Evaluate threshold and return recall/FPR/confusion/support counts."""
    s = np.asarray(test_scores, dtype=float)
    y = np.asarray(test_labels, dtype=int)
    pred = s >= float(threshold)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    pos_n = max(int((y == 1).sum()), 1)
    neg_n = max(int((y == 0).sum()), 1)

    return {
        "threshold": float(threshold),
        "recall": float(tp / pos_n),
        "fpr": float(fp / neg_n),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "n_total": int(len(y)),
        "n_vpn": int((y == 1).sum()),
        "n_benign": int((y == 0).sum()),
    }


def compute_session_auc(test_scores: np.ndarray | pd.Series, test_labels: np.ndarray | pd.Series) -> float | None:
    """Return ROC-AUC if both classes exist, else None."""
    s = np.asarray(test_scores, dtype=float)
    y = np.asarray(test_labels, dtype=int)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def final_action(strict_trigger: bool, balanced_trigger: bool) -> str:
    """Map strict/balanced triggers into final policy action."""
    if strict_trigger:
        return "BLOCK"
    if balanced_trigger and not strict_trigger:
        return "FLAG_REVIEW"
    return "PASS"


def check_flow_capture_overlap(df: pd.DataFrame) -> dict:
    """Check flow_id/capture_id overlaps across train/val/test splits where available."""
    out: dict[str, object] = {
        "has_split_col": "split" in df.columns,
        "flow_overlap_train_val": None,
        "flow_overlap_train_test": None,
        "flow_overlap_val_test": None,
        "capture_overlap_train_val": None,
        "capture_overlap_train_test": None,
        "capture_overlap_val_test": None,
    }
    if "split" not in df.columns:
        return out

    def _set_for(col: str, a: str, b: str) -> int | None:
        if col not in df.columns:
            return None
        sa = set(df.loc[df["split"] == a, col].dropna().astype(str).unique())
        sb = set(df.loc[df["split"] == b, col].dropna().astype(str).unique())
        return int(len(sa & sb))

    out["flow_overlap_train_val"] = _set_for("flow_id", "train", "val")
    out["flow_overlap_train_test"] = _set_for("flow_id", "train", "test")
    out["flow_overlap_val_test"] = _set_for("flow_id", "val", "test")
    out["capture_overlap_train_val"] = _set_for("capture_id", "train", "val")
    out["capture_overlap_train_test"] = _set_for("capture_id", "train", "test")
    out["capture_overlap_val_test"] = _set_for("capture_id", "val", "test")
    return out


