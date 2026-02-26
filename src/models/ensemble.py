from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import yaml
import json
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

from src.eval.metrics import binary_metrics, select_policy_thresholds, _policy_key_from_fpr


@dataclass(frozen=True)
class EnsembleResult:
    preds_path: Path
    metrics_path: Path
    metrics: Dict[str, Any]


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def find_optimal_weights(df_val: pd.DataFrame, model_cols: list[str], label_col: str) -> np.ndarray:
    """Find optimal weights for a weighted average ensemble."""
    y_true = df_val[label_col].to_numpy()
    preds = df_val[model_cols].to_numpy()

    def objective(weights):
        weights = np.abs(weights) # Ensure non-negative
        weights /= weights.sum() # Normalize to sum to 1
        
        weighted_preds = np.dot(preds, weights)
        return -roc_auc_score(y_true, weighted_preds)

    # Initial guess: equal weights
    initial_weights = np.ones(len(model_cols)) / len(model_cols)
    
    # Constraint: weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    
    # Bounds: weights must be between 0 and 1
    bounds = [(0, 1)] * len(model_cols)

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    return optimal_weights / optimal_weights.sum() # Re-normalize for safety


def create_ensemble(
    *,
    paths,
    ensemble_yaml: Path,
    df: Optional[pd.DataFrame] = None,
) -> EnsembleResult:
    cfg = _load_yaml(ensemble_yaml)

    data_cfg = cfg.get("data") or {}
    label_col = str(data_cfg.get("label_col", "label"))
    split_col = str(data_cfg.get("split_col", "split"))

    training_cfg = cfg.get("training") or {}
    split_names = training_cfg.get("split_names") or {}
    val_name = str(split_names.get("val", "val"))

    policy_fprs = training_cfg.get("policy_fprs", [0.001, 0.01, 0.05])
    policy_fprs = [float(x) for x in policy_fprs]
    firewall_fpr = float(training_cfg.get("firewall_policy", 0.001))
    policy_mode = str(training_cfg.get("policy_mode", "max_recall_under_fpr"))

    model_cols = cfg.get("models", [])
    if not model_cols:
        raise ValueError("ensemble.yaml must contain a 'models' list.")

    out_cfg = cfg.get("outputs") or {}
    preds_path = (
        paths.repo_root / str(out_cfg.get("preds_path", "artifacts/ensemble/preds.parquet"))
    ).resolve()
    metrics_path = (
        paths.repo_root / str(out_cfg.get("metrics_path", "artifacts/ensemble/metrics.json"))
    ).resolve()

    _ensure_parent(preds_path)
    _ensure_parent(metrics_path)

    # Load predictions from individual models
    df_list = []
    for model_name in model_cols:
        p = paths.artifacts_dir / model_name / "preds.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Prediction file not found for model '{model_name}': {p}")
        
        df_model = pd.read_parquet(p)
        # Keep only essential columns to avoid conflicts
        df_list.append(df_model[["flow_id", f"p_{model_name}"]])

    # Merge predictions
    df_preds = df_list[0]
    for i in range(1, len(df_list)):
        df_preds = pd.merge(df_preds, df_list[i], on="flow_id", how="inner")

    # Add back label and split info from one of the dataframes
    base_df = pd.read_parquet(paths.artifacts_dir / model_cols[0] / "preds.parquet")
    df_preds = pd.merge(df_preds, base_df[["flow_id", "label", "split", "capture_id"]], on="flow_id", how="left")

    # Find optimal weights on validation set
    df_val = df_preds[df_preds[split_col] == val_name].copy()
    pred_cols = [f"p_{m}" for m in model_cols]
    optimal_weights = find_optimal_weights(df_val, pred_cols, label_col)

    print("Optimal Weights:")
    for name, weight in zip(model_cols, optimal_weights):
        print(f"  - {name}: {weight:.4f}")

    # Apply weights to create ensemble predictions
    df_preds["p_ensemble"] = np.dot(df_preds[pred_cols].to_numpy(), optimal_weights)
    df_preds["p_raw"] = df_preds["p_ensemble"] # for compatibility with downstream tools

    # --- Metrics ---
    # IMPORTANT: Re-select the validation set AFTER adding p_ensemble
    df_val_final = df_preds[df_preds[split_col] == val_name].copy()
    
    val_df_for_sel = pd.DataFrame({
        split_col: df_val_final[split_col],
        label_col: df_val_final[label_col],
        "p": df_val_final["p_ensemble"]
    })
    
    all_fprs = sorted(list(set(policy_fprs + [firewall_fpr])))
    
    selected_thresholds = select_policy_thresholds(
        val_df_for_sel,
        label_col=label_col,
        prob_col="p",
        split_col=split_col,
        split_name=val_name,
        policy_fprs=tuple(all_fprs),
        policy_mode=policy_mode
    )

    firewall_policy_name = _policy_key_from_fpr(firewall_fpr)
    firewall_thr = selected_thresholds[firewall_policy_name]

    metrics = {
        "models": model_cols,
        "weights": dict(zip(model_cols, optimal_weights.tolist())),
        "policy_thresholds": {k: {"threshold": v} for k, v in selected_thresholds.items()},
        "firewall_policy": {
            "chosen": firewall_policy_name,
            "fpr_target": firewall_fpr,
            "threshold": firewall_thr,
            "mode": policy_mode,
        },
        "splits": {}
    }

    for split_name in df_preds[split_col].unique():
        df_split = df_preds[df_preds[split_col] == split_name]
        y = df_split[label_col].to_numpy()
        p = df_split["p_ensemble"].to_numpy()
        metrics["splits"][split_name] = binary_metrics(y, p, threshold=0.5, fixed_policy_thresholds=selected_thresholds)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    df_preds.to_parquet(preds_path, index=False)

    return EnsembleResult(
        preds_path=preds_path,
        metrics_path=metrics_path,
        metrics=metrics,
    )
