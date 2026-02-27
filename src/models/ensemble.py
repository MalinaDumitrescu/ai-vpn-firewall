from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import yaml
import json
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

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


def find_optimal_weights(df_val: pd.DataFrame, model_cols: List[str], label_col: str) -> np.ndarray:
    """Find optimal weights for a weighted average ensemble."""
    y_true = df_val[label_col].to_numpy()
    preds = df_val[model_cols].to_numpy()

    def objective(weights):
        # Ensure non-negative and sum to 1
        w = np.abs(weights)
        if w.sum() == 0:
            w = np.ones_like(w)
        w /= w.sum()
        
        weighted_preds = np.dot(preds, w)
        return -roc_auc_score(y_true, weighted_preds)

    # Initial guess: equal weights
    initial_weights = np.ones(len(model_cols)) / len(model_cols)
    
    # Constraint: weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    
    # Bounds: weights must be between 0 and 1
    bounds = [(0, 1)] * len(model_cols)

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = np.abs(result.x)
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

    # The YAML should list model names (e.g. ["xgb", "lgbm", "catboost"])
    model_names = cfg.get("models", [])
    if not model_names:
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
    # We expect each model to have saved a preds.parquet with "p_raw" (calibrated prob)
    # or we can use the model-specific column if we know it.
    # Let's assume standard "p_raw" is the calibrated output from train_*.py
    
    dfs = []
    for m in model_names:
        # Construct path: artifacts/{model_name}/preds.parquet
        # This assumes a standard directory structure
        p = paths.artifacts_dir / m / "preds.parquet"
        if not p.exists():
            # Fallback for "xgboost" vs "xgb" naming if needed, or raise error
            # Try "xgb" if "xgboost" was passed
            if m == "xgboost": p = paths.artifacts_dir / "xgb" / "preds.parquet"
            elif m == "lightgbm": p = paths.artifacts_dir / "lgbm" / "preds.parquet"
            
            if not p.exists():
                raise FileNotFoundError(f"Prediction file not found for model '{m}': {p}")
        
        d = pd.read_parquet(p)
        # We need flow_id to merge
        if "flow_id" not in d.columns:
            raise ValueError(f"Model {m} preds missing flow_id")
        
        # Rename p_raw to p_{model}
        # If p_raw doesn't exist, try p_{model} or p_xgb etc.
        col_name = "p_raw"
        if col_name not in d.columns:
            # Try specific names
            if f"p_{m}" in d.columns: col_name = f"p_{m}"
            elif "p_xgb" in d.columns: col_name = "p_xgb"
            elif "p_lgbm" in d.columns: col_name = "p_lgbm"
            elif "p_catboost" in d.columns: col_name = "p_catboost"
            else:
                raise ValueError(f"Could not find probability column in {p}")
        
        # Keep only ID and prob
        subset = d[["flow_id", col_name]].rename(columns={col_name: f"p_{m}"})
        dfs.append(subset)

    # Merge all predictions
    df_ens = dfs[0]
    for i in range(1, len(dfs)):
        df_ens = pd.merge(df_ens, dfs[i], on="flow_id", how="inner")

    # Merge back metadata (label, split, capture_id) from the first model's full file
    # (Assuming all models were trained on same splits/flows)
    base_df = pd.read_parquet(paths.artifacts_dir / model_names[0] / "preds.parquet")
    # Check if base_df has the metadata
    meta_cols = ["flow_id", label_col, split_col, "capture_id"]
    # Filter to available columns
    meta_cols = [c for c in meta_cols if c in base_df.columns]
    
    df_ens = pd.merge(df_ens, base_df[meta_cols], on="flow_id", how="left")

    # 1. Find Optimal Weights on VAL
    val_mask = df_ens[split_col] == val_name
    if not val_mask.any():
        raise ValueError(f"No validation samples found for split '{val_name}'")
    
    df_val = df_ens[val_mask].copy()
    pred_cols = [f"p_{m}" for m in model_names]
    
    weights = find_optimal_weights(df_val, pred_cols, label_col)
    weight_dict = dict(zip(model_names, weights.tolist()))
    print("Ensemble Weights:", weight_dict)

    # 2. Compute Weighted Average (Ensemble Score)
    # p_ensemble = w1*p1 + w2*p2 ...
    # We do this via dot product
    probs_matrix = df_ens[pred_cols].to_numpy()
    df_ens["p_ensemble"] = np.dot(probs_matrix, weights)

    # 3. Calibrate the Ensemble Score
    # We fit a calibrator on the ensemble score using VAL set
    print("Calibrating Ensemble...")
    calib_X = df_ens.loc[val_mask, "p_ensemble"].values.reshape(-1, 1)
    calib_y = df_ens.loc[val_mask, label_col].values
    
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(calib_X, calib_y)
    
    # Apply to all
    all_X = df_ens["p_ensemble"].values.reshape(-1, 1)
    df_ens["p_calib"] = calibrator.predict_proba(all_X)[:, 1]
    
    # 4. Select Thresholds on Calibrated Ensemble Score (VAL)
    val_df_calib = pd.DataFrame({
        split_col: [val_name] * len(calib_y),
        label_col: calib_y,
        "p": df_ens.loc[val_mask, "p_calib"].values
    })
    
    all_fprs = sorted(list(set(policy_fprs + [firewall_fpr])))
    selected_thresholds = select_policy_thresholds(
        val_df_calib,
        label_col=label_col,
        prob_col="p",
        split_col=split_col,
        split_name=val_name,
        policy_fprs=tuple(all_fprs),
        policy_mode=policy_mode
    )
    
    firewall_policy_name = _policy_key_from_fpr(firewall_fpr)
    firewall_thr = selected_thresholds[firewall_policy_name]

    # 5. Compute Metrics for All Splits
    split_metrics = {}
    for sp in df_ens[split_col].unique():
        mask = df_ens[split_col] == sp
        y_sp = df_ens.loc[mask, label_col].to_numpy()
        p_sp = df_ens.loc[mask, "p_calib"].to_numpy()
        
        split_metrics[str(sp)] = binary_metrics(
            y_sp, 
            p_sp, 
            threshold=0.5, 
            fixed_policy_thresholds=selected_thresholds
        )

    metrics = {
        "models": model_names,
        "weights": weight_dict,
        "policy_thresholds": {k: {"threshold": v} for k, v in selected_thresholds.items()},
        "firewall_policy": {
            "chosen": firewall_policy_name,
            "fpr_target": firewall_fpr,
            "threshold": firewall_thr,
            "mode": policy_mode,
        },
        "splits": split_metrics
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    df_ens.to_parquet(preds_path, index=False)

    return EnsembleResult(
        preds_path=preds_path,
        metrics_path=metrics_path,
        metrics=metrics,
    )
