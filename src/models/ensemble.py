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
isotonic import IsotonicRegression

from src.eval.metrics import binary_metrics, select_policy_thresholds, _policy_key_from_fpr, threshold_at_fpr


@dataclass(frozen=True)
class EnsembleResult:
    preds_path: Path
    metrics_path: Path
    metrics: Dict[str, Any]


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def find_optimal_weights_firewall(
    df_val: pd.DataFrame, 
    model_cols: List[str], 
    label_col: str,
    target_fpr: float = 0.001
) -> np.ndarray:
    """
    Find optimal weights for a weighted average ensemble maximizing RECALL at target FPR.
    """
    y_true = df_val[label_col].to_numpy()
    preds = df_val[model_cols].to_numpy()

    # Pre-compute negatives mask for speed
    neg_mask = (y_true == 0)
    pos_mask = (y_true == 1)
    
    # If no positives or negatives, fallback to equal weights
    if not np.any(neg_mask) or not np.any(pos_mask):
        return np.ones(len(model_cols)) / len(model_cols)

    def objective(weights):
        # Ensure non-negative and sum to 1
        w = np.abs(weights)
        if w.sum() == 0:
            w = np.ones_like(w)
        w /= w.sum()
        
        # 1. Compute ensemble score
        p = np.dot(preds, w)
        
        # 2. Calibrate on validation (Isotonic) - simplified for optimization loop
        # Full isotonic is slow inside optimization loop.
        # Approximation: Just use rank-based thresholding on raw ensemble score.
        # Since Isotonic is monotonic, maximizing recall on raw score at fixed FPR 
        # is equivalent to maximizing recall on calibrated score at fixed FPR.
        
        # 3. Choose threshold t that achieves FPR <= target_fpr on validation NEGATIVES
        neg_scores = p[neg_mask]
        
        # Threshold is the (1-target_fpr) quantile of negative scores
        # We want p > t to be positive.
        t = np.quantile(neg_scores, 1.0 - target_fpr)
        
        # 4. Compute recall/TPR at that threshold
        pos_scores = p[pos_mask]
        
        tp = np.sum(pos_scores >= t)
        recall = tp / len(pos_scores)
        
        # Optional: Add a secondary term for monitor threshold (0.01) to avoid pathological solutions
        # t_monitor = np.quantile(neg_scores, 1.0 - 0.01)
        # tp_monitor = np.sum(pos_scores >= t_monitor)
        # recall_monitor = tp_monitor / len(pos_scores)
        
        # objective = -recall - 0.1 * recall_monitor
        return -recall

    # Initial guess: equal weights
    initial_weights = np.ones(len(model_cols)) / len(model_cols)
    
    # Constraint: weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - np.sum(w)})
    
    # Bounds: weights must be between 0 and 1
    bounds = [(0, 1)] * len(model_cols)

    # Use SLSQP
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = np.abs(result.x)
    return optimal_weights / optimal_weights.sum()


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
    dfs = []
    for m in model_names:
        p = paths.artifacts_dir / m / "preds.parquet"
        if not p.exists():
            if m == "xgboost": p = paths.artifacts_dir / "xgb" / "preds.parquet"
            elif m == "lightgbm": p = paths.artifacts_dir / "lgbm" / "preds.parquet"
            
            if not p.exists():
                raise FileNotFoundError(f"Prediction file not found for model '{m}': {p}")
        
        d = pd.read_parquet(p)
        if "flow_id" not in d.columns:
            raise ValueError(f"Model {m} preds missing flow_id")
        
        # Use p_raw (calibrated from model training) or specific column
        col_name = "p_raw"
        if col_name not in d.columns:
            if f"p_{m}" in d.columns: col_name = f"p_{m}"
            elif "p_xgb" in d.columns: col_name = "p_xgb"
            elif "p_lgbm" in d.columns: col_name = "p_lgbm"
            elif "p_catboost" in d.columns: col_name = "p_catboost"
            else:
                raise ValueError(f"Could not find probability column in {p}")
        
        subset = d[["flow_id", col_name]].rename(columns={col_name: f"p_{m}"})
        dfs.append(subset)

    # Merge all predictions
    df_ens = dfs[0]
    for i in range(1, len(dfs)):
        df_ens = pd.merge(df_ens, dfs[i], on="flow_id", how="inner")

    # Merge back metadata
    base_df = pd.read_parquet(paths.artifacts_dir / model_names[0] / "preds.parquet")
    meta_cols = ["flow_id", label_col, split_col, "capture_id", "dataset"]
    meta_cols = [c for c in meta_cols if c in base_df.columns]
    
    df_ens = pd.merge(df_ens, base_df[meta_cols], on="flow_id", how="left")

    # Ensure dataset column exists
    if "dataset" not in df_ens.columns:
        # Try to infer or fail? For now, if missing, fill with "unknown"
        print("WARNING: 'dataset' column missing in ensemble inputs. Filling with 'unknown'.")
        df_ens["dataset"] = "unknown"

    # 1. Find Optimal Weights on VAL (Firewall Objective)
    val_mask = df_ens[split_col] == val_name
    if not val_mask.any():
        raise ValueError(f"No validation samples found for split '{val_name}'")
    
    df_val_for_weights = df_ens[val_mask].copy()
    pred_cols = [f"p_{m}" for m in model_names]
    
    print(f"Optimizing ensemble weights for Firewall Recall at FPR={firewall_fpr}...")
    weights = find_optimal_weights_firewall(df_val_for_weights, pred_cols, label_col, target_fpr=firewall_fpr)
    weight_dict = dict(zip(model_names, weights.tolist()))
    print("Ensemble Weights:", weight_dict)

    # 2. Compute Weighted Average (Ensemble Score)
    probs_matrix = df_ens[pred_cols].to_numpy()
    df_ens["p_ensemble"] = np.dot(probs_matrix, weights)

    # 3. Calibrate the Ensemble Score (Isotonic per Dataset)
    print("Calibrating Ensemble (Isotonic per dataset)...")
    
    # We need to fit calibrators on VAL set
    val_df_for_calib = df_ens[val_mask]
    
    calibrators = {}
    datasets = df_ens["dataset"].unique()
    
    # Global calibrator (fallback)
    global_iso = IsotonicRegression(out_of_bounds="clip")
    global_iso.fit(val_df_for_calib["p_ensemble"], val_df_for_calib[label_col])
    calibrators["global"] = global_iso
    
    # Per-dataset calibrators
    for ds in datasets:
        ds_val = val_df_for_calib[val_df_for_calib["dataset"] == ds]
        if len(ds_val) > 50: # Only fit if enough samples
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(ds_val["p_ensemble"], ds_val[label_col])
            calibrators[ds] = iso
        else:
            print(f"Warning: Not enough validation samples for dataset '{ds}' to fit calibrator. Using global.")
            calibrators[ds] = global_iso

    # Apply calibration
    df_ens["p_calib"] = 0.0
    
    # Apply global first as default
    df_ens["p_calib"] = calibrators["global"].transform(df_ens["p_ensemble"])
    
    # Then apply specific calibrators where applicable
    for ds in datasets:
        if ds in calibrators and ds != "global":
            mask = df_ens["dataset"] == ds
            if mask.any():
                df_ens.loc[mask, "p_calib"] = calibrators[ds].transform(df_ens.loc[mask, "p_ensemble"])

    # 4. Select Thresholds per Dataset (on VAL)
    # Get a fresh slice of the validation data which now includes 'p_calib'
    val_df_calibrated = df_ens[val_mask]

    thresholds_map = {}
    
    # Helper to compute thresholds for a subset
    def compute_subset_thresholds(subset_df):
        y = subset_df[label_col].to_numpy()
        p = subset_df["p_calib"].to_numpy()
        
        # Compute for all policy FPRs + firewall FPR
        all_fprs = sorted(list(set(policy_fprs + [firewall_fpr])))
        
        thrs = {}
        for fpr in all_fprs:
            t = threshold_at_fpr(y, p, fpr)
            thrs[_policy_key_from_fpr(fpr)] = t
        return thrs

    # Global thresholds
    thresholds_map["global"] = compute_subset_thresholds(val_df_calibrated)
    
    # Per-dataset thresholds
    for ds in datasets:
        ds_val = val_df_calibrated[val_df_calibrated["dataset"] == ds]
        if len(ds_val) > 0:
            thresholds_map[ds] = compute_subset_thresholds(ds_val)

    # 5. Compute Metrics for All Splits
    # We report metrics using the PER-DATASET thresholds if available, else global
    split_metrics = {}
    
    for sp in df_ens[split_col].unique():
        sp_df = df_ens[df_ens[split_col] == sp]
        
        # We want to report metrics broken down by dataset as well
        sp_metrics = {}
        
        # Overall metrics for this split (using global thresholds)
        y_sp = sp_df[label_col].to_numpy()
        p_sp = sp_df["p_calib"].to_numpy()
        
        # Use firewall threshold from global set for the main "firewall_policy" entry
        global_firewall_thr = thresholds_map["global"][_policy_key_from_fpr(firewall_fpr)]
        
        sp_metrics["overall"] = binary_metrics(
            y_sp, 
            p_sp, 
            threshold=global_firewall_thr,
            fixed_policy_thresholds=thresholds_map["global"]
        )
        
        # Per-dataset metrics
        for ds in sp_df["dataset"].unique():
            ds_df = sp_df[sp_df["dataset"] == ds]
            if len(ds_df) == 0: continue
            
            y_ds = ds_df[label_col].to_numpy()
            p_ds = ds_df["p_calib"].to_numpy()
            
            # Use per-dataset thresholds if available, else global
            thrs = thresholds_map.get(ds, thresholds_map["global"])
            ds_firewall_thr = thrs[_policy_key_from_fpr(firewall_fpr)]
            
            sp_metrics[ds] = binary_metrics(
                y_ds,
                p_ds,
                threshold=ds_firewall_thr,
                fixed_policy_thresholds=thrs
            )
            
        split_metrics[str(sp)] = sp_metrics

    metrics = {
        "models": model_names,
        "weights": weight_dict,
        "thresholds_map": thresholds_map,
        "firewall_policy_target_fpr": firewall_fpr        "policy_thresholds": {k: {"threshold": v} for k, v in selected_thresholds.items()},
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
