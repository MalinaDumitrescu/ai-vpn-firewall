#!/usr/bin/env python3
"""
train_balanced_bagging_ensemble.py

Implements a Balanced Bagging Ensemble for VPN detection to address class imbalance.
- Splits data into Train/Val/Test (Group-aware).
- Creates multiple "bags" from TRAIN:
    - Each bag has ALL minority samples (VPN).
    - Each bag has a RANDOM SUBSET of majority samples (Non-VPN).
- Trains XGBoost, LightGBM, and CatBoost models on these bags.
- Ensembles predictions via averaging.
- Tunes thresholds on VAL for specific FPR targets (Firewall Policy).
- Evaluates on TEST (per dataset).

Usage:
    python src/models/train_balanced_bagging_ensemble.py \
        --data_path data/processed/combined_features.parquet \
        --bags_per_family 3 \
        --majority_ratio 1.0
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_fscore_support
)

# Attempt imports for GBDT libraries
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

# Local imports (assuming running from project root or src in pythonpath)
# Adjust pythonpath if needed
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logging import setup_logger
from src.eval.metrics import threshold_at_fpr
# NEW: Import existing split IO logic
from src.splits.io import load_splits

# --- Configuration & Constants ---
DEFAULT_SEED = 42
DEFAULT_BAGS = 3
DEFAULT_RATIO = 1.0  # 1:1 balance in bags
DEFAULT_FPRS = "0.001,0.005,0.01"

logger = logging.getLogger("BalancedBagging")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Balanced Bagging Ensemble")
    parser.add_argument("--data_path", type=str, required=True, help="Path to features parquet file")
    parser.add_argument("--label_col", type=str, default="label", help="Name of label column")
    parser.add_argument("--group_col", type=str, default="capture_id", help="Name of group column for splitting")
    parser.add_argument("--dataset_col", type=str, default="dataset", help="Name of dataset column (vnat/iscx)")
    parser.add_argument("--split_col", type=str, default="split", help="Name of split column if pre-split")
    
    # Split file paths (optional, for loading canonical splits)
    parser.add_argument("--train_list", type=str, default=None, help="Path to train_captures.txt")
    parser.add_argument("--val_list", type=str, default=None, help="Path to val_captures.txt")
    parser.add_argument("--test_list", type=str, default=None, help="Path to test_captures.txt")
    
    parser.add_argument("--bags_per_family", type=int, default=DEFAULT_BAGS, help="Number of bags per model type")
    parser.add_argument("--majority_ratio", type=float, default=DEFAULT_RATIO, help="Ratio of Majority:Minority in bags")
    parser.add_argument("--target_fprs", type=str, default=DEFAULT_FPRS, help="Comma-separated FPR targets")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    
    parser.add_argument("--output_dir", type=str, default="artifacts/balanced_bagging", help="Output directory")
    
    # NEW: Allow selecting specific model types
    parser.add_argument("--model_types", type=str, default="xgb,lgbm,cat", help="Comma-separated list of model types to train")

    return parser.parse_args()

def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    logger.info(f"Loading data from {path}...")
    df = pd.read_parquet(p)
    return df

def validate_schema(df: pd.DataFrame, args: argparse.Namespace):
    required = [args.label_col]
    if args.group_col in df.columns:
        logger.info(f"Found group column: {args.group_col}")
    else:
        logger.warning(f"Group column '{args.group_col}' not found. Splitting might leak info!")
    
    if args.dataset_col not in df.columns:
        logger.warning(f"Dataset column '{args.dataset_col}' not found. Will assume single dataset.")
        df[args.dataset_col] = "unknown"
        
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

def apply_canonical_splits(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Tries to load splits from text files and apply them to the dataframe.
    Returns the dataframe with a 'split' column populated/updated.
    """
    if not (args.train_list and args.val_list and args.test_list):
        return df

    logger.info("Loading canonical splits from provided text files...")
    try:
        splits = load_splits(Path(args.train_list), Path(args.val_list), Path(args.test_list))
        
        # Create a mapping: capture_id -> split_name
        cap_to_split = {}
        for split_name, caps in splits.items():
            for cap in caps:
                cap_to_split[cap] = split_name
        
        # Apply mapping
        # Ensure capture_id is string for matching
        df[args.group_col] = df[args.group_col].astype(str)
        
        # We use map, but we need to handle captures that might not be in the lists (e.g. ISCX if lists are VNAT only)
        # If a capture is not in the list, it gets NaN. We will fill NaNs later or warn.
        df[args.split_col] = df[args.group_col].map(cap_to_split)
        
        # Check coverage
        missing = df[df[args.split_col].isna()]
        if len(missing) > 0:
            logger.warning(f"{len(missing)} rows (captures: {missing[args.group_col].nunique()}) were not found in split lists.")
            # If we have mixed datasets (VNAT+ISCX) but only VNAT split lists, this is expected.
            # We might need a strategy here. For now, we leave them as NaN to be handled by fallback or filtered.
            
    except Exception as e:
        logger.error(f"Failed to load/apply splits: {e}")
        
    return df

def create_splits(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates Train/Val/Test splits.
    Priority:
    1. Canonical split files (if provided via args)
    2. Existing 'split' column in dataframe
    3. GroupShuffleSplit fallback
    """
    
    # 1. Try to apply canonical splits if files provided
    if args.train_list:
        df = apply_canonical_splits(df, args)

    # 2. Use 'split' column if it exists and covers the data
    if args.split_col in df.columns:
        # Check if we have valid splits
        valid_splits = {"train", "val", "test"}
        existing_splits = set(df[args.split_col].dropna().unique())
        
        if valid_splits.issubset(existing_splits):
            logger.info(f"Using existing '{args.split_col}' column for splits.")
            train = df[df[args.split_col] == "train"].copy()
            val = df[df[args.split_col] == "val"].copy()
            test = df[df[args.split_col] == "test"].copy()
            
            if len(train) > 0:
                return train, val, test
            else:
                logger.warning("Existing split column found but 'train' split is empty. Falling back.")
        else:
            logger.warning(f"Existing split column has incomplete values: {existing_splits}. Falling back.")

    # 3. Fallback: GroupShuffleSplit
    logger.info("Performing dynamic GroupShuffleSplit (Fallback)...")
    
    # 1. Split Train+Val vs Test (20% Test)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    groups = df[args.group_col] if args.group_col in df.columns else np.arange(len(df))
    
    train_val_idx, test_idx = next(gss_test.split(df, groups=groups))
    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    # 2. Split Train vs Val (20% of Train+Val -> Val)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed) # 0.25 * 0.8 = 0.2 total
    groups_tv = train_val_df[args.group_col] if args.group_col in train_val_df.columns else np.arange(len(train_val_df))
    
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=groups_tv))
    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()
    
    logger.info(f"Splits created: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

def create_balanced_bags(
    train_df: pd.DataFrame, 
    label_col: str, 
    n_bags: int, 
    ratio: float, 
    seed: int
) -> List[pd.DataFrame]:
    """
    Creates balanced bags from TRAIN set.
    Each bag: All Minority + Random Sample of Majority (size = n_min * ratio)
    """
    minority = train_df[train_df[label_col] == 1]
    majority = train_df[train_df[label_col] == 0]
    
    n_min = len(minority)
    n_maj_sample = int(n_min * ratio)
    
    logger.info(f"Creating {n_bags} bags. Minority count: {n_min}. Majority sample size: {n_maj_sample}")
    
    bags = []
    rng = np.random.default_rng(seed)
    
    for i in range(n_bags):
        # Sample majority without replacement for this bag
        # We use a different seed offset for each bag to ensure diversity
        maj_sample = majority.sample(n=n_maj_sample, replace=False, random_state=seed + i)
        
        bag = pd.concat([minority, maj_sample]).sample(frac=1.0, random_state=seed + i) # Shuffle
        bags.append(bag)
        
    return bags

def train_model(
    model_type: str, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    seed: int
):
    """
    Trains a single model (xgb, lgbm, or catboost).
    """
    if model_type == "xgb":
        if xgb is None: raise ImportError("XGBoost not installed")
        model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            early_stopping_rounds=50,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model
        
    elif model_type == "lgbm":
        if lgb is None: raise ImportError("LightGBM not installed")
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            n_jobs=1,
            verbose=-1
        )
        model.fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)], 
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        return model
        
    elif model_type == "cat":
        if cb is None: raise ImportError("CatBoost not installed")
        model = cb.CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_seed=seed,
            thread_count=1,
            verbose=False,
            allow_writing_files=False
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        return model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_feature_cols(df: pd.DataFrame, args: argparse.Namespace) -> List[str]:
    # Exclude metadata columns
    exclude = {args.label_col, args.group_col, args.dataset_col, args.split_col, "flow_id", "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "sample_weight"}
    features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return features

def evaluate_preds(y_true, y_prob, thresholds: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes metrics for a set of predictions given thresholds.
    """
    res = {}
    
    # Threshold-free
    try:
        res["auc"] = roc_auc_score(y_true, y_prob)
        res["pr_auc"] = average_precision_score(y_true, y_prob)
    except:
        res["auc"] = None
        res["pr_auc"] = None
        
    # Threshold-based
    for name, thr in thresholds.items():
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        res[name] = {
            "threshold": thr,
            "precision": p,
            "recall": r,
            "f1": f1,
            "fpr": fpr,
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        }
        
    return res

def run_balanced_bagging(
    df: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "capture_id",
    dataset_col: str = "dataset",
    split_col: str = "split",
    bags_per_family: int = 3,
    majority_ratio: float = 1.0,
    target_fprs: str = "0.001,0.005,0.01",
    seed: int = 42,
    output_dir: str = "artifacts/balanced_bagging",
    train_list: Optional[str] = None,
    val_list: Optional[str] = None,
    test_list: Optional[str] = None,
    model_types: Optional[List[str]] = None, # NEW: Allow filtering model types
):
    """
    Programmatic entry point for running the balanced bagging pipeline.
    """
    # Create a dummy args object to reuse existing functions
    args = argparse.Namespace(
        data_path="IN_MEMORY",
        label_col=label_col,
        group_col=group_col,
        dataset_col=dataset_col,
        split_col=split_col,
        bags_per_family=bags_per_family,
        majority_ratio=majority_ratio,
        target_fprs=target_fprs,
        seed=seed,
        output_dir=output_dir,
        train_list=train_list,
        val_list=val_list,
        test_list=test_list
    )
    
    setup_logger()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    validate_schema(df, args)
    
    # B) Create Splits
    train_df, val_df, test_df = create_splits(df, args)
    
    feature_cols = get_feature_cols(train_df, args)
    logger.info(f"Training on {len(feature_cols)} features.")
    
    X_val = val_df[feature_cols]
    y_val = val_df[args.label_col]
    
    # C) Build Bags & D) Train Models
    # Determine model types to run
    available_types = []
    if xgb: available_types.append("xgb")
    if lgb: available_types.append("lgbm")
    if cb: available_types.append("cat")
    
    if model_types is None:
        selected_types = available_types
    else:
        selected_types = [m for m in model_types if m in available_types]
        if len(selected_types) < len(model_types):
            logger.warning(f"Some requested models not available/installed. Requested: {model_types}, Available: {selected_types}")

    if not selected_types:
        logger.error("No valid model types selected. Exiting.")
        return
        
    logger.info(f"Training {args.bags_per_family} bags for each of: {selected_types}")
    
    trained_models = []
    
    for m_type in selected_types:
        bags = create_balanced_bags(train_df, args.label_col, args.bags_per_family, args.majority_ratio, args.seed)
        
        for i, bag in enumerate(bags):
            logger.info(f"Training {m_type} bag {i+1}/{args.bags_per_family}...")
            X_bag = bag[feature_cols]
            y_bag = bag[args.label_col]
            
            model_seed = args.seed + (i * 100)
            model = train_model(m_type, X_bag, y_bag, X_val, y_val, model_seed)
            
            trained_models.append({
                "type": m_type,
                "bag_idx": i,
                "model": model
            })
            
            joblib.dump(model, out_dir / f"model_{m_type}_bag{i}.pkl")

    # E) Produce Predictions (Ensemble)
    logger.info("Generating predictions...")
    
    def get_ensemble_proba(X):
        preds = []
        for tm in trained_models:
            m = tm["model"]
            if tm["type"] == "xgb":
                p = m.predict_proba(X)[:, 1]
            elif tm["type"] == "lgbm":
                p = m.predict_proba(X)[:, 1]
            elif tm["type"] == "cat":
                p = m.predict_proba(X)[:, 1]
            preds.append(p)
        if not preds: return np.zeros(len(X))
        return np.mean(preds, axis=0)

    # Predict on ALL splits
    logger.info("Predicting on Train/Val/Test...")
    
    # We need to predict on the original full dataframes to save them
    # But we only have train_df, val_df, test_df slices.
    # Let's compute probabilities for each slice.
    
    train_probs = get_ensemble_proba(train_df[feature_cols])
    val_probs = get_ensemble_proba(val_df[feature_cols])
    test_probs = get_ensemble_proba(test_df[feature_cols])
    
    # G) Threshold Tuning (on Val)
    logger.info("Tuning thresholds on Validation set...")
    target_fprs_list = [float(x) for x in args.target_fprs.split(",")]
    thresholds = {"default": 0.5}
    
    for fpr in target_fprs_list:
        thr = threshold_at_fpr(y_val.values, val_probs, fpr)
        thresholds[f"fpr_{fpr}"] = thr
        logger.info(f"  FPR {fpr}: Threshold = {thr:.4f}")
        
    # F) Metrics & Evaluation
    results = {}
    results["val"] = evaluate_preds(y_val.values, val_probs, thresholds)
    results["test_overall"] = evaluate_preds(test_df[args.label_col].values, test_probs, thresholds)
    
    datasets = test_df[args.dataset_col].unique()
    for ds in datasets:
        mask = test_df[args.dataset_col] == ds
        if mask.sum() == 0: continue
        
        y_ds = test_df[args.label_col][mask]
        p_ds = test_probs[mask]
        results[f"test_{ds}"] = evaluate_preds(y_ds.values, p_ds, thresholds)
        
    # H) Artifacts
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        
    # Save predictions CSV (ALL splits)
    # We concatenate the splits back together
    
    def prepare_pred_df(df_split, probs, split_name):
        out = df_split[[args.group_col, args.dataset_col, args.label_col]].copy()
        out["prob"] = probs
        out["split"] = split_name
        return out
        
    pred_df_train = prepare_pred_df(train_df, train_probs, "train")
    pred_df_val = prepare_pred_df(val_df, val_probs, "val")
    pred_df_test = prepare_pred_df(test_df, test_probs, "test")
    
    full_pred_df = pd.concat([pred_df_train, pred_df_val, pred_df_test], ignore_index=True)
    
    # Rename to predictions.csv (generic)
    full_pred_df.to_csv(out_dir / "predictions.csv", index=False)
    
    logger.info(f"Done. Results saved to {out_dir}")
    return results

def main():
    args = parse_args()
    df = load_data(args.data_path)
    
    # Parse model types from CLI
    model_types = args.model_types.split(",") if args.model_types else None
    
    run_balanced_bagging(
        df,
        label_col=args.label_col,
        group_col=args.group_col,
        dataset_col=args.dataset_col,
        split_col=args.split_col,
        bags_per_family=args.bags_per_family,
        majority_ratio=args.majority_ratio,
        target_fprs=args.target_fprs,
        seed=args.seed,
        output_dir=args.output_dir,
        train_list=args.train_list,
        val_list=args.val_list,
        test_list=args.test_list,
        model_types=model_types
    )

if __name__ == "__main__":
    main()
