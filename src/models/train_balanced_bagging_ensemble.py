#!/usr/bin/env python3
"""
train_balanced_bagging_ensemble.py

Balanced Bagging Ensemble for VPN detection.

Method:
- Use a FIXED pre-existing split column (train/val/test), or apply canonical split lists.
- Do NOT create new splits dynamically.
- Create multiple bags from TRAIN:
    - each bag contains ALL minority samples (VPN)
    - each bag contains a RANDOM SUBSET of majority samples (Non-VPN)
- Train XGBoost, LightGBM, and CatBoost on those bags.
- Average probabilities within each family.
- Combine family probabilities with configurable weights.
- Fit isotonic calibration on VAL.
- Fit logistic calibration (Platt scaling) on VAL.
- Tune thresholds on calibrated VAL for target FPRs for each calibration mode.
- Evaluate on TEST (overall + per dataset) for each calibration mode.

This file is intentionally strict:
- no fallback GroupShuffleSplit
- no silent dropping of uncovered rows
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
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

# Local imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logging import setup_logger
from src.eval.metrics import threshold_at_fpr
from src.splits.io import load_splits


DEFAULT_SEED = 42
DEFAULT_BAGS = 3
DEFAULT_RATIO = 1.0
DEFAULT_FPRS = "0.001,0.005,0.01"

logger = logging.getLogger("BalancedBagging")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Balanced Bagging Ensemble")
    parser.add_argument("--data_path", type=str, required=True, help="Path to features parquet file")
    parser.add_argument("--label_col", type=str, default="label", help="Name of label column")
    parser.add_argument("--group_col", type=str, default="capture_id", help="Name of group column")
    parser.add_argument("--dataset_col", type=str, default="dataset", help="Name of dataset column")
    parser.add_argument("--split_col", type=str, default="split", help="Name of split column")

    parser.add_argument("--train_list", type=str, default=None, help="Path to train capture list")
    parser.add_argument("--val_list", type=str, default=None, help="Path to val capture list")
    parser.add_argument("--test_list", type=str, default=None, help="Path to test capture list")

    parser.add_argument("--bags_per_family", type=int, default=DEFAULT_BAGS, help="Number of bags per model family")
    parser.add_argument("--majority_ratio", type=float, default=DEFAULT_RATIO, help="Majority:minority ratio per bag")
    parser.add_argument("--target_fprs", type=str, default=DEFAULT_FPRS, help="Comma-separated FPR targets")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="artifacts/balanced_bagging", help="Output directory")
    parser.add_argument("--model_types", type=str, default="xgb,lgbm,cat", help="Comma-separated model types")

    # Optional family weights
    parser.add_argument("--weight_xgb", type=float, default=1.0, help="Weight for XGBoost family")
    parser.add_argument("--weight_lgbm", type=float, default=1.0, help="Weight for LightGBM family")
    parser.add_argument("--weight_cat", type=float, default=1.0, help="Weight for CatBoost family")

    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    logger.info(f"Loading data from {path}...")
    return pd.read_parquet(p)


def validate_schema(df: pd.DataFrame, args: argparse.Namespace):
    required = [args.label_col, args.group_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if args.dataset_col not in df.columns:
        logger.warning(f"Dataset column '{args.dataset_col}' not found. Filling with 'unknown'.")
        df[args.dataset_col] = "unknown"

    if args.split_col not in df.columns and not (args.train_list and args.val_list and args.test_list):
        raise ValueError(
            f"Missing split column '{args.split_col}' and no canonical split lists were provided."
        )


def apply_canonical_splits(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Apply canonical capture split lists to the dataframe.
    Fails if any capture is uncovered.
    """
    if not (args.train_list and args.val_list and args.test_list):
        return df

    logger.info("Applying canonical splits from provided text files...")
    splits = load_splits(Path(args.train_list), Path(args.val_list), Path(args.test_list))

    cap_to_split = {}
    for split_name, caps in splits.items():
        for cid in caps:
            cap_to_split[str(cid)] = split_name

    out = df.copy()
    out[args.group_col] = out[args.group_col].astype(str)
    out[args.split_col] = out[args.group_col].map(cap_to_split)

    missing_caps = sorted(out.loc[out[args.split_col].isna(), args.group_col].unique().tolist())
    if missing_caps:
        raise ValueError(
            f"Canonical split lists do not cover all captures in this dataframe. "
            f"Missing captures: {len(missing_caps)}. Examples: {missing_caps[:10]}"
        )

    return out


def create_splits(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    STRICT split loading:
    1. Apply canonical split lists if provided
    2. Require a complete split column containing train/val/test
    3. No dynamic fallback
    """
    if args.train_list and args.val_list and args.test_list:
        df = apply_canonical_splits(df, args)

    if args.split_col not in df.columns:
        raise ValueError(f"Missing split column '{args.split_col}' after canonical split application.")

    out = df.copy()
    out[args.split_col] = out[args.split_col].astype(str)

    valid_splits = {"train", "val", "test"}
    existing_splits = set(out[args.split_col].dropna().unique())

    if not valid_splits.issubset(existing_splits):
        raise ValueError(
            f"Split column must contain train/val/test. Found: {sorted(existing_splits)}"
        )

    if out[args.split_col].isna().any():
        missing_caps = sorted(out.loc[out[args.split_col].isna(), args.group_col].astype(str).unique().tolist())
        raise ValueError(
            f"Some rows have no split assigned. Missing captures: {len(missing_caps)}. "
            f"Examples: {missing_caps[:10]}"
        )

    bad_values = sorted(set(out[args.split_col].unique()) - valid_splits)
    if bad_values:
        raise ValueError(f"Unexpected split values found: {bad_values}")

    train = out[out[args.split_col] == "train"].copy()
    val = out[out[args.split_col] == "val"].copy()
    test = out[out[args.split_col] == "test"].copy()

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            f"Split sizes invalid: train={len(train)}, val={len(val)}, test={len(test)}"
        )

    logger.info(f"Using fixed splits: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def create_balanced_bags(
    train_df: pd.DataFrame,
    label_col: str,
    n_bags: int,
    ratio: float,
    seed: int,
) -> List[pd.DataFrame]:
    """
    Create balanced bags from TRAIN.
    Each bag = ALL minority + RANDOM subset of majority.
    """
    minority = train_df[train_df[label_col] == 1].copy()
    majority = train_df[train_df[label_col] == 0].copy()

    n_min = len(minority)
    n_maj = len(majority)

    if n_min == 0:
        raise ValueError("No minority samples found in train split.")
    if n_maj == 0:
        raise ValueError("No majority samples found in train split.")
    if ratio <= 0:
        raise ValueError(f"majority_ratio must be > 0. Got {ratio}")

    n_maj_sample = min(int(round(n_min * ratio)), n_maj)
    if n_maj_sample == 0:
        raise ValueError("Majority sample size computed as 0. Increase ratio or check train data.")

    logger.info(
        f"Creating {n_bags} bags. Minority={n_min}, Majority={n_maj}, "
        f"Sampled majority per bag={n_maj_sample}"
    )

    bags = []
    for i in range(n_bags):
        maj_sample = majority.sample(
            n=n_maj_sample,
            replace=False,
            random_state=seed + i,
        )
        bag = pd.concat([minority, maj_sample], ignore_index=True)
        bag = bag.sample(frac=1.0, random_state=seed + i).reset_index(drop=True)
        bags.append(bag)

    return bags


def train_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
    model_params: Optional[Dict[str, Any]] = None,
):
    if model_type == "xgb":
        if xgb is None:
            raise ImportError("XGBoost not installed")

        # Default params
        params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "n_jobs": 1,
            "early_stopping_rounds": 50,
            "eval_metric": "logloss",
        }
        
        # Override with provided params if any
        if model_params:
            logger.info(f"Overriding XGBoost params: {model_params}")
            params.update(model_params)
            # Ensure random_state is preserved if not explicitly overridden
            if "random_state" not in model_params:
                params["random_state"] = seed

        # For XGBoost >= 1.6 (and 2.0+), early_stopping_rounds is an init parameter.
        # We pass everything in params to the constructor.
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
        return model

    elif model_type == "lgbm":
        if lgb is None:
            raise ImportError("LightGBM not installed")

        params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": seed,
            "n_jobs": 1,
            "verbose": -1,
        }
        
        if model_params:
            logger.info(f"Overriding LightGBM params: {model_params}")
            params.update(model_params)
            if "random_state" not in model_params:
                params["random_state"] = seed

        model = lgb.LGBMClassifier(**params)
        
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        return model

    elif model_type == "cat":
        if cb is None:
            raise ImportError("CatBoost not installed")

        params = {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": seed,
            "thread_count": 1,
            "verbose": False,
            "allow_writing_files": False,
        }
        
        if model_params:
            logger.info(f"Overriding CatBoost params: {model_params}")
            params.update(model_params)
            if "random_seed" not in model_params:
                params["random_seed"] = seed

        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        return model

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_feature_cols(df: pd.DataFrame, args: argparse.Namespace) -> List[str]:
    """
    Fallback feature discovery only.
    Prefer passing explicit feature_cols into run_balanced_bagging().
    """
    exclude = {
        args.label_col,
        args.group_col,
        args.dataset_col,
        args.split_col,
        "flow_id",
        "timestamp",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
        "sample_weight",
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def evaluate_preds(y_true, y_prob, thresholds: Dict[str, float]) -> Dict[str, Any]:
    res = {}

    try:
        res["auc"] = roc_auc_score(y_true, y_prob)
        res["pr_auc"] = average_precision_score(y_true, y_prob)
    except Exception:
        res["auc"] = None
        res["pr_auc"] = None

    for name, thr in thresholds.items():
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        p, r, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        res[name] = {
            "threshold": float(thr),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "fpr": float(fpr),
            "confusion": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

    return res


def _normalize_family_weights(
    selected_types: List[str],
    weight_xgb: float,
    weight_lgbm: float,
    weight_cat: float,
) -> Dict[str, float]:
    raw = {
        "xgb": float(weight_xgb),
        "lgbm": float(weight_lgbm),
        "cat": float(weight_cat),
    }

    active = {k: raw[k] for k in selected_types}
    if any(v < 0 for v in active.values()):
        raise ValueError(f"Family weights must be >= 0. Got: {active}")

    total = sum(active.values())
    if total <= 0:
        raise ValueError(f"Sum of active family weights must be > 0. Got: {active}")

    return {k: v / total for k, v in active.items()}


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
    model_types: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None,
    weight_xgb: float = 1.0,
    weight_lgbm: float = 1.0,
    weight_cat: float = 1.0,
    xgb_params: Optional[Dict[str, Any]] = None,
    lgbm_params: Optional[Dict[str, Any]] = None,
    cat_params: Optional[Dict[str, Any]] = None,
):
    """
    Programmatic entry point.
    Requires fixed split labels or canonical split lists.
    """
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
        test_list=test_list,
        weight_xgb=weight_xgb,
        weight_lgbm=weight_lgbm,
        weight_cat=weight_cat,
    )

    setup_logger()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_schema(df, args)

    train_df, val_df, test_df = create_splits(df, args)

    if feature_cols is None:
        feature_cols = get_feature_cols(train_df, args)

    if not feature_cols:
        raise ValueError("No feature columns available for training.")

    missing_train = [c for c in feature_cols if c not in train_df.columns]
    missing_val = [c for c in feature_cols if c not in val_df.columns]
    missing_test = [c for c in feature_cols if c not in test_df.columns]
    if missing_train or missing_val or missing_test:
        raise ValueError(
            f"Feature column mismatch.\n"
            f"Missing in train: {missing_train[:10]}\n"
            f"Missing in val: {missing_val[:10]}\n"
            f"Missing in test: {missing_test[:10]}"
        )

    logger.info(f"Training on {len(feature_cols)} features.")
    logger.info(f"First 20 features: {feature_cols[:20]}")

    X_val = val_df[feature_cols]
    y_val = val_df[label_col].astype(int)

    available_types = []
    if xgb is not None:
        available_types.append("xgb")
    if lgb is not None:
        available_types.append("lgbm")
    if cb is not None:
        available_types.append("cat")

    if model_types is None:
        selected_types = available_types
    else:
        selected_types = [m for m in model_types if m in available_types]
        if len(selected_types) < len(model_types):
            logger.warning(
                f"Some requested models are unavailable. "
                f"Requested={model_types}, available={selected_types}"
            )

    if not selected_types:
        raise ValueError("No valid model types selected.")

    family_weights = _normalize_family_weights(
        selected_types=selected_types,
        weight_xgb=weight_xgb,
        weight_lgbm=weight_lgbm,
        weight_cat=weight_cat,
    )

    logger.info(f"Training {bags_per_family} bags for each of: {selected_types}")
    logger.info(f"Normalized family weights: {family_weights}")

    trained_models = []

    for m_type in selected_types:
        bags = create_balanced_bags(
            train_df=train_df,
            label_col=label_col,
            n_bags=bags_per_family,
            ratio=majority_ratio,
            seed=seed,
        )

        # Select params for this model type
        current_params = None
        if m_type == "xgb":
            current_params = xgb_params
        elif m_type == "lgbm":
            current_params = lgbm_params
        elif m_type == "cat":
            current_params = cat_params

        for i, bag in enumerate(bags):
            logger.info(f"Training {m_type} bag {i + 1}/{bags_per_family}...")
            X_bag = bag[feature_cols]
            y_bag = bag[label_col].astype(int)

            model_seed = seed + (i * 100) + (10000 * selected_types.index(m_type))
            model = train_model(m_type, X_bag, y_bag, X_val, y_val, model_seed, model_params=current_params)

            trained_models.append(
                {
                    "type": m_type,
                    "bag_idx": i,
                    "model": model,
                }
            )

            joblib.dump(model, out_dir / f"model_{m_type}_bag{i}.pkl")

    logger.info("Generating family-level predictions...")

    def get_family_probas(X: pd.DataFrame) -> Dict[str, Optional[np.ndarray]]:
        fam_preds: Dict[str, List[np.ndarray]] = {
            "xgb": [],
            "lgbm": [],
            "cat": [],
        }

        for tm in trained_models:
            p = tm["model"].predict_proba(X)[:, 1]
            fam_preds[tm["type"]].append(p)

        fam_means: Dict[str, Optional[np.ndarray]] = {}
        for fam, preds in fam_preds.items():
            if preds:
                fam_means[fam] = np.mean(preds, axis=0)
            else:
                fam_means[fam] = None

        return fam_means

    def combine_family_probs(fam_probs: Dict[str, Optional[np.ndarray]]) -> np.ndarray:
        pieces = []
        n_ref = None

        for fam in ["xgb", "lgbm", "cat"]:
            p = fam_probs.get(fam)
            if p is None:
                continue
            if n_ref is None:
                n_ref = len(p)
            pieces.append(family_weights.get(fam, 0.0) * p)

        if not pieces:
            if n_ref is None:
                return np.array([], dtype=float)
            return np.zeros(n_ref, dtype=float)

        return np.sum(pieces, axis=0)

    train_family_probs = get_family_probas(train_df[feature_cols])
    val_family_probs = get_family_probas(val_df[feature_cols])
    test_family_probs = get_family_probas(test_df[feature_cols])

    train_probs_raw = combine_family_probs(train_family_probs)
    val_probs_raw = combine_family_probs(val_family_probs)
    test_probs_raw = combine_family_probs(test_family_probs)

    # --- Calibration ---
    logger.info("Fitting calibrators on validation set...")

    # 1. Isotonic
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(val_probs_raw, y_val.values)

    # 2. Platt (Logistic)
    # Reshape for sklearn
    X_val_calib = val_probs_raw.reshape(-1, 1)
    platt = LogisticRegression(random_state=seed, solver='lbfgs')
    platt.fit(X_val_calib, y_val.values)

    # Transform all splits
    train_probs_iso = isotonic.transform(train_probs_raw)
    val_probs_iso = isotonic.transform(val_probs_raw)
    test_probs_iso = isotonic.transform(test_probs_raw)

    train_probs_platt = platt.predict_proba(train_probs_raw.reshape(-1, 1))[:, 1]
    val_probs_platt = platt.predict_proba(val_probs_raw.reshape(-1, 1))[:, 1]
    test_probs_platt = platt.predict_proba(test_probs_raw.reshape(-1, 1))[:, 1]

    # --- Threshold Tuning ---
    logger.info("Tuning thresholds on calibrated validation set...")
    target_fprs_list = [float(x) for x in target_fprs.split(",")]

    def tune_thresholds(y_true, y_prob, mode_name):
        thrs = {"default": 0.5}
        for fpr in target_fprs_list:
            thr = threshold_at_fpr(y_true, y_prob, fpr)
            thrs[f"fpr_{fpr}"] = float(thr)
            logger.info(f"  [{mode_name}] FPR {fpr}: threshold = {thr:.4f}")
        return thrs

    thresholds_raw = tune_thresholds(y_val.values, val_probs_raw, "raw")
    thresholds_iso = tune_thresholds(y_val.values, val_probs_iso, "isotonic")
    thresholds_platt = tune_thresholds(y_val.values, val_probs_platt, "platt")

    # --- Evaluation ---
    results: Dict[str, Any] = {
        "ensemble_weights": family_weights,
        "raw": {},
        "isotonic": {},
        "platt": {}
    }

    def run_eval(y_true, y_prob, thrs, section, split_key):
        if section not in results:
            results[section] = {}
        results[section][split_key] = evaluate_preds(y_true, y_prob, thrs)

    # Validation
    run_eval(y_val.values, val_probs_raw, thresholds_raw, "raw", "val")
    run_eval(y_val.values, val_probs_iso, thresholds_iso, "isotonic", "val")
    run_eval(y_val.values, val_probs_platt, thresholds_platt, "platt", "val")

    # Test Overall
    y_test = test_df[label_col].astype(int).values
    run_eval(y_test, test_probs_raw, thresholds_raw, "raw", "test_overall")
    run_eval(y_test, test_probs_iso, thresholds_iso, "isotonic", "test_overall")
    run_eval(y_test, test_probs_platt, thresholds_platt, "platt", "test_overall")

    # Test Per Dataset
    for ds in test_df[dataset_col].astype(str).unique():
        mask = test_df[dataset_col].astype(str) == ds
        if mask.sum() == 0:
            continue
        y_ds = test_df.loc[mask, label_col].astype(int).values
        
        p_raw_ds = test_probs_raw[mask]
        p_iso_ds = test_probs_iso[mask]
        p_platt_ds = test_probs_platt[mask]

        run_eval(y_ds, p_raw_ds, thresholds_raw, "raw", f"test_{ds}")
        run_eval(y_ds, p_iso_ds, thresholds_iso, "isotonic", f"test_{ds}")
        run_eval(y_ds, p_platt_ds, thresholds_platt, "platt", f"test_{ds}")

    # Save metrics
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x,
        )

    # Save calibrators
    joblib.dump(isotonic, out_dir / "isotonic_calibrator.pkl")
    joblib.dump(platt, out_dir / "platt_calibrator.pkl")

    # Save predictions
    def prepare_pred_df(
        df_split: pd.DataFrame,
        fam_probs: Dict[str, Optional[np.ndarray]],
        p_raw: np.ndarray,
        p_iso: np.ndarray,
        p_platt: np.ndarray,
        split_name: str,
    ) -> pd.DataFrame:
        out = df_split[[group_col, dataset_col, label_col]].copy()
        if "flow_id" in df_split.columns:
            out["flow_id"] = df_split["flow_id"].values

        out["p_xgb_raw"] = fam_probs["xgb"] if fam_probs["xgb"] is not None else np.nan
        out["p_lgbm_raw"] = fam_probs["lgbm"] if fam_probs["lgbm"] is not None else np.nan
        out["p_cat_raw"] = fam_probs["cat"] if fam_probs["cat"] is not None else np.nan

        out["prob_raw"] = p_raw
        out["prob_iso"] = p_iso
        out["prob_platt"] = p_platt
        # Backward compatibility alias
        out["prob"] = p_iso
        
        out["split"] = split_name
        return out

    pred_df_train = prepare_pred_df(train_df, train_family_probs, train_probs_raw, train_probs_iso, train_probs_platt, "train")
    pred_df_val = prepare_pred_df(val_df, val_family_probs, val_probs_raw, val_probs_iso, val_probs_platt, "val")
    pred_df_test = prepare_pred_df(test_df, test_family_probs, test_probs_raw, test_probs_iso, test_probs_platt, "test")

    full_pred_df = pd.concat([pred_df_train, pred_df_val, pred_df_test], ignore_index=True)
    full_pred_df.to_csv(out_dir / "predictions.csv", index=False)

    logger.info(f"Done. Results saved to {out_dir}")
    return results


def main():
    args = parse_args()
    df = load_data(args.data_path)

    model_types = args.model_types.split(",") if args.model_types else None

    run_balanced_bagging(
        df=df,
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
        model_types=model_types,
        feature_cols=None,
        weight_xgb=args.weight_xgb,
        weight_lgbm=args.weight_lgbm,
        weight_cat=args.weight_cat,
    )


if __name__ == "__main__":
    main()