import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

# Local imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logging import setup_logger
from src.utils.paths import load_paths
from src.pipeline.feature_pipeline import FeaturePipeline
from src.pipeline.data_preparation import load_and_prepare_data
from src.optimization.firewall_objective import compute_firewall_score

logger = setup_logger(level="INFO")

def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val, val_groups):
    """
    Optuna objective function for XGBoost tuning.
    Evaluates on validation set only.
    """
    # 1. Hyperparameters
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "booster": "gbtree",
        "tree_method": "hist",  # Faster training
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 100.0, log=True),
        "random_state": 42,
        "n_jobs": 1,
    }

    # 2. Train Model
    model = xgb.XGBClassifier(**params)
    
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        early_stopping_rounds=500,
        verbose=False
    )
    
    # 3. Predict on Val (for calibration)
    best_iteration = model.best_iteration
    p_val_raw = model.predict_proba(X_val, iteration_range=(0, best_iteration + 1))[:, 1]
    
    # 4. Calibrate (Isotonic)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_raw, y_val)
    p_val_calib = iso.transform(p_val_raw)

    # 5. Predict on Val and Evaluate Objective
    val_res = pd.DataFrame({
        "capture_id": val_groups,
        "label": y_val,
        "prob": p_val_calib
    })

    # 6. Compute block recall at FPR=0.01 on Val
    score = compute_firewall_score(val_res)
    
    return score

def run_optimization(train_datasets: list, test_dataset: str, n_trials=1000):
    paths = load_paths()
    
    # 1. Load Data
    logger.info("Loading data...")
    # Load raw data using the new helper
    df_all = load_and_prepare_data()
    
    # 2. Feature Pipeline
    logger.info("Fitting feature pipeline...")
    # Fit on LOOD TRAIN only
    train_filter = (df_all["split"] == "train") & (df_all["dataset"].isin(train_datasets))
    train_split = df_all[train_filter]
    pipeline = FeaturePipeline().fit(train_split)
    feature_cols = pipeline.model_feature_names()
    
    logger.info(f"Transforming data... ({len(feature_cols)} features)")
    df_transformed = pipeline.transform(df_all)
    
    # Add metadata back
    meta_cols = ["label", "split", "capture_id", "dataset"]
    for c in meta_cols:
        df_transformed[c] = df_all[c].values
        
    # 3. Prepare LOOD Splits
    train_df = df_transformed[(df_transformed["split"] == "train") & (df_transformed["dataset"].isin(train_datasets))]
    val_df = df_transformed[(df_transformed["split"] == "val") & (df_transformed["dataset"].isin(train_datasets))]

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values
    val_groups = val_df["capture_id"].values # For session aggregation

    logger.info(f"LOOD Train datasets: {train_datasets}, Test dataset: {test_dataset}")
    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # 4. Run Optuna
    logger.info(f"Starting Optuna optimization ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, val_groups),
        n_trials=n_trials
    )
    
    logger.info("Optimization finished.")
    logger.info(f"Best trial val score: {study.best_value}")
    logger.info("Best params:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    # After tuning, evaluate on TEST using the best parameters
    logger.info("Evaluating best model on holdout test set...")

    best_params = study.best_params
    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "logloss"
    best_params["booster"] = "gbtree"
    best_params["tree_method"] = "hist"
    best_params["n_estimators"] = 1000
    best_params["random_state"] = 42
    best_params["n_jobs"] = 1

    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=500,
        verbose=False
    )

    best_iteration = best_model.best_iteration
    p_val_raw = best_model.predict_proba(X_val, iteration_range=(0, best_iteration + 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_raw, y_val)

    test_df = df_transformed[(df_transformed["split"] == "test") & (df_transformed["dataset"] == test_dataset)]
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values
    test_groups = test_df["capture_id"].values

    p_test_raw = best_model.predict_proba(X_test, iteration_range=(0, best_iteration + 1))[:, 1]
    p_test_calib = iso.transform(p_test_raw)

    test_res = pd.DataFrame({
        "capture_id": test_groups,
        "label": y_test,
        "prob": p_test_calib
    })

    test_score = compute_firewall_score(test_res)
    logger.info(f"Final TEST Firewall Score: {test_score:.4f}")

    # Save best params
    out_path = paths.artifacts_dir / f"optuna_xgboost_LOOD_{'_'.join(train_datasets)}_test_{test_dataset}_best_params.json"
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    logger.info(f"Saved best params to {out_path}")
    
    return study.best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials per LOOD")
    args = parser.parse_args()

    paths = load_paths()

    # LOOD setups
    lood_setups = [
        (["iscx", "vnat"], "usbvpn"),
        (["iscx", "usbvpn"], "vnat"),
        (["vnat", "usbvpn"], "iscx"),
    ]

    all_best_params = {}
    for train_datasets, test_dataset in lood_setups:
        logger.info(f"Optimizing for LOOD: Train {train_datasets}, Test {test_dataset}")
        best_params = run_optimization(train_datasets=train_datasets, test_dataset=test_dataset, n_trials=args.trials)
        key = f"{'_'.join(train_datasets)}_test_{test_dataset}"
        all_best_params[key] = best_params

    # Save all best params
    out_path = paths.artifacts_dir / "optuna_xgboost_LOOD_all_best_params.json"
    with open(out_path, "w") as f:
        json.dump(all_best_params, f, indent=2)
    logger.info(f"Saved all best params to {out_path}")