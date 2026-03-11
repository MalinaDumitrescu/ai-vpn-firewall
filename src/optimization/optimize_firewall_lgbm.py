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
import lightgbm as lgb
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
    Optuna objective function for LightGBM tuning.
    """
    # 1. Hyperparameters
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": 1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 100.0, log=True),
    }

    # 2. Train Model
    model = lgb.LGBMClassifier(**params)
    
    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(0)
    ]
    
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        eval_metric="binary_logloss",
        callbacks=callbacks
    )
    
    # 3. Predict (Raw Scores)
    # LGBM returns raw probabilities by default
    p_val_raw = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
    
    # 4. Calibrate (Isotonic)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_raw, y_val)
    p_val_calib = iso.transform(p_val_raw)
    
    # 5. Session Aggregation
    val_res = pd.DataFrame({
        "capture_id": val_groups,
        "label": y_val,
        "prob": p_val_calib
    })

    # 6. Compute Score
    score = compute_firewall_score(val_res)
    
    return score

def run_optimization(n_trials=100):
    paths = load_paths()
    
    # 1. Load Data
    logger.info("Loading data...")
    df_all = load_and_prepare_data()
    
    # 2. Feature Pipeline
    logger.info("Fitting feature pipeline...")
    train_split = df_all[df_all["split"] == "train"]
    pipeline = FeaturePipeline().fit(train_split)
    feature_cols = pipeline.model_feature_names()
    
    logger.info(f"Transforming data... ({len(feature_cols)} features)")
    df_transformed = pipeline.transform(df_all)
    
    # Add metadata back
    meta_cols = ["label", "split", "capture_id"]
    for c in meta_cols:
        df_transformed[c] = df_all[c].values
        
    # 3. Prepare Splits
    train_df = df_transformed[df_transformed["split"] == "train"]
    val_df = df_transformed[df_transformed["split"] == "val"]
    
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values
    val_groups = val_df["capture_id"].values 
    
    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    
    # 4. Run Optuna
    logger.info(f"Starting Optuna optimization ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, val_groups),
        n_trials=n_trials
    )
    
    logger.info("Optimization finished.")
    logger.info(f"Best trial score: {study.best_value}")
    logger.info("Best params:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")
        
    # Save best params
    out_path = paths.artifacts_dir / "optuna_lgbm_best_params.json"
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    logger.info(f"Saved best params to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    args = parser.parse_args()

    run_optimization(n_trials=args.trials)
