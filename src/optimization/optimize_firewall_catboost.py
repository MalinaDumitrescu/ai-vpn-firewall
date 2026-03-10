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
import catboost as cb
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
    Optuna objective function for CatBoost tuning.
    """
    # 1. Hyperparameters
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 100.0, log=True),
        "random_seed": 42,
        "thread_count": 1,
        "verbose": False,
        "allow_writing_files": False,
        "early_stopping_rounds": 500
    }

    # 2. Train Model
    train_pool = cb.Pool(X_train, y_train)
    val_pool = cb.Pool(X_val, y_val)
    
    model = cb.CatBoostClassifier(**params)
    
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True
    )
    
    # 3. Predict (Raw Scores)
    p_val_raw = model.predict_proba(val_pool)[:, 1]
    
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

def run_optimization(n_trials=1000):
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
    out_path = paths.artifacts_dir / "optuna_catboost_best_params.json"
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    logger.info(f"Saved best params to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    args = parser.parse_args()

    run_optimization(n_trials=args.trials)
