#!/usr/bin/env python
"""
LOOD (Leave-One-Out-Dataset) Training Runner

Trains unified models on combined datasets with rotating test sets:
1. Train on ISCX + USBVPN, Test on VNAT
2. Train on VNAT + USBVPN, Test on ISCX  
3. Train on VNAT + ISCX, Test on USBVPN

This increases VPN training signal significantly, especially for datasets with 
limited VPN samples (VNAT, USBVPN).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import json
import numpy as np
from dataclasses import asdict

from src.utils.paths import load_paths
from src.utils.logging import setup_logger
from src.pipeline.feature_pipeline import FeaturePipeline
from src.pipeline.data_preparation import load_and_prepare_data
from src.pipeline.artifacts import default_feature_artifacts
from src.features.extract import load_feature_config, feature_config_hash_text
from src.eval.lood import LOODEvaluator, LOODFold
from src.eval.metrics import select_policy_thresholds, binary_metrics

from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import yaml

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

logger = setup_logger(level="INFO")


def train_lood_fold_lgbm(
    fold: LOODFold,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    pipeline: FeaturePipeline,
    lgbm_config: Dict,
    output_dir: Path,
) -> Dict:
    """
    Train LightGBM model for a single LOOD fold.

    Args:
        fold: LOODFold specification
        df_train: Combined train+val data
        df_test: Test data
        pipeline: Fitted FeaturePipeline
        lgbm_config: LightGBM configuration dict
        output_dir: Output directory for results

    Returns:
        Results dict with metrics and metadata
    """
    if lgb is None:
        raise ImportError("lightgbm not installed")

    logger.info(f"\n{'='*70}")
    logger.info(f"Training LOOD Fold: {fold.fold_name}")
    logger.info(f"{'='*70}")

    # Get model features
    feature_cols = pipeline.model_feature_names()
    logger.info(f"Model features: {len(feature_cols)}")

    # Transform data
    logger.info("Transforming data through pipeline...")
    X_full = pipeline.transform(df_train)
    X_test_full = pipeline.transform(df_test)

    # Split train/val
    train_mask = df_train["split"] == "train"
    val_mask = df_train["split"] == "val"

    X_train = X_full[train_mask][feature_cols].values.astype(np.float32)
    y_train = df_train[train_mask]["label"].values.astype(np.int32)

    X_val = X_full[val_mask][feature_cols].values.astype(np.float32)
    y_val = df_train[val_mask]["label"].values.astype(np.int32)

    X_test = X_test_full[feature_cols].values.astype(np.float32)
    y_test = df_test["label"].values.astype(np.int32)

    logger.info(f"Training set: {X_train.shape}, VPN: {(y_train == 1).sum()}")
    logger.info(f"Validation set: {X_val.shape}, VPN: {(y_val == 1).sum()}")
    logger.info(f"Test set: {X_test.shape}, VPN: {(y_test == 1).sum()}")

    # Prepare data for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Extract training parameters
    params = lgbm_config.get("params", {})
    
    # Add class weighting
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    if n_pos > 0:
        params['scale_pos_weight'] = n_neg / n_pos
        logger.info(f"Applied scale_pos_weight: {params['scale_pos_weight']:.2f}")

    num_leaves = params.get("num_leaves", 31)
    learning_rate = params.get("learning_rate", 0.1)
    num_rounds = lgbm_config.get("num_rounds", 1000)

    logger.info(f"LightGBM params: num_leaves={num_leaves}, lr={learning_rate}, rounds={num_rounds}")

    # Train model
    logger.info("Training LightGBM model...")
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=num_rounds,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.log_evaluation(50),
            lgb.early_stopping(50),
        ],
    )

    # Predict
    logger.info("Generating predictions...")
    p_train = booster.predict(X_train, num_iteration=booster.best_iteration)
    p_val = booster.predict(X_val, num_iteration=booster.best_iteration)
    p_test = booster.predict(X_test, num_iteration=booster.best_iteration)

    # Evaluate
    logger.info("Computing metrics...")
    
    train_auc = roc_auc_score(y_train, p_train)
    train_ap = average_precision_score(y_train, p_train)
    
    val_auc = roc_auc_score(y_val, p_val)
    val_ap = average_precision_score(y_val, p_val)
    
    test_auc = roc_auc_score(y_test, p_test)
    test_ap = average_precision_score(y_test, p_test)

    logger.info(f"Train AUC: {train_auc:.4f}, AP: {train_ap:.4f}")
    logger.info(f"Val AUC:   {val_auc:.4f}, AP: {val_ap:.4f}")
    logger.info(f"Test AUC:  {test_auc:.4f}, AP: {test_ap:.4f}")

    # Build results
    results = {
        "fold_id": fold.fold_id,
        "fold_name": fold.fold_name,
        "train_datasets": fold.train_datasets,
        "test_dataset": fold.test_dataset,
        "metrics": {
            "train_auc": float(train_auc),
            "train_ap": float(train_ap),
            "val_auc": float(val_auc),
            "val_ap": float(val_ap),
            "test_auc": float(test_auc),
            "test_ap": float(test_ap),
        },
        "data_shapes": {
            "train_samples": int(X_train.shape[0]),
            "train_vpn": int((y_train == 1).sum()),
            "val_samples": int(X_val.shape[0]),
            "val_vpn": int((y_val == 1).sum()),
            "test_samples": int(X_test.shape[0]),
            "test_vpn": int((y_test == 1).sum()),
        },
    }

    # Save results
    fold_dir = output_dir / fold.fold_id
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = fold_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")

    # Save predictions
    preds_df = pd.DataFrame({
        "fold_id": [fold.fold_id] * len(y_test),
        "label": y_test,
        "p_raw": p_test,
    })
    preds_path = fold_dir / "predictions.parquet"
    preds_df.to_parquet(preds_path, index=False)
    logger.info(f"Saved predictions: {preds_path}")

    # Save model
    model_path = fold_dir / "model.txt"
    booster.save_model(str(model_path))
    logger.info(f"Saved model: {model_path}")

    return results


def summarize_lood_results(all_results: List[Dict]) -> Dict:
    """
    Summarize results across all LOOD folds.

    Args:
        all_results: List of result dicts from each fold

    Returns:
        Summary dict with aggregated metrics
    """
    logger.info("\n" + "="*70)
    logger.info("LOOD Results Summary")
    logger.info("="*70)

    # Extract metrics
    test_aucs = [r["metrics"]["test_auc"] for r in all_results]
    test_aps = [r["metrics"]["test_ap"] for r in all_results]

    summary = {
        "num_folds": len(all_results),
        "test_auc_mean": float(np.mean(test_aucs)),
        "test_auc_std": float(np.std(test_aucs)),
        "test_auc_min": float(np.min(test_aucs)),
        "test_auc_max": float(np.max(test_aucs)),
        "test_ap_mean": float(np.mean(test_aps)),
        "test_ap_std": float(np.std(test_aps)),
        "test_ap_min": float(np.min(test_aps)),
        "test_ap_max": float(np.max(test_aps)),
        "fold_results": all_results,
    }

    # Print summary
    logger.info(f"\nTest AUC: {summary['test_auc_mean']:.4f} ± {summary['test_auc_std']:.4f}")
    logger.info(f"Test AP:  {summary['test_ap_mean']:.4f} ± {summary['test_ap_std']:.4f}")

    logger.info("\nPer-fold results:")
    for result in all_results:
        logger.info(f"  {result['fold_id']}: AUC={result['metrics']['test_auc']:.4f}, AP={result['metrics']['test_ap']:.4f}")

    return summary


def main():
    """Main LOOD training pipeline."""
    paths = load_paths()
    paths.ensure_dirs()

    # Output directory
    lood_dir = paths.artifacts_dir / "lood_training"
    lood_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {lood_dir}")

    # Load combined data
    logger.info("Loading combined dataset...")
    df_all = load_and_prepare_data(vnat_only=False)
    logger.info(f"Loaded {len(df_all)} flows from {len(df_all['dataset'].unique())} datasets")

    # Create LOOD folds
    evaluator = LOODEvaluator()
    folds = evaluator.create_folds(["vnat", "iscx", "usbvpn"])
    evaluator.print_lood_summary()

    # Prepare data for all folds
    lood_data = evaluator.prepare_all_lood_data(df_all)

    # Load configs
    features_yaml = paths.configs_dir / "features.yaml"
    lgbm_yaml = paths.configs_dir / "lgbm.yaml"

    with open(lgbm_yaml) as f:
        lgbm_config = yaml.safe_load(f)

    feature_hash = feature_config_hash_text(features_yaml)
    logger.info(f"Feature config hash: {feature_hash[:16]}...")

    # Train model for each fold
    all_results = []

    for fold in folds:
        df_train, df_test = lood_data[fold.fold_id]

        # Fit pipeline on train+val data
        logger.info(f"Fitting feature pipeline for {fold.fold_id}...")
        pipeline = FeaturePipeline().fit(df_train)

        # Train LOOD model
        try:
            result = train_lood_fold_lgbm(
                fold=fold,
                df_train=df_train,
                df_test=df_test,
                pipeline=pipeline,
                lgbm_config=lgbm_config,
                output_dir=lood_dir,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error training fold {fold.fold_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summarize results
    if all_results:
        summary = summarize_lood_results(all_results)

        # Save summary
        summary_path = lood_dir / "lood_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"\nSaved summary: {summary_path}")

        logger.info("\n" + "="*70)
        logger.info("LOOD Training Complete")
        logger.info("="*70)
        logger.info(f"Trained {summary['num_folds']} LOOD models")
        logger.info(f"Macro-averaged Test AUC: {summary['test_auc_mean']:.4f}")
        logger.info(f"Results saved to: {lood_dir}")
    else:
        logger.error("No results to summarize. Check errors above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
