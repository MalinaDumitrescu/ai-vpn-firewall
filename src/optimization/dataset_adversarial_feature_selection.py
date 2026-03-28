import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.logging import setup_logger
from src.utils.paths import load_paths
from src.pipeline.feature_pipeline import FeaturePipeline
from src.pipeline.data_preparation import load_and_prepare_data

logger = setup_logger(level="INFO")


def train_dataset_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> Tuple[xgb.XGBClassifier, float]:
    """
    Train dataset detector on train split and evaluate on holdout split.
    """
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "booster": "gbtree",
        "tree_method": "hist",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": 1,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_eval)
    auc = roc_auc_score(y_eval, y_pred_proba, multi_class="ovr", average="macro")
    return model, auc


def get_candidate_features(feature_cols: List[str]) -> List[str]:
    """
    Keep only raw/base traffic features for adversarial selection.
    Exclude downstream engineered or potentially leaky/meta features.
    """
    allowed = []
    for c in feature_cols:
        if c.startswith("q_"):
            continue
        if c.startswith("session_"):
            continue
        if "prob" in c.lower():
            continue
        allowed.append(c)
    return allowed


def run_adversarial_feature_selection(
    threshold: float = 0.75,
    min_features: int = 5,
) -> List[str]:
    paths = load_paths()

    logger.info("Loading data...")
    df_all = load_and_prepare_data()

    df_train = df_all[df_all["split"] == "train"].copy()
    df_val = df_all[df_all["split"] == "val"].copy()
    df_test = df_all[df_all["split"] == "test"].copy()

    logger.info("Fitting feature pipeline on TRAIN only...")
    pipeline = FeaturePipeline().fit(df_train)

    feature_cols = pipeline.model_feature_names()
    feature_cols = get_candidate_features(feature_cols)

    logger.info(f"Candidate feature count after filtering: {len(feature_cols)}")

    logger.info("Transforming splits...")
    train_t = pipeline.transform(df_train, strict=False)
    val_t = pipeline.transform(df_val, strict=False)
    test_t = pipeline.transform(df_test, strict=False)

    le = LabelEncoder()
    le.fit(df_all["dataset"])

    y_train = le.transform(train_t["dataset"])
    y_val = le.transform(val_t["dataset"])
    y_test = le.transform(test_t["dataset"])

    logger.info(f"Dataset classes: {list(le.classes_)}")

    selected_features = feature_cols.copy()

    logger.info("Training initial dataset detector (train -> val)...")
    X_train = train_t[selected_features].values
    X_val = val_t[selected_features].values
    model, auc_val = train_dataset_detector(X_train, y_train, X_val, y_val)
    logger.info(f"Initial validation dataset-detector AUC: {auc_val:.4f}")

    iteration = 0
    history = []

    while auc_val >= threshold and len(selected_features) > min_features:
        iteration += 1
        logger.info(
            f"Iteration {iteration}: val AUC = {auc_val:.4f}, features = {len(selected_features)}"
        )

        importances = model.feature_importances_
        max_idx = int(np.argmax(importances))
        feat_to_remove = selected_features[max_idx]
        importance_val = float(importances[max_idx])

        logger.info(f"Removing feature '{feat_to_remove}' with importance {importance_val:.4f}")
        history.append({
            "iteration": iteration,
            "removed_feature": feat_to_remove,
            "importance": importance_val,
            "val_auc_before_removal": float(auc_val),
            "n_features_before_removal": len(selected_features),
        })

        selected_features.remove(feat_to_remove)

        X_train = train_t[selected_features].values
        X_val = val_t[selected_features].values
        model, auc_val = train_dataset_detector(X_train, y_train, X_val, y_val)

    logger.info(f"Stopped with validation dataset-detector AUC: {auc_val:.4f}")
    logger.info(f"Selected {len(selected_features)} features")

    # Final check on TEST, only once at the end
    X_train_final = train_t[selected_features].values
    X_test_final = test_t[selected_features].values
    final_model, auc_test = train_dataset_detector(X_train_final, y_train, X_test_final, y_test)

    logger.info(f"Final holdout TEST dataset-detector AUC: {auc_test:.4f}")

    out_json = {
        "selected_features": selected_features,
        "val_auc_final": float(auc_val),
        "test_auc_final": float(auc_test),
        "min_features": min_features,
        "threshold": threshold,
        "history": history,
    }

    out_path = paths.artifacts_dir / "dataset_adversarial_selected_features.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    logger.info(f"Saved selected features to {out_path}")
    return selected_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--min-features", type=int, default=5)
    args = parser.parse_args()

    selected = run_adversarial_feature_selection(
        threshold=args.threshold,
        min_features=args.min_features,
    )
    print("Selected features:", selected)