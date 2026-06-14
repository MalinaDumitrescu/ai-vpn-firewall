#!/usr/bin/env python3
"""
train_3dataset_and_lodo.py
==========================

Retrains the firewall balanced-bagging ensemble (3x XGB + 3x LGBM + 3x CatBoost
with isotonic + Platt calibration) on all THREE datasets pooled
(ISCX + USBVPN + VNAT) and then runs a full LODO (Leave-One-Dataset-Out)
evaluation: for each dataset, train on the other two and test on the held-out
one.

Outputs (all FRESH; production weights at
`artifacts/balanced_bagging_firewall_tuned_ensemble/` are untouched):

  artifacts/balanced_bagging_firewall_tuned_ensemble_3dataset_REFRESH/
  artifacts/lood_firewall_tuned/
      hold_iscx/     <- trained on usbvpn+vnat, tested on iscx
      hold_usbvpn/   <- trained on iscx+vnat,   tested on usbvpn
      hold_vnat/     <- trained on iscx+usbvpn, tested on vnat
      lodo_summary.json
      pooled_3ds_summary.json

Reuses the exact recipe from notebook 29, cell 3 (RETRAIN_3DS=True).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import load_paths
from src.utils.logging import setup_logger
from src.pipeline.feature_pipeline import FeaturePipeline
from src.pipeline.artifacts import default_feature_artifacts
from src.pipeline.data_preparation import load_and_prepare_data
from src.models.train_balanced_bagging_ensemble import run_balanced_bagging
from src.features.extract import load_feature_config, feature_config_hash_text


LOG = logging.getLogger("train_3ds_lodo")


def _load_optuna_params(paths) -> tuple[dict, dict, dict]:
    """Load firewall-objective Optuna hyperparameters for each family."""
    with open(paths.artifacts_dir / "optuna_xgboost_firewall_best_params.json") as f:
        xgb_params = json.load(f)
    xgb_params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": "gbtree",
            "tree_method": "hist",
            "n_estimators": 1000,
            "random_state": 42,
            "n_jobs": 1,
            "early_stopping_rounds": 50,
        }
    )
    with open(paths.artifacts_dir / "optuna_catboost_firewall_best_params.json") as f:
        cat_params = json.load(f)
    cat_params.update(
        {
            "iterations": 1000,
            "random_seed": 42,
            "thread_count": 1,
            "verbose": False,
            "allow_writing_files": False,
            "early_stopping_rounds": 150,
        }
    )
    with open(paths.artifacts_dir / "optuna_lgbm_firewall_best_params.json") as f:
        lgbm_params = json.load(f)
    lgbm_params.update(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 1000,
            "verbose": -1,
            "random_state": 42,
            "n_jobs": 1,
        }
    )
    return xgb_params, lgbm_params, cat_params


def _fit_pipeline_and_transform(df_all: pd.DataFrame, paths):
    """Fit the FeaturePipeline on TRAIN rows and transform the full df."""
    features_yaml = paths.configs_dir / "features.yaml"
    pipeline = FeaturePipeline().fit(df_all[df_all["split"] == "train"].copy())
    feature_art = default_feature_artifacts(paths.artifacts_dir / "features")
    pipeline.save(feature_art, feature_config_hash=feature_config_hash_text(features_yaml))
    feature_cols = pipeline.model_feature_names()

    df_t = pipeline.transform(df_all)
    for col in ("label", "split", "capture_id", "dataset", "flow_id"):
        if col in df_all.columns:
            df_t[col] = df_all[col].values
    return df_t, feature_cols


def _train_one(
    df: pd.DataFrame,
    feature_cols: list[str],
    xgb_params: dict,
    lgbm_params: dict,
    cat_params: dict,
    output_dir: Path,
    tag: str,
) -> dict:
    """Run one balanced-bagging training and return its metrics dict."""
    LOG.info("=" * 88)
    LOG.info(f"[{tag}] Training balanced bagging ensemble -> {output_dir}")
    LOG.info("=" * 88)
    counts = pd.crosstab(df["dataset"], df["split"])
    LOG.info(f"[{tag}] Split x Dataset counts:\n{counts}\n")
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = run_balanced_bagging(
        df=df,
        label_col="label",
        group_col="capture_id",
        dataset_col="dataset",
        split_col="split",
        bags_per_family=3,
        majority_ratio=1.0,
        target_fprs="0.0,0.001,0.005,0.01",
        seed=42,
        output_dir=str(output_dir),
        model_types=["xgb", "lgbm", "cat"],
        feature_cols=feature_cols,
        weight_xgb=1.0,
        weight_lgbm=1.0,
        weight_cat=1.0,
        xgb_params=xgb_params,
        lgbm_params=lgbm_params,
        cat_params=cat_params,
    )
    dt = time.time() - t0
    LOG.info(f"[{tag}] Training finished in {dt/60:.1f} min")
    return results


def _make_lodo_df(df_transformed: pd.DataFrame, holdout: str) -> pd.DataFrame:
    """
    Build a LODO-shaped DataFrame:
      - train  : original train rows from datasets != holdout
      - val    : original val   rows from datasets != holdout
      - test   : ALL rows of the held-out dataset
    """
    df = df_transformed.copy()
    other = df["dataset"] != holdout
    held = df["dataset"] == holdout

    out_parts = []
    out_parts.append(df.loc[other & (df["split"] == "train")])
    out_parts.append(df.loc[other & (df["split"] == "val")])
    held_df = df.loc[held].copy()
    held_df["split"] = "test"
    out_parts.append(held_df)
    out = pd.concat(out_parts, ignore_index=True)
    return out


def _summarize_pooled(results: dict) -> dict:
    """Extract the headline metrics from a pooled (3-dataset) training run."""
    out = {}
    for cal in ("raw", "isotonic", "platt"):
        sec = results.get(cal, {})
        for split_key in ("val", "test_overall", "test_iscx", "test_usbvpn", "test_vnat"):
            s = sec.get(split_key)
            if not s:
                continue
            sm = s.get("session_metrics", {})
            out[f"{cal}/{split_key}"] = {
                "auc": s.get("auc"),
                "pr_auc": s.get("pr_auc"),
                "fpr_0.0_recall": s.get("fpr_0.0", {}).get("recall"),
                "fpr_0.0_threshold": s.get("fpr_0.0", {}).get("threshold"),
                "fpr_0.0_fpr_actual": s.get("fpr_0.0", {}).get("fpr"),
                "session_roc_auc": sm.get("session_roc_auc"),
                "block_recall_at_zero_fp": sm.get("block_recall_at_zero_fp"),
            }
    return out


def _summarize_lodo(results: dict, holdout: str) -> dict:
    """Extract LODO-relevant metrics: held-out test_overall == test_<holdout>."""
    out = {"holdout": holdout}
    for cal in ("raw", "isotonic", "platt"):
        sec = results.get(cal, {})
        v = sec.get("val", {})
        t = sec.get("test_overall", {})
        out[cal] = {
            "val_auc": v.get("auc"),
            "val_pr_auc": v.get("pr_auc"),
            "holdout_auc": t.get("auc"),
            "holdout_pr_auc": t.get("pr_auc"),
            "holdout_recall_at_zero_fpr": (t.get("fpr_0.0") or {}).get("recall"),
            "holdout_session_roc_auc": (t.get("session_metrics") or {}).get("session_roc_auc"),
            "holdout_block_recall_at_zero_fp": (t.get("session_metrics") or {}).get(
                "block_recall_at_zero_fp"
            ),
        }
    return out


def main():
    setup_logger(level="INFO")
    paths = load_paths()

    artifacts_dir: Path = paths.artifacts_dir
    pooled_out = artifacts_dir / "balanced_bagging_firewall_tuned_ensemble_3dataset_REFRESH"
    lodo_root = artifacts_dir / "lood_firewall_tuned"
    lodo_root.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading all three datasets ...")
    df_all = load_and_prepare_data(vnat_only=False)
    LOG.info(f"Loaded {len(df_all):,} flows from {sorted(df_all['dataset'].unique())}")
    LOG.info(f"Pooled split x dataset:\n{pd.crosstab(df_all['dataset'], df_all['split'])}")

    LOG.info("Fitting FeaturePipeline on pooled TRAIN ...")
    df_t, feature_cols = _fit_pipeline_and_transform(df_all, paths)
    LOG.info(f"Pipeline produced {len(feature_cols)} features: {feature_cols}")

    xgb_params, lgbm_params, cat_params = _load_optuna_params(paths)

    # ---------- 1) Pooled 3-dataset retrain ----------
    pooled_results = _train_one(
        df=df_t,
        feature_cols=feature_cols,
        xgb_params=xgb_params,
        lgbm_params=lgbm_params,
        cat_params=cat_params,
        output_dir=pooled_out,
        tag="POOLED-3DS",
    )
    pooled_summary = _summarize_pooled(pooled_results)
    with open(lodo_root / "pooled_3ds_summary.json", "w") as f:
        json.dump(pooled_summary, f, indent=2, default=float)
    LOG.info(f"Pooled summary saved -> {lodo_root / 'pooled_3ds_summary.json'}")

    # ---------- 2) LODO ----------
    lodo_all: list[dict] = []
    for holdout in ("iscx", "usbvpn", "vnat"):
        df_lodo = _make_lodo_df(df_t, holdout)

        # Guard: ensure both classes are present in val and test
        for split in ("train", "val", "test"):
            sub = df_lodo[df_lodo["split"] == split]
            if sub["label"].nunique() < 2 and split in ("val", "test"):
                raise RuntimeError(
                    f"LODO fold hold_{holdout}: split={split} lacks both classes "
                    f"(label counts={sub['label'].value_counts().to_dict()})"
                )

        fold_dir = lodo_root / f"hold_{holdout}"
        fold_results = _train_one(
            df=df_lodo,
            feature_cols=feature_cols,
            xgb_params=xgb_params,
            lgbm_params=lgbm_params,
            cat_params=cat_params,
            output_dir=fold_dir,
            tag=f"LODO-hold_{holdout}",
        )
        fold_summary = _summarize_lodo(fold_results, holdout)
        lodo_all.append(fold_summary)
        with open(fold_dir / "lodo_summary.json", "w") as f:
            json.dump(fold_summary, f, indent=2, default=float)

    with open(lodo_root / "lodo_summary.json", "w") as f:
        json.dump(lodo_all, f, indent=2, default=float)
    LOG.info(f"LODO summary saved -> {lodo_root / 'lodo_summary.json'}")

    # ---------- Print compact comparison ----------
    print("\n" + "=" * 88)
    print("POOLED 3-DATASET TRAINING (test on held-out captures from all 3 datasets)")
    print("=" * 88)
    for cal in ("raw", "isotonic", "platt"):
        for sk in ("test_overall", "test_iscx", "test_usbvpn", "test_vnat"):
            row = pooled_summary.get(f"{cal}/{sk}")
            if not row:
                continue
            print(
                f"  [{cal:8s}] {sk:14s}  AUC={row['auc']:.4f}  "
                f"PR-AUC={row['pr_auc']:.4f}  "
                f"sess_AUC={row['session_roc_auc']}  "
                f"block_rec@0={row['block_recall_at_zero_fp']}"
            )

    print("\n" + "=" * 88)
    print("LODO (train on 2 datasets, test on the held-out one)")
    print("=" * 88)
    iso_aucs = []
    for entry in lodo_all:
        ho = entry["holdout"]
        iso = entry["isotonic"]
        iso_aucs.append(iso["holdout_auc"])
        print(
            f"  hold_{ho:7s}  iso  holdout_AUC={iso['holdout_auc']:.4f}  "
            f"PR-AUC={iso['holdout_pr_auc']:.4f}  "
            f"sess_AUC={iso['holdout_session_roc_auc']}  "
            f"block_rec@0={iso['holdout_block_recall_at_zero_fp']}"
        )
    iso_aucs = [a for a in iso_aucs if a is not None]
    if iso_aucs:
        print(
            f"\n  LODO summary (isotonic):  min_AUC={min(iso_aucs):.4f}  "
            f"mean_AUC={np.mean(iso_aucs):.4f}  max_AUC={max(iso_aucs):.4f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
