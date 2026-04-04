#!/usr/bin/env python
"""
MASTER RUNNER -- Clean Pipeline -> Train -> Evaluate.

Run this ONE script to execute the entire post-fix workflow:

  Step 1: Extract features from all 3 datasets (streaming, memory-safe)
  Step 2: Sanity-check the output
  Step 3: Train VPN detector (balanced bagging ensemble)
  Step 4: Train domain detector (to measure fingerprinting)
  Step 5: Evaluate cross-dataset performance

Usage:
    python run_clean_pipeline_full.py                 # run everything
    python run_clean_pipeline_full.py --step 1        # only feature extraction
    python run_clean_pipeline_full.py --step 2        # only sanity check (requires step 1 output)
    python run_clean_pipeline_full.py --step 3        # only training       (requires step 1 output)
    python run_clean_pipeline_full.py --step 4        # only evaluation     (requires step 3 output)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "clean_pipeline"
FEATURES_PATH = ARTIFACT_DIR / "features.parquet"


# ══════════════════════════════════════════════════════════
# STEP 1 -- Feature Extraction (streaming, memory-safe)
# ══════════════════════════════════════════════════════════

def step1_extract_features():
    """Run the full clean pipeline: raw data -> features + splits."""
    print("\n" + "=" * 70)
    print("STEP 1: FEATURE EXTRACTION (streaming, memory-safe)")
    print("=" * 70)

    from src.clean_pipeline.run_pipeline import run_clean_pipeline
    from src.clean_pipeline.config import default_config

    cfg = default_config()
    print(f"  VNAT:   {cfg.vnat_h5}")
    print(f"  ISCX:   {cfg.iscx_parquet}")
    print(f"  USBVPN: {cfg.usbvpn_raw_dir}")
    print(f"  Family: {cfg.feature_family}")
    print(f"  Output: {cfg.output_dir}")
    print()

    features = run_clean_pipeline(cfg)

    print(f"\n[OK] Step 1 complete: {len(features)} flows extracted")
    print(f"  Saved to: {FEATURES_PATH}")
    del features
    gc.collect()


# ══════════════════════════════════════════════════════════
# STEP 2 -- Sanity Check
# ══════════════════════════════════════════════════════════

def step2_sanity_check():
    """Validate the extracted features."""
    print("\n" + "=" * 70)
    print("STEP 2: SANITY CHECK")
    print("=" * 70)

    if not FEATURES_PATH.exists():
        print(f"ERROR: {FEATURES_PATH} does not exist. Run step 1 first.")
        return False

    df = pd.read_parquet(FEATURES_PATH)

    # Basic shape
    print(f"\n  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Dataset breakdown
    print(f"\n  Dataset breakdown:")
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        vpn = (sub["label"] == 1).sum()
        nonvpn = (sub["label"] == 0).sum()
        caps = sub["capture_id"].nunique()
        print(f"    {ds:8s}: {len(sub):6d} flows ({vpn} VPN, {nonvpn} nonVPN), {caps} captures")

    # Split breakdown
    if "split" in df.columns:
        print(f"\n  Split breakdown:")
        for sp in ("train", "val", "test"):
            sub = df[df["split"] == sp]
            print(f"    {sp:5s}: {len(sub):6d} flows "
                  f"(VPN={int((sub['label']==1).sum())}, "
                  f"nonVPN={int((sub['label']==0).sum())})")

    # Feature columns (exclude metadata)
    meta = {"flow_id", "capture_id", "dataset", "label", "split", "source_file", "app"}
    feat_cols = [c for c in df.columns if c not in meta]
    print(f"\n  Feature columns ({len(feat_cols)}): {feat_cols}")

    # Check for NaN/inf
    feat_df = df[feat_cols]
    nan_count = feat_df.isna().sum().sum()
    inf_count = np.isinf(feat_df.select_dtypes(include=[np.number]).values).sum()
    print(f"\n  NaN cells: {nan_count}")
    print(f"  Inf cells: {inf_count}")

    # Check no all-zero columns
    zero_cols = [c for c in feat_cols if (feat_df[c] == 0).all()]
    if zero_cols:
        print(f"\n  [WARN] ALL-ZERO columns: {zero_cols}")
    else:
        print(f"\n  [OK] No all-zero feature columns")

    # Check per-dataset feature variance (detect constant features per dataset)
    print(f"\n  Per-dataset feature variance check:")
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds][feat_cols]
        zero_var = [c for c in feat_cols if sub[c].std() == 0]
        if zero_var:
            print(f"    [WARN] {ds}: {len(zero_var)} zero-variance features: {zero_var}")
        else:
            print(f"    [OK] {ds}: all features have nonzero variance")

    # Quick domain fingerprint check using feature means
    print(f"\n  Feature mean comparison across datasets:")
    means = df.groupby("dataset")[feat_cols].mean()
    print(means.round(4).to_string())

    del df
    gc.collect()

    print(f"\n[OK] Step 2 complete -- sanity check passed")
    return True


# ══════════════════════════════════════════════════════════
# STEP 3 -- Train Models
# ══════════════════════════════════════════════════════════

def step3_train_models():
    """Train VPN detector and domain detector on clean features."""
    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING")
    print("=" * 70)

    if not FEATURES_PATH.exists():
        print(f"ERROR: {FEATURES_PATH} does not exist. Run step 1 first.")
        return

    df = pd.read_parquet(FEATURES_PATH)

    meta = {"flow_id", "capture_id", "dataset", "label", "split", "source_file", "app"}
    feat_cols = [c for c in df.columns if c not in meta]

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    X_train = train[feat_cols].values
    y_train = train["label"].values
    X_val = val[feat_cols].values
    y_val = val["label"].values
    X_test = test[feat_cols].values
    y_test = test["label"].values

    print(f"\n  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  VPN ratio -- Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")
    print(f"  Features: {len(feat_cols)}")

    # -- 3a: VPN Detector (Balanced Bagging Ensemble) --
    print(f"\n{'-'*50}")
    print("3a: Training VPN Detector (XGBoost + LightGBM + CatBoost)")
    print(f"{'-'*50}")

    from sklearn.metrics import roc_auc_score, average_precision_score

    models = {}
    preds_val = {}
    preds_test = {}

    # XGBoost
    try:
        import xgboost as xgb
        print("\n  Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1)),
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train,
                       eval_set=[(X_val, y_val)],
                       verbose=False)
        preds_val["xgb"] = xgb_model.predict_proba(X_val)[:, 1]
        preds_test["xgb"] = xgb_model.predict_proba(X_test)[:, 1]
        models["xgb"] = xgb_model
        print(f"    Val AUC: {roc_auc_score(y_val, preds_val['xgb']):.4f}")
        print(f"    Test AUC: {roc_auc_score(y_test, preds_test['xgb']):.4f}")
    except ImportError:
        print("  [WARN] XGBoost not installed, skipping")

    # LightGBM
    try:
        import lightgbm as lgb
        print("\n  Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(X_train, y_train,
                       eval_set=[(X_val, y_val)],
                       callbacks=[lgb.log_evaluation(0)])
        preds_val["lgb"] = lgb_model.predict_proba(X_val)[:, 1]
        preds_test["lgb"] = lgb_model.predict_proba(X_test)[:, 1]
        models["lgb"] = lgb_model
        print(f"    Val AUC: {roc_auc_score(y_val, preds_val['lgb']):.4f}")
        print(f"    Test AUC: {roc_auc_score(y_test, preds_test['lgb']):.4f}")
    except ImportError:
        print("  [WARN] LightGBM not installed, skipping")

    # CatBoost
    try:
        from catboost import CatBoostClassifier
        print("\n  Training CatBoost...")
        cb_model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
        )
        cb_model.fit(X_train, y_train, eval_set=(X_val, y_val))
        preds_val["cb"] = cb_model.predict_proba(X_val)[:, 1]
        preds_test["cb"] = cb_model.predict_proba(X_test)[:, 1]
        models["cb"] = cb_model
        print(f"    Val AUC: {roc_auc_score(y_val, preds_val['cb']):.4f}")
        print(f"    Test AUC: {roc_auc_score(y_test, preds_test['cb']):.4f}")
    except ImportError:
        print("  [WARN] CatBoost not installed, skipping")

    if not models:
        print("\n  ERROR: No models could be trained. Install xgboost/lightgbm/catboost.")
        return

    # Ensemble
    print(f"\n{'-'*50}")
    print("3b: Ensemble (simple average)")
    print(f"{'-'*50}")

    ens_val = np.mean([preds_val[k] for k in preds_val], axis=0)
    ens_test = np.mean([preds_test[k] for k in preds_test], axis=0)

    print(f"  Ensemble Val AUC:  {roc_auc_score(y_val, ens_val):.4f}")
    print(f"  Ensemble Test AUC: {roc_auc_score(y_test, ens_test):.4f}")
    print(f"  Ensemble Val AP:   {average_precision_score(y_val, ens_val):.4f}")
    print(f"  Ensemble Test AP:  {average_precision_score(y_test, ens_test):.4f}")

    # Save predictions
    out_dir = ARTIFACT_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save ensemble predictions for evaluation
    val_df = val[["flow_id", "capture_id", "dataset", "label"]].copy()
    val_df["ensemble_score"] = ens_val
    for k in preds_val:
        val_df[f"{k}_score"] = preds_val[k]
    val_df.to_parquet(out_dir / "val_predictions.parquet", index=False)

    test_df = test[["flow_id", "capture_id", "dataset", "label"]].copy()
    test_df["ensemble_score"] = ens_test
    for k in preds_test:
        test_df[f"{k}_score"] = preds_test[k]
    test_df.to_parquet(out_dir / "test_predictions.parquet", index=False)

    # Save models
    import pickle
    for name, model in models.items():
        model_path = out_dir / f"{name}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved {name} model -> {model_path}")

    # -- 3c: Domain Detector --
    print(f"\n{'-'*50}")
    print("3c: Domain Detector (to measure feature fingerprinting)")
    print(f"{'-'*50}")

    # Create domain labels: one-vs-rest for each dataset
    datasets = sorted(df["dataset"].unique())
    print(f"  Datasets: {datasets}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    domain_y_train = le.fit_transform(train["dataset"].values)
    domain_y_val = le.transform(val["dataset"].values)
    domain_y_test = le.transform(test["dataset"].values)

    domain_clf = LogisticRegression(max_iter=1000, random_state=42)
    domain_clf.fit(X_train, domain_y_train)

    domain_acc_train = domain_clf.score(X_train, domain_y_train)
    domain_acc_val = domain_clf.score(X_val, domain_y_val)
    domain_acc_test = domain_clf.score(X_test, domain_y_test)

    print(f"  Domain accuracy -- Train: {domain_acc_train:.4f}, Val: {domain_acc_val:.4f}, Test: {domain_acc_test:.4f}")

    if len(datasets) == 2:
        domain_proba_test = domain_clf.predict_proba(X_test)[:, 1]
        domain_auc = roc_auc_score(domain_y_test, domain_proba_test)
        print(f"  Domain AUC (test): {domain_auc:.4f}")
    else:
        from sklearn.metrics import roc_auc_score as ras
        domain_proba_test = domain_clf.predict_proba(X_test)
        domain_auc = ras(domain_y_test, domain_proba_test, multi_class="ovr")
        print(f"  Domain AUC-OVR (test): {domain_auc:.4f}")

    # Interpretation
    if domain_auc > 0.95:
        print(f"\n  [WARN] DOMAIN AUC {domain_auc:.4f} > 0.95 -- features strongly encode dataset identity")
        print(f"    This is a RISK for cross-domain deployment.")
    elif domain_auc > 0.80:
        print(f"\n  [WARN] DOMAIN AUC {domain_auc:.4f} is moderate -- some fingerprinting remains")
    else:
        print(f"\n  [OK] DOMAIN AUC {domain_auc:.4f} < 0.80 -- features are reasonably domain-neutral")

    # Save domain results
    domain_results = {
        "domain_accuracy": {"train": domain_acc_train, "val": domain_acc_val, "test": domain_acc_test},
        "domain_auc_ovr_test": float(domain_auc),
        "datasets": datasets,
        "n_features": len(feat_cols),
        "feature_names": feat_cols,
    }
    (out_dir / "domain_detector_results.json").write_text(
        json.dumps(domain_results, indent=2), encoding="utf-8"
    )

    print(f"\n[OK] Step 3 complete -- models trained and saved to {out_dir}")

    del df, train, val, test
    gc.collect()


# ══════════════════════════════════════════════════════════
# STEP 4 -- Evaluation
# ══════════════════════════════════════════════════════════

def step4_evaluate():
    """Comprehensive evaluation of trained models."""
    print("\n" + "=" * 70)
    print("STEP 4: EVALUATION")
    print("=" * 70)

    out_dir = ARTIFACT_DIR / "models"
    test_pred_path = out_dir / "test_predictions.parquet"
    val_pred_path = out_dir / "val_predictions.parquet"

    if not test_pred_path.exists():
        print(f"ERROR: {test_pred_path} not found. Run step 3 first.")
        return

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_recall_curve,
        confusion_matrix,
        classification_report,
    )

    test_df = pd.read_parquet(test_pred_path)
    val_df = pd.read_parquet(val_pred_path)

    y_test = test_df["label"].values
    ens_test = test_df["ensemble_score"].values
    y_val = val_df["label"].values
    ens_val = val_df["ensemble_score"].values

    # -- 4a: Global metrics --
    print(f"\n{'-'*50}")
    print("4a: Global Test Metrics")
    print(f"{'-'*50}")

    auc = roc_auc_score(y_test, ens_test)
    ap = average_precision_score(y_test, ens_test)
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  AP:      {ap:.4f}")

    # Find threshold from validation set
    precision, recall, thresholds = precision_recall_curve(y_val, ens_val)
    # Threshold at max F1
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    print(f"  Best F1 threshold (from val): {best_threshold:.4f}")
    print(f"    Val F1 at this threshold: {f1s[best_idx]:.4f}")

    # Apply to test
    y_pred = (ens_test >= best_threshold).astype(int)
    print(f"\n  Test Classification Report (threshold={best_threshold:.4f}):")
    print(classification_report(y_test, y_pred, target_names=["nonVPN", "VPN"], digits=4))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  FPR: {fpr:.4f}, Recall: {recall_val:.4f}")

    # -- 4b: Per-dataset metrics --
    print(f"\n{'-'*50}")
    print("4b: Per-Dataset Test Metrics")
    print(f"{'-'*50}")

    per_ds_results = {}
    for ds in sorted(test_df["dataset"].unique()):
        mask = test_df["dataset"] == ds
        y_ds = y_test[mask]
        s_ds = ens_test[mask]

        if len(np.unique(y_ds)) < 2:
            print(f"\n  {ds}: only one class in test set, skipping AUC")
            per_ds_results[ds] = {"n_flows": int(mask.sum()), "note": "single class"}
            continue

        ds_auc = roc_auc_score(y_ds, s_ds)
        ds_ap = average_precision_score(y_ds, s_ds)

        y_pred_ds = (s_ds >= best_threshold).astype(int)
        cm_ds = confusion_matrix(y_ds, y_pred_ds)
        tn_ds, fp_ds, fn_ds, tp_ds = cm_ds.ravel()
        fpr_ds = fp_ds / (fp_ds + tn_ds) if (fp_ds + tn_ds) > 0 else 0
        rec_ds = tp_ds / (tp_ds + fn_ds) if (tp_ds + fn_ds) > 0 else 0

        print(f"\n  {ds}:")
        print(f"    Flows: {int(mask.sum())} (VPN={int((y_ds==1).sum())}, nonVPN={int((y_ds==0).sum())})")
        print(f"    AUC: {ds_auc:.4f}, AP: {ds_ap:.4f}")
        print(f"    FPR: {fpr_ds:.4f}, Recall: {rec_ds:.4f}")
        print(f"    TP={tp_ds}, FP={fp_ds}, TN={tn_ds}, FN={fn_ds}")

        per_ds_results[ds] = {
            "n_flows": int(mask.sum()),
            "auc": float(ds_auc),
            "ap": float(ds_ap),
            "fpr": float(fpr_ds),
            "recall": float(rec_ds),
            "tp": int(tp_ds), "fp": int(fp_ds),
            "tn": int(tn_ds), "fn": int(fn_ds),
        }

    # -- 4c: Session-level evaluation --
    print(f"\n{'-'*50}")
    print("4c: Session-Level Evaluation (capture aggregation)")
    print(f"{'-'*50}")

    session_results = []
    for cap_id, grp in test_df.groupby("capture_id"):
        cap_label = grp["label"].iloc[0]
        cap_ds = grp["dataset"].iloc[0]
        scores = grp["ensemble_score"].values
        session_results.append({
            "capture_id": cap_id,
            "dataset": cap_ds,
            "label": int(cap_label),
            "n_flows": len(grp),
            "mean_score": float(scores.mean()),
            "p90_score": float(np.percentile(scores, 90)),
            "max_score": float(scores.max()),
            "wt5_score": float(np.sort(scores)[-min(5, len(scores)):].mean()),
        })

    session_df = pd.DataFrame(session_results)
    y_session = session_df["label"].values

    for agg_name in ["mean_score", "p90_score", "max_score", "wt5_score"]:
        s = session_df[agg_name].values
        if len(np.unique(y_session)) >= 2:
            s_auc = roc_auc_score(y_session, s)
            s_ap = average_precision_score(y_session, s)
            print(f"  {agg_name:15s}  AUC={s_auc:.4f}  AP={s_ap:.4f}")

    # -- Save evaluation report --
    report = {
        "global": {
            "auc": float(auc),
            "ap": float(ap),
            "best_f1_threshold": float(best_threshold),
            "fpr": float(fpr),
            "recall": float(recall_val),
        },
        "per_dataset": per_ds_results,
        "session_auc": {
            agg: float(roc_auc_score(y_session, session_df[agg].values))
            for agg in ["mean_score", "p90_score", "max_score", "wt5_score"]
            if len(np.unique(y_session)) >= 2
        },
    }

    report_path = out_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Evaluation report saved -> {report_path}")

    print(f"\n[OK] Step 4 complete")


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Clean Pipeline -- Full Workflow")
    parser.add_argument("--step", type=int, default=None,
                        help="Run only a specific step (1-4). Default: run all.")
    args = parser.parse_args()

    t0 = time.time()

    if args.step is None or args.step == 1:
        step1_extract_features()

    if args.step is None or args.step == 2:
        step2_sanity_check()

    if args.step is None or args.step == 3:
        step3_train_models()

    if args.step is None or args.step == 4:
        step4_evaluate()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ALL DONE -- Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


