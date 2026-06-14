#!/usr/bin/env python3
"""
train_clean_pipeline_singles_pooled.py
======================================

Retrains the three single-family models (XGBoost, LightGBM, CatBoost) on the
LARGE pooled multi-dataset feature pool (66,862 flows from ISCX + USBVPN +
VNAT), replacing the small clean_pipeline subset (~4,712 flows). Uses the
firewall-objective Optuna hyperparameters and the same FeaturePipeline as the
balanced-bagging ensemble.

Outputs:
  artifacts/clean_pipeline/models_3ds_66k/
      xgb_model.pkl
      lgb_model.pkl
      cb_model.pkl
      evaluation_report.json
      predictions.csv  (val + test, all 3 datasets)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.utils.paths import load_paths
from src.utils.logging import setup_logger
from src.pipeline.feature_pipeline import FeaturePipeline
from src.pipeline.data_preparation import load_and_prepare_data

import xgboost as xgb
import lightgbm as lgb
import catboost as cb


def _load_params(art):
    with open(art / "optuna_xgboost_firewall_best_params.json") as f: xp = json.load(f)
    xp.update({"objective": "binary:logistic", "eval_metric": "logloss",
               "booster": "gbtree", "tree_method": "hist",
               "n_estimators": 1000, "random_state": 42, "n_jobs": 1,
               "early_stopping_rounds": 50})
    with open(art / "optuna_catboost_firewall_best_params.json") as f: cp = json.load(f)
    cp.update({"iterations": 1000, "random_seed": 42, "thread_count": 1,
               "verbose": False, "allow_writing_files": False,
               "early_stopping_rounds": 150})
    with open(art / "optuna_lgbm_firewall_best_params.json") as f: lp = json.load(f)
    lp.update({"objective": "binary", "metric": "binary_logloss",
               "boosting_type": "gbdt", "n_estimators": 1000,
               "verbose": -1, "random_state": 42, "n_jobs": 1})
    return xp, lp, cp


def main():
    setup_logger(level="INFO")
    paths = load_paths()
    out_dir = paths.artifacts_dir / "clean_pipeline" / "models_3ds_66k"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pooled (3-DS) dataset ...")
    df_all = load_and_prepare_data(vnat_only=False)
    print(pd.crosstab(df_all["dataset"], df_all["split"]))

    pipeline = FeaturePipeline().fit(df_all[df_all["split"] == "train"].copy())
    feature_cols = pipeline.model_feature_names()
    df_t = pipeline.transform(df_all)
    for c in ("label", "split", "capture_id", "dataset", "flow_id"):
        if c in df_all.columns:
            df_t[c] = df_all[c].values
    print(f"Pipeline produced {len(feature_cols)} features.")

    X = df_t[feature_cols]; y = df_t["label"].astype(int)
    m_tr = df_t["split"] == "train"; m_va = df_t["split"] == "val"; m_te = df_t["split"] == "test"
    Xtr, ytr = X[m_tr], y[m_tr]
    Xva, yva = X[m_va], y[m_va]
    Xte, yte = X[m_te], y[m_te]
    print(f"Sizes  train={len(Xtr):,}  val={len(Xva):,}  test={len(Xte):,}")

    xp, lp, cp = _load_params(paths.artifacts_dir)

    models, probs_val, probs_test = {}, {}, {}

    print("\nTraining XGBoost ...")
    t0 = time.time()
    m = xgb.XGBClassifier(**xp); m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    models["xgb"] = m
    probs_val["xgb"]  = m.predict_proba(Xva)[:, 1]
    probs_test["xgb"] = m.predict_proba(Xte)[:, 1]
    joblib.dump(m, out_dir / "xgb_model.pkl")
    print(f"  done in {time.time()-t0:.1f}s")

    print("Training LightGBM ...")
    t0 = time.time()
    m = lgb.LGBMClassifier(**lp)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    models["lgb"] = m
    probs_val["lgb"]  = m.predict_proba(Xva)[:, 1]
    probs_test["lgb"] = m.predict_proba(Xte)[:, 1]
    joblib.dump(m, out_dir / "lgb_model.pkl")
    print(f"  done in {time.time()-t0:.1f}s")

    print("Training CatBoost ...")
    t0 = time.time()
    m = cb.CatBoostClassifier(**cp)
    m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=150)
    models["cb"] = m
    probs_val["cb"]  = m.predict_proba(Xva)[:, 1]
    probs_test["cb"] = m.predict_proba(Xte)[:, 1]
    joblib.dump(m, out_dir / "cb_model.pkl")
    print(f"  done in {time.time()-t0:.1f}s")

    # Simple mean ensemble
    ens_val  = np.mean(list(probs_val.values()), axis=0)
    ens_test = np.mean(list(probs_test.values()), axis=0)

    def _metrics(y, p):
        return {"auc": float(roc_auc_score(y, p)),
                "pr_auc": float(average_precision_score(y, p))}

    report = {"global": {}, "per_dataset": {}}
    for name, p_val, p_te in [("xgb", probs_val["xgb"], probs_test["xgb"]),
                              ("lgb", probs_val["lgb"], probs_test["lgb"]),
                              ("cb",  probs_val["cb"],  probs_test["cb"]),
                              ("ensemble_mean", ens_val, ens_test)]:
        report["global"][name] = {"val": _metrics(yva, p_val),
                                  "test": _metrics(yte, p_te)}

    ds_test = df_t.loc[m_te, "dataset"].values
    for ds in sorted(np.unique(ds_test)):
        mask = ds_test == ds
        y_ds = yte.values[mask]
        if len(np.unique(y_ds)) < 2:
            report["per_dataset"][ds] = {"note": "single class on test"}
            continue
        report["per_dataset"][ds] = {
            "n_flows": int(mask.sum()),
            "n_vpn": int(y_ds.sum()),
            "xgb": _metrics(y_ds, probs_test["xgb"][mask]),
            "lgb": _metrics(y_ds, probs_test["lgb"][mask]),
            "cb":  _metrics(y_ds, probs_test["cb"][mask]),
            "ensemble_mean": _metrics(y_ds, ens_test[mask]),
        }

    with open(out_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Predictions CSV (val + test)
    pred_rows = []
    for split, mask, probs in [("val", m_va, probs_val), ("test", m_te, probs_test)]:
        sub = df_t.loc[mask, ["capture_id", "dataset", "label", "flow_id"]].copy()
        sub["xgb_score"] = probs["xgb"]
        sub["lgb_score"] = probs["lgb"]
        sub["cb_score"]  = probs["cb"]
        sub["ensemble_score"] = np.mean([probs["xgb"], probs["lgb"], probs["cb"]], axis=0)
        sub["split"] = split
        pred_rows.append(sub)
    pd.concat(pred_rows).to_csv(out_dir / "predictions.csv", index=False)

    print("\nEvaluation report:")
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
