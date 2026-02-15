from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json
import numpy as np
import pandas as pd
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

import joblib


@dataclass(frozen=True)
class CalibrateResult:
    calibrator_path: Path
    metrics_path: Path
    preds_calibrated_path: Path
    metrics: Dict[str, Any]


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _evaluate(split_df: pd.DataFrame, prob_col: str) -> Dict[str, Any]:
    y = split_df["label"].to_numpy(dtype=int)
    p = split_df[prob_col].to_numpy(dtype=float)

    # Guard: if split has only one class, AUC is undefined
    if len(np.unique(y)) < 2:
        roc = None
        pr = None
    else:
        roc = float(roc_auc_score(y, p))
        pr = float(average_precision_score(y, p))

    y_hat = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_hat, labels=[0, 1]).ravel()

    precision = float(tp / (tp + fp + 1e-9))
    recall = float(tp / (tp + fn + 1e-9))

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "threshold_0.5": {
            "precision": precision,
            "recall": recall,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
    }


def calibrate_xgb_predictions(
    *,
    paths,
    calib_yaml: Path,
    df_preds: Optional[pd.DataFrame] = None,
) -> CalibrateResult:
    """
    Calibrate XGBoost probabilities (p_xgb) using validation split only.

    Input:
      artifacts/xgb/preds.parquet with columns:
        split, label, p_xgb (+ optional flow_id, capture_id)

    Output:
      artifacts/xgb/calibrator.pkl
      artifacts/xgb/metrics_calibrated.json
      artifacts/xgb/preds_calibrated.parquet  (adds p_calib)
    """
    cfg = _load_yaml(calib_yaml)

    method = str(cfg.get("method", "platt")).strip().lower()
    if method not in {"platt", "isotonic"}:
        raise ValueError("method must be one of: platt, isotonic")

    splits = cfg.get("splits") or {}
    train_name = str(splits.get("train", "train"))
    val_name = str(splits.get("val", "val"))
    test_name = str(splits.get("test", "test"))

    in_cfg = cfg.get("inputs") or {}
    preds_rel = str(in_cfg.get("preds_path", "artifacts/xgb/preds.parquet"))
    preds_path = (paths.repo_root / preds_rel).resolve()

    out_cfg = cfg.get("outputs") or {}
    calibrator_rel = str(out_cfg.get("calibrator_path", "artifacts/xgb/calibrator.pkl"))
    metrics_rel = str(out_cfg.get("metrics_path", "artifacts/xgb/metrics_calibrated.json"))
    preds_calib_rel = str(out_cfg.get("preds_calibrated_path", "artifacts/xgb/preds_calibrated.parquet"))

    calibrator_path = (paths.repo_root / calibrator_rel).resolve()
    metrics_path = (paths.repo_root / metrics_rel).resolve()
    preds_calibrated_path = (paths.repo_root / preds_calib_rel).resolve()

    _ensure_parent(calibrator_path)
    _ensure_parent(metrics_path)
    _ensure_parent(preds_calibrated_path)

    if df_preds is None:
        if not preds_path.exists():
            raise FileNotFoundError(
                f"Missing preds.parquet at: {preds_path}\n"
                "Run training first (Step 04) to generate raw predictions."
            )
        df_preds = pd.read_parquet(preds_path)

    required = {"split", "label", "p_xgb"}
    missing = required - set(df_preds.columns)
    if missing:
        raise ValueError(f"preds.parquet missing required columns: {missing}")

    # Split
    train_df = df_preds[df_preds["split"] == train_name].copy()
    val_df = df_preds[df_preds["split"] == val_name].copy()
    test_df = df_preds[df_preds["split"] == test_name].copy()

    if len(val_df) == 0:
        raise ValueError("Validation split is empty. Cannot calibrate.")
    if val_df["label"].nunique() < 2:
        raise ValueError(
            "Validation split has only one class (all 0 or all 1). "
            "Platt/Isotonic calibration needs both classes."
        )

    x_val = val_df["p_xgb"].to_numpy(dtype=float)

    if method == "platt":
        X = x_val.reshape(-1, 1)
        y = val_df["label"].to_numpy(dtype=int)
        calibrator = LogisticRegression(solver="lbfgs")
        calibrator.fit(X, y)

        def predict_proba(x: np.ndarray) -> np.ndarray:
            return calibrator.predict_proba(x.reshape(-1, 1))[:, 1]

        calibrator_info = {
            "type": "LogisticRegression",
            "coef": calibrator.coef_.tolist(),
            "intercept": calibrator.intercept_.tolist(),
        }

    else:
        y = val_df["label"].to_numpy(dtype=int)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(x_val, y)

        def predict_proba(x: np.ndarray) -> np.ndarray:
            return calibrator.predict(x)

        calibrator_info = {"type": "IsotonicRegression"}

    def apply(df_: pd.DataFrame) -> pd.DataFrame:
        x = df_["p_xgb"].to_numpy(dtype=float)
        df_["p_calib"] = predict_proba(x)
        return df_

    train_df = apply(train_df)
    val_df = apply(val_df)
    test_df = apply(test_df)

    # Metrics (raw vs calibrated)
    metrics: Dict[str, Any] = {
        "calibration": {
            "method": method,
            "fitted_on": "val",
            "val_pos": int(val_df["label"].sum()),
            "val_neg": int((val_df["label"] == 0).sum()),
            "calibrator_info": calibrator_info,
        },
        "raw": {
            "val": _evaluate(val_df, "p_xgb"),
            "test": _evaluate(test_df, "p_xgb"),
        },
        "calibrated": {
            "val": _evaluate(val_df, "p_calib"),
            "test": _evaluate(test_df, "p_calib"),
        },
    }

    joblib.dump(calibrator, calibrator_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    df_out = df_preds.copy()
    df_out.loc[df_out["split"] == train_name, "p_calib"] = train_df["p_calib"].to_numpy()
    df_out.loc[df_out["split"] == val_name, "p_calib"] = val_df["p_calib"].to_numpy()
    df_out.loc[df_out["split"] == test_name, "p_calib"] = test_df["p_calib"].to_numpy()

    df_out.to_parquet(preds_calibrated_path, index=False)

    return CalibrateResult(
        calibrator_path=calibrator_path,
        metrics_path=metrics_path,
        preds_calibrated_path=preds_calibrated_path,
        metrics=metrics,
    )
