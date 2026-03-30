from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json
import pandas as pd
import yaml

from src.eval.calibration import fit_calibrator_from_df, calibration_report


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


def calibrate_xgb_predictions(
    *,
    paths,
    calib_yaml: Path,
    df_preds: Optional[pd.DataFrame] = None,
) -> CalibrateResult:
    """
    Calibrate XGBoost probabilities (p_raw/p_xgb) using validation split only.
    Uses src.eval.calibration.ProbabilityCalibrator for consistency.

    Input:
      artifacts/xgb/preds.parquet with columns:
        split, label, p_raw (or p_xgb) (+ optional flow_id, capture_id)

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
    val_name = str(splits.get("val", "val"))
    fallback_fit_splits = cfg.get("fallback_fit_splits", ["train"])
    if fallback_fit_splits is None:
        fallback_fit_splits = []
    fallback_fit_splits = [str(s) for s in fallback_fit_splits]

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

    # Determine prob column
    prob_col = "p_raw"
    if "p_raw" not in df_preds.columns:
        if "p_xgb" in df_preds.columns:
            prob_col = "p_xgb"
        else:
            raise ValueError("preds.parquet missing probability column (p_raw or p_xgb)")

    required = {"split", "label", prob_col}
    missing = required - set(df_preds.columns)
    if missing:
        raise ValueError(f"preds.parquet missing required columns: {missing}")

    # 1. Fit Calibrator on Validation Split
    calibrator = fit_calibrator_from_df(
        df_preds,
        prob_col=prob_col,
        label_col="label",
        split_col="split",
        fit_split=val_name,
        method=method,
        fallback_splits=fallback_fit_splits,
    )

    # 2. Apply Calibration to All Splits
    df_out = df_preds.copy()
    df_out["p_calib"] = calibrator.predict(df_out[prob_col].to_numpy())

    # 3. Compute Metrics
    metrics = calibration_report(
        df_out,
        raw_col=prob_col,
        calib_col="p_calib",
        label_col="label",
        split_col="split",
        threshold=0.5,
    )
    
    # Add metadata about the calibration run
    metrics["calibration_info"] = {
        "method": method,
        "fit_split_requested": val_name,
        "fit_split_used": calibrator.metadata.get("fit_split_used", val_name),
        "fallback_used": calibrator.metadata.get("fallback_used", False),
        "candidate_splits": calibrator.metadata.get("candidate_splits", [val_name]),
        "class_counts_fit": calibrator.metadata.get("class_counts", {}),
        "n_samples_fit": calibrator.metadata.get("n_samples", 0),
        "calibrator_path": str(calibrator_path),
        "input_prob_col": prob_col,
    }

    # 4. Save Artifacts
    calibrator.save(calibrator_path, extra_metadata={"source": "xgb_calibrate.py"})
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    df_out.to_parquet(preds_calibrated_path, index=False)

    return CalibrateResult(
        calibrator_path=calibrator_path,
        metrics_path=metrics_path,
        preds_calibrated_path=preds_calibrated_path,
        metrics=metrics,
    )
