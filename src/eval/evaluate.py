# src/eval/evaluate.py

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.eval.metrics import group_metrics_by_split, select_policy_thresholds
from src.eval.calibration import ProbabilityCalibrator, fit_calibrator_from_df, calibration_report
from src.eval.plots import plot_roc_curves, plot_pr_curves, plot_confusion_matrices, plot_probability_distributions


try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("xgboost is not installed. pip install xgboost") from e


LEAKAGE_COLS = {
    "label",
    "split",
    "flow_id",
    "capture_id",
    "file_names",
    "connection_str",
    "app",
}


def load_feature_columns_from_artifacts(feature_columns_json: Path) -> List[str]:
    cols = json.loads(feature_columns_json.read_text(encoding="utf-8"))
    # supports your FeaturePipeline.save() dict format
    if isinstance(cols, dict):
        if "model_feature_order" in cols:
            cols = cols["model_feature_order"]
        else:
            cols = cols.get("scale_cols", []) + cols.get("passthrough_cols", [])

    if not isinstance(cols, list) or not cols:
        raise ValueError("feature_columns.json must be a non-empty list (or dict with model_feature_order).")

    leaks = sorted(set(cols) & LEAKAGE_COLS)
    if leaks:
        raise ValueError(f"Leakage columns found in feature columns list: {leaks}")

    return [str(c) for c in cols]


def _ensure_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(str(p))


def _predict_xgb(model_path: Path, df_features: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} feature columns in features df. Examples: {missing[:10]}")

    X = df_features[feature_cols].to_numpy(dtype=float, copy=False)
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    p = booster.predict(dmat)
    return np.asarray(p, dtype=float)


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    threshold = metrics.get("threshold", 0.5)
    print(f"Global Threshold: {threshold}")
    
    if "fixed_policy_thresholds" in metrics and metrics["fixed_policy_thresholds"]:
        fit_split = metrics.get("policy_fit_split", "val")
        print(f"Policy Thresholds (derived on '{fit_split}' split, applied unchanged to all splits):")
        for k, v in metrics["fixed_policy_thresholds"].items():
            print(f"  {k}: {v:.6f}")
    
    # Handle both "raw" (from evaluate_xgb) and "splits" (from train_xgboost)
    raw_metrics = metrics.get("raw") or metrics.get("splits")
    
    if raw_metrics:
        print("\n--- Raw Probabilities ---")
        for split, m in raw_metrics.items():
            print(f"\nSplit: {split.upper()}")
            print(f"  N: {m.get('n', 'N/A')} (Pos: {m.get('pos', 'N/A')}, Neg: {m.get('neg', 'N/A')})")
            print(f"  ROC AUC: {m.get('roc_auc', 'N/A')}")
            print(f"  PR AUC:  {m.get('pr_auc', 'N/A')}")
            
            # CHANGED: Use "firewall_policy" key instead of "threshold_0.5"
            thr_metrics = m.get("firewall_policy", {})
            if thr_metrics:
                actual_thr = thr_metrics.get('threshold', threshold)
                print(f"  At Firewall Threshold {actual_thr:.4f}:")
                print(f"    Precision: {thr_metrics.get('precision', 'N/A')}")
                print(f"    Recall:    {thr_metrics.get('recall', 'N/A')}")
                print(f"    F1 Score:  {thr_metrics.get('f1', 'N/A')}")
                cm = thr_metrics.get('confusion_matrix', {})
                print(f"    Confusion Matrix: TN={cm.get('tn')}, FP={cm.get('fp')}, FN={cm.get('fn')}, TP={cm.get('tp')}")

            # Optional: Print monitor threshold (e.g. fpr_0.01)
            pol_metrics = m.get("policy_thresholds", {})
            if "fpr_1pct" in pol_metrics:
                mon = pol_metrics["fpr_1pct"]
                print(f"  At Monitor Threshold (FPR ~1%) {mon.get('threshold', 'N/A'):.4f}:")
                print(f"    Recall: {mon.get('recall', 'N/A')}")
                cm = mon.get('confusion_matrix', {})
                print(f"    Confusion Matrix: TN={cm.get('tn')}, FP={cm.get('fp')}, FN={cm.get('fn')}, TP={cm.get('tp')}")


    if "calibrated" in metrics:
        print("\n--- Calibrated Probabilities ---")
        for split, m in metrics["calibrated"].items():
            print(f"\nSplit: {split.upper()}")
            print(f"  ROC AUC: {m.get('roc_auc', 'N/A')}")
            print(f"  PR AUC:  {m.get('pr_auc', 'N/A')}")
            print(f"  Brier:   {m.get('brier', 'N/A')}")
            print(f"  LogLoss: {m.get('logloss', 'N/A')}")

    print("="*60 + "\n")


@dataclass(frozen=True)
class EvalOutputs:
    preds_path: Path
    metrics_path: Path
    metrics: Dict[str, Any]


def evaluate_xgb(
    *,
    features_parquet: Path,
    model_path: Path,
    feature_columns_json: Path,
    out_dir: Path,
    threshold: float = 0.5,
    policy_fprs: tuple[float, ...] = (0.01, 0.05),
    policy_mode: str = "max_recall_under_fpr",
    calibrate: bool = False,
    calib_method: str = "platt",
    calib_fit_split: str = "val",
    calibrator_path: Optional[Path] = None,
    policy_fit_split: str = "val",
) -> EvalOutputs:
    _ensure_exists(features_parquet)
    _ensure_exists(model_path)
    _ensure_exists(feature_columns_json)

    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "preds.parquet"
    metrics_path = out_dir / "metrics_eval.json"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    df = pd.read_parquet(features_parquet)
    for c in ["split", "label", "flow_id", "capture_id"]:
        if c not in df.columns:
            raise ValueError(f"features.parquet missing required column: {c}")

    feature_cols = load_feature_columns_from_artifacts(feature_columns_json)
    p_raw = _predict_xgb(model_path, df, feature_cols)

    preds = df[["split", "label", "flow_id", "capture_id"]].copy()
    preds["p_raw"] = p_raw

    # Optional calibration
    cal: Optional[ProbabilityCalibrator] = None
    if calibrate:
        if calibrator_path is not None and calibrator_path.exists():
            cal = ProbabilityCalibrator.load(calibrator_path)
        else:
            cal = fit_calibrator_from_df(
                preds,
                prob_col="p_raw",
                label_col="label",
                split_col="split",
                fit_split=calib_fit_split,
                method=calib_method,  # "platt" or "isotonic"
            )
            if calibrator_path is not None:
                cal.save(calibrator_path)

        preds["p_calib"] = cal.predict(preds["p_raw"].to_numpy())

    preds.to_parquet(preds_path, index=False)

    # --- Policy Threshold Selection ---
    # Thresholds are derived on the validation split only and applied unchanged to test (and train) splits.
    # This prevents data leakage by ensuring test performance is evaluated with thresholds tuned on val.
    fixed_thresholds = None
    if policy_fit_split in preds["split"].unique():
        fixed_thresholds = select_policy_thresholds(
            preds,
            label_col="label",
            prob_col="p_raw",
            split_col="split",
            split_name=policy_fit_split,
            policy_fprs=policy_fprs,
            policy_mode=policy_mode
        )
    else:
        print(f"WARNING: Policy fit split '{policy_fit_split}' not found in data. "
              f"Policy thresholds will be computed per-split (potentially optimistic).")

    # Metrics
    metrics: Dict[str, Any] = {
        "inputs": {
            "features_parquet": str(features_parquet.resolve()),
            "model_path": str(model_path.resolve()),
            "feature_columns_json": str(feature_columns_json.resolve()),
        },
        "threshold": float(threshold),
        "policy_fprs": [float(x) for x in policy_fprs],
        "policy_mode": policy_mode,
        "policy_fit_split": policy_fit_split,
        "fixed_policy_thresholds": fixed_thresholds,
        "raw": group_metrics_by_split(
            preds.rename(columns={"p_raw": "p"}),
            label_col="label",
            prob_col="p",
            split_col="split",
            threshold=threshold,
            policy_fprs=policy_fprs,
            policy_mode=policy_mode,
            fixed_policy_thresholds=fixed_thresholds,
        ),
    }

    if calibrate and "p_calib" in preds.columns:
        fixed_calib_thresholds = None
        if policy_fit_split in preds["split"].unique():
            fixed_calib_thresholds = select_policy_thresholds(
                preds,
                label_col="label",
                prob_col="p_calib",
                split_col="split",
                split_name=policy_fit_split,
                policy_fprs=policy_fprs,
                policy_mode=policy_mode
            )

        metrics["calibrated"] = group_metrics_by_split(
            preds.rename(columns={"p_calib": "p"}),
            label_col="label",
            prob_col="p",
            split_col="split",
            threshold=threshold,
            policy_fprs=policy_fprs,
            policy_mode=policy_mode,
            fixed_policy_thresholds=fixed_calib_thresholds,
        )
        metrics["calibration_report"] = calibration_report(
            preds,
            raw_col="p_raw",
            calib_col="p_calib",
            label_col="label",
            split_col="split",
            threshold=threshold,
        )
        metrics["calibration"] = {
            "enabled": True,
            "method": calib_method,
            "fit_split": calib_fit_split,
            "calibrator_path": str(calibrator_path.resolve()) if calibrator_path else None,
        }
    else:
        metrics["calibration"] = {"enabled": False}

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # --- Plotting ---
    prob_col = "p_calib" if calibrate and "p_calib" in preds.columns else "p_raw"
    
    plot_roc_curves(preds, prob_col=prob_col, save_path=plots_dir / "roc_curves.png")
    plot_pr_curves(preds, prob_col=prob_col, save_path=plots_dir / "pr_curves.png")
    plot_confusion_matrices(preds, prob_col=prob_col, threshold=threshold, save_dir=plots_dir)
    plot_probability_distributions(preds, prob_col=prob_col, save_dir=plots_dir)

    # --- Console Output ---
    print_metrics_summary(metrics)

    return EvalOutputs(preds_path=preds_path, metrics_path=metrics_path, metrics=metrics)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate XGBoost model on a features.parquet file.")
    ap.add_argument("--features", type=str, required=True, help="Path to features.parquet")
    ap.add_argument("--model", type=str, required=True, help="Path to xgb model.json")
    ap.add_argument("--feature_cols", type=str, required=True, help="Path to artifacts/features/feature_columns.json")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for preds + metrics")

    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--policy_fprs", type=str, default="0.01,0.05", help="Comma-separated FPR targets (e.g. 0.01,0.05)")
    ap.add_argument("--policy_mode", type=str, default="max_recall_under_fpr", choices=["max_recall_under_fpr", "most_conservative"])
    ap.add_argument("--policy_fit_split", type=str, default="val", help="Split name to use for selecting policy thresholds (default: val)")

    ap.add_argument("--calibrate", action="store_true", help="Enable probability calibration")
    ap.add_argument("--calib_method", type=str, default="platt", choices=["platt", "isotonic"])
    ap.add_argument("--calib_fit_split", type=str, default="val")
    ap.add_argument("--calibrator_path", type=str, default="", help="Path to save/load calibrator.pkl (optional)")

    return ap


def main() -> None:
    args = build_argparser().parse_args()

    policy_fprs = tuple(float(x.strip()) for x in args.policy_fprs.split(",") if x.strip())
    calibrator_path = Path(args.calibrator_path) if args.calibrator_path.strip() else None

    res = evaluate_xgb(
        features_parquet=Path(args.features),
        model_path=Path(args.model),
        feature_columns_json=Path(args.feature_cols),
        out_dir=Path(args.out_dir),
        threshold=float(args.threshold),
        policy_fprs=policy_fprs,
        policy_mode=str(args.policy_mode),
        calibrate=bool(args.calibrate),
        calib_method=str(args.calib_method),
        calib_fit_split=str(args.calib_fit_split),
        calibrator_path=calibrator_path,
        policy_fit_split=str(args.policy_fit_split),
    )

    print("Saved preds :", res.preds_path)
    print("Saved metrics:", res.metrics_path)


if __name__ == "__main__":
    main()
