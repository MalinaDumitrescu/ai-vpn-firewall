from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

from src.pipeline.artifacts import default_feature_artifacts
from src.pipeline.feature_pipeline import FeaturePipeline

import json
import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError(
        "xgboost is not installed. Add it to pyproject.toml / pip install xgboost."
    ) from e


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    metrics_path: Path
    preds_path: Path
    metrics: Dict[str, Any]


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def load_feature_columns(paths) -> list[str]:
    cols_path = paths.repo_root / "artifacts" / "features" / "feature_columns.json"
    if not cols_path.exists():
        raise FileNotFoundError(
            f"Missing feature_columns.json at: {cols_path}\n"
            "Run your feature pipeline export first (Step 3)."
        )

    cols = json.loads(cols_path.read_text(encoding="utf-8"))

    # Support either list or dict export formats
    if isinstance(cols, dict):
        if "model_feature_order" in cols:
            cols = cols["model_feature_order"]
        else:
            cols = cols.get("scale_cols", []) + cols.get("passthrough_cols", [])

    if not isinstance(cols, list) or not cols:
        raise ValueError(
            "feature_columns.json must be a non-empty JSON list "
            "(or dict with model_feature_order)."
        )
    return cols


def _basic_sanity(
    df: pd.DataFrame, *, label_col: str, split_col: str, feature_cols: list[str]
) -> None:
    needed = {label_col, split_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required cols: {missing}")

    feat_missing = set(feature_cols) - set(df.columns)
    if feat_missing:
        raise ValueError(
            f"features.parquet missing {len(feat_missing)} feature cols "
            f"(first 20): {sorted(list(feat_missing))[:20]}"
        )

    if df[label_col].isna().any():
        raise ValueError("Found NaNs in label column.")
    uniq = set(df[label_col].unique().tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(
            f"Label column must be binary 0/1. Found: {sorted(list(uniq))}"
        )

    X = df[feature_cols]
    if X.isna().any().any():
        bad = X.isna().sum().sort_values(ascending=False).head(10)
        raise ValueError(f"Found NaNs in features (top 10 cols):\n{bad}")
    arr = X.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Found inf/-inf values in feature matrix.")

    if df[split_col].isna().any():
        raise ValueError("Found NaNs in split column.")


def _fixed_fpr_threshold(
    y_true: np.ndarray, y_score: np.ndarray, fpr_target: float
) -> Dict[str, Any]:
    """
    Pick the highest threshold that keeps FPR <= fpr_target.
    If impossible, force predict-all-negative using a finite threshold > max(score).
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)

    idx = np.where(fpr <= fpr_target)[0]
    if len(idx) == 0:
        threshold = float(np.nextafter(np.max(y_score), np.inf))
    else:
        threshold = float(thr[int(idx[-1])])
        if not np.isfinite(threshold):
            threshold = float(np.nextafter(np.max(y_score), np.inf))

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    achieved_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "fpr_target": float(fpr_target),
        "fpr_achieved": float(achieved_fpr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_xgboost(
    *,
    paths,
    xgb_yaml: Path,
    df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[list[str]] = None,
) -> TrainResult:
    cfg = _load_yaml(xgb_yaml)

    seed = int(cfg.get("seed", 42))

    data_cfg = cfg.get("data") or {}
    dataset = str(data_cfg.get("dataset", "vnat"))
    features_filename = str(data_cfg.get("features_filename", "features.parquet"))
    label_col = str(data_cfg.get("label_col", "label"))
    split_col = str(data_cfg.get("split_col", "split"))

    training_cfg = cfg.get("training") or {}
    split_names = training_cfg.get("split_names") or {}
    train_name = str(split_names.get("train", "train"))
    val_name = str(split_names.get("val", "val"))
    test_name = str(split_names.get("test", "test"))

    use_scale_pos_weight = bool(training_cfg.get("use_scale_pos_weight", True))
    early_stopping_rounds = int(training_cfg.get("early_stopping_rounds", 200))
    num_boost_round = int(training_cfg.get("num_boost_round", 4000))
    verbose_eval = int(training_cfg.get("verbose_eval", 100))

    xgb_params = dict(cfg.get("xgb_params") or {})
    xgb_params["seed"] = seed

    # Ensure objective exists (your YAML already has it, but keep this safe)
    xgb_params.setdefault("objective", "binary:logistic")

    out_cfg = cfg.get("outputs") or {}
    model_path = (paths.repo_root / str(out_cfg.get("model_path", "artifacts/xgb/model.json"))).resolve()
    metrics_path = (paths.repo_root / str(out_cfg.get("metrics_path", "artifacts/xgb/metrics.json"))).resolve()
    preds_path = (paths.repo_root / str(out_cfg.get("preds_path", "artifacts/xgb/preds.parquet"))).resolve()

    _ensure_parent(model_path)
    _ensure_parent(metrics_path)
    _ensure_parent(preds_path)

    features_path = (paths.data_processed / dataset / features_filename).resolve()

    if df is None:
        if not features_path.exists():
            raise FileNotFoundError(f"Missing features file: {features_path}")
        df = pd.read_parquet(features_path)

    if feature_cols is None:
        feature_cols = load_feature_columns(paths)

    # sanity-check the RAW parquet still (NaNs, inf, label/split present, feature cols exist)
    _basic_sanity(df, label_col=label_col, split_col=split_col, feature_cols=feature_cols)

    # ---- NEW: load the feature pipeline and transform df (scaled + column enforced)
    feature_art = default_feature_artifacts(paths.artifacts_dir / "features")
    pipeline = FeaturePipeline.load(feature_art)

    # transform full df once; pipeline should drop non-feature cols internally
    X_all = pipeline.transform(df)

    # IMPORTANT: use the pipeline’s model feature order (matches eval)
    feature_cols = pipeline.model_feature_names()
    bad = {"label", "split", "flow_id", "capture_id", "file_names", "connection_str", "app"}
    leaks = sorted(set(feature_cols) & bad)
    if leaks:
        raise ValueError(f"LEAKAGE: model feature list contains forbidden cols: {leaks}")

    # splits (use indices so we can slice X_all safely)
    train_df = df[df[split_col] == train_name].copy()
    val_df = df[df[split_col] == val_name].copy()
    test_df = df[df[split_col] == test_name].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Empty split detected. sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        )

    train_idx = train_df.index
    val_idx = val_df.index
    test_idx = test_df.index

    X_train = X_all.loc[train_idx, feature_cols].to_numpy(dtype=float)
    y_train = train_df[label_col].to_numpy(dtype=int)

    X_val = X_all.loc[val_idx, feature_cols].to_numpy(dtype=float)
    y_val = val_df[label_col].to_numpy(dtype=int)

    X_test = X_all.loc[test_idx, feature_cols].to_numpy(dtype=float)
    y_test = test_df[label_col].to_numpy(dtype=int)

    # quick sanity: transformed features should be roughly standardized
    print("DEBUG mean/std (first 5):")
    print(np.round(X_train[:, :5].mean(axis=0), 3))
    print(np.round(X_train[:, :5].std(axis=0), 3))

    scale_pos_weight = None
    if use_scale_pos_weight:
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        if n_pos == 0:
            raise ValueError("Train split has 0 positive samples. Cannot train.")
        scale_pos_weight = n_neg / n_pos
        xgb_params["scale_pos_weight"] = float(scale_pos_weight)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    evals_result: Dict[str, Any] = {}
    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
        evals_result=evals_result,
    )

    booster.save_model(str(model_path))

    # Predict using best_iteration
    it_range = (0, int(booster.best_iteration) + 1)
    p_train = booster.predict(dtrain, iteration_range=it_range)
    p_val = booster.predict(dval, iteration_range=it_range)
    p_test = booster.predict(dtest, iteration_range=it_range)

    def pack_split(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
        roc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else None
        ap = average_precision_score(y, p) if len(np.unique(y)) > 1 else None

        yhat = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, yhat, average="binary", zero_division=0
        )

        return {
            "n": int(len(y)),
            "pos": int((y == 1).sum()),
            "neg": int((y == 0).sum()),
            "roc_auc": None if roc is None else float(roc),
            "pr_auc": None if ap is None else float(ap),
            "threshold_0.5": {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            },
        }

    # Best score can be str in some versions/metrics
    best_score_raw = booster.best_score
    try:
        best_score = float(best_score_raw) if best_score_raw is not None else None
    except (TypeError, ValueError):
        best_score = None

    metrics: Dict[str, Any] = {
        "dataset": dataset,
        "features_path": str(features_path),
        "xgb_yaml": str(xgb_yaml.resolve()),
        "seed": seed,
        "best_iteration": int(booster.best_iteration),
        "best_score": best_score,
        "scale_pos_weight": None if scale_pos_weight is None else float(scale_pos_weight),
        "feature_cols_n": int(len(feature_cols)),
        "feature_cols_hash": __import__("hashlib").sha256(",".join(feature_cols).encode("utf-8")).hexdigest(),
        "splits": {
            "train": pack_split(y_train, p_train),
            "val": pack_split(y_val, p_val),
            "test": pack_split(y_test, p_test),
        },
        "policy_thresholds": {
            "val_fpr_1pct": _fixed_fpr_threshold(y_val, p_val, fpr_target=0.01),
            "val_fpr_5pct": _fixed_fpr_threshold(y_val, p_val, fpr_target=0.05),
        },
        "evals_result": evals_result,
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Build preds with explicit index alignment
    preds = pd.DataFrame(index=df.index)
    preds["split"] = df[split_col].astype(str)
    preds["label"] = df[label_col].astype(int)

    for c in ["flow_id", "capture_id"]:
        if c in df.columns:
            preds[c] = df[c].astype(str)

    preds.loc[train_df.index, "p_xgb"] = p_train
    preds.loc[val_df.index, "p_xgb"] = p_val
    preds.loc[test_df.index, "p_xgb"] = p_test

    preds.reset_index(drop=True).to_parquet(preds_path, index=False)

    return TrainResult(
        model_path=model_path,
        metrics_path=metrics_path,
        preds_path=preds_path,
        metrics=metrics,
    )


