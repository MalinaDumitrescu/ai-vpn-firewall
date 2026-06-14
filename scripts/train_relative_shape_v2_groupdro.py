#!/usr/bin/env python3
"""
train_relative_shape_v2_groupdro.py
====================================

Experiment branch: relative_shape_v2 + GroupDRO

Feature family  – only normalized/ratio features (no absolute scales):
    sz_cv, sz_iqr_norm_median, sz_qratio, sz_median_to_mean,
    sz_p25_median_ratio, sz_p75_median_ratio,
    iat_cv, iat_iqr_norm_median, iat_qratio, iat_median_to_mean,
    relative_burstiness

Explicitly avoided:
    flow_duration, total_packets, total_bytes, packet_rate, byte_rate

Training phases
---------------
1. Classical balanced-bagging baselines: XGB, LGBM, CatBoost
2. GroupDRO-style iterative worst-domain reweighting (pure-Python,
   no extra dependencies) applied to LGBM as the representative learner.

Evaluation
----------
- Pooled AUC (test split)
- LODO-min AUC (leave-one-dataset-out, minimum across 3 folds)
- Per-domain AUC
- FPR at fixed thresholds
- ECE (Expected Calibration Error)

Success criteria (beats full_canonical__lgbm baseline)
-------------------------------------------------------
    LODO-min AUC > 0.6164
    OR domain-AUC std meaningfully lower while pooled AUC is acceptable

Outputs  →  artifacts/relative_shape_v2_groupdro/
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import load_paths
from src.utils.logging import setup_logger
from src.pipeline.data_preparation import load_and_prepare_data

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

LOG = logging.getLogger("relshape_v2_gdro")

# ---------------------------------------------------------------------------
# Feature definition
# ---------------------------------------------------------------------------
RELATIVE_SHAPE_V2_FEATURES: List[str] = [
    "sz_cv",
    "sz_iqr_norm_median",
    "sz_qratio",
    "sz_median_to_mean",
    "sz_p25_median_ratio",
    "sz_p75_median_ratio",
    "iat_cv",
    "iat_iqr_norm_median",
    "iat_qratio",
    "iat_median_to_mean",
    "relative_burstiness",
]

BASELINE_LODO_MIN_AUC = 0.6164  # full_canonical__lgbm

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_relative_shape_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive all relative_shape_v2 features from raw stat columns already
    present in the loaded features DataFrame.

    Expected raw columns (produced by extract_features_from_flows or cached):
        sz_coef_variation, sz_iqr_norm_median, sz_p25_median_ratio,
        sz_p75_median_ratio,
        iat_all_mean, iat_all_std, iat_all_p25, iat_all_median, iat_all_p75

    Or alternatively USBVPN-style column names. We handle both.
    """
    eps = 1e-9
    out = df.copy()

    # ── Size features ──────────────────────────────────────────────────────
    # sz_cv  (coef of variation)
    if "sz_coef_variation" in df.columns:
        out["sz_cv"] = df["sz_coef_variation"].astype(float)
    elif all(c in df.columns for c in ["sz_std", "sz_mean"]):
        out["sz_cv"] = df["sz_std"].astype(float) / (df["sz_mean"].astype(float) + eps)
    else:
        out["sz_cv"] = 0.0

    # sz_iqr_norm_median  (already computed in extract_features_from_flows)
    if "sz_iqr_norm_median" not in df.columns:
        if all(c in df.columns for c in ["sz_p75", "sz_p25", "sz_median"]):
            out["sz_iqr_norm_median"] = (
                (df["sz_p75"] - df["sz_p25"]) / (df["sz_median"].astype(float) + eps)
            )
        else:
            out["sz_iqr_norm_median"] = 0.0

    # sz_qratio  (Q3/Q1)
    if all(c in df.columns for c in ["sz_p75", "sz_p25"]):
        out["sz_qratio"] = df["sz_p75"].astype(float) / (df["sz_p25"].astype(float) + eps)
    else:
        out["sz_qratio"] = 1.0

    # sz_median_to_mean
    if all(c in df.columns for c in ["sz_median", "sz_mean"]):
        out["sz_median_to_mean"] = df["sz_median"].astype(float) / (df["sz_mean"].astype(float) + eps)
    elif all(c in df.columns for c in ["sz_p25_median_ratio", "sz_coef_variation"]):
        # Fallback: cannot derive; set neutral value
        out["sz_median_to_mean"] = 1.0
    else:
        out["sz_median_to_mean"] = 1.0

    # sz_p25_median_ratio / sz_p75_median_ratio  (already in extract output)
    for col in ("sz_p25_median_ratio", "sz_p75_median_ratio"):
        if col not in df.columns:
            out[col] = 0.0

    # ── IAT features ───────────────────────────────────────────────────────
    # Try both naming conventions: iat_all_* (VNAT/ISCX) and direct iat_* (USBVPN)
    def _iat_col(name: str) -> Optional[pd.Series]:
        for candidate in [f"iat_all_{name}", f"iat_{name}"]:
            if candidate in df.columns:
                return df[candidate].astype(float)
        return None

    iat_mean = _iat_col("mean")
    iat_std = _iat_col("std")
    iat_p25 = _iat_col("p25")
    iat_median = _iat_col("median")
    iat_p75 = _iat_col("p75")

    out["iat_cv"] = (iat_std / (iat_mean + eps)) if (iat_std is not None and iat_mean is not None) else 0.0

    out["iat_iqr_norm_median"] = (
        ((iat_p75 - iat_p25) / (iat_median + eps))
        if (iat_p75 is not None and iat_p25 is not None and iat_median is not None)
        else 0.0
    )

    out["iat_qratio"] = (
        (iat_p75 / (iat_p25 + eps))
        if (iat_p75 is not None and iat_p25 is not None)
        else 1.0
    )

    out["iat_median_to_mean"] = (
        (iat_median / (iat_mean + eps))
        if (iat_median is not None and iat_mean is not None)
        else 1.0
    )

    # relative_burstiness  = (iat_max - iat_median) / (iat_median + eps)
    iat_max = _iat_col("max") if _iat_col("max") is not None else None
    if iat_max is not None and iat_median is not None:
        out["relative_burstiness"] = (iat_max - iat_median) / (iat_median + eps)
    elif iat_std is not None and iat_mean is not None:
        # Approximation if max not available: use (mean + 2*std - median) / (median+eps)
        iat_approx_max = iat_mean + 2.0 * iat_std
        out["relative_burstiness"] = (
            (iat_approx_max - (iat_median if iat_median is not None else iat_mean))
            / ((iat_median if iat_median is not None else iat_mean) + eps)
        )
    else:
        out["relative_burstiness"] = 0.0

    # Clip / sanitize
    for col in RELATIVE_SHAPE_V2_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

    return out


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """QuantileTransform to [0,1] uniform fitted on train only."""
    n_quantiles = max(10, min(1000, len(df_train)))
    qt = QuantileTransformer(
        output_distribution="uniform",
        n_quantiles=n_quantiles,
        random_state=42,
    )
    X_tr = qt.fit_transform(df_train[feature_cols].to_numpy(float))
    X_va = qt.transform(df_val[feature_cols].to_numpy(float))
    X_te = qt.transform(df_test[feature_cols].to_numpy(float))
    return X_tr, X_va, X_te, qt


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece_val += mask.sum() * abs(acc - conf)
    return float(ece_val / len(y_true))


def fpr_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    pred = (y_prob >= threshold).astype(int)
    neg = y_true == 0
    if neg.sum() == 0:
        return 0.0
    return float((pred[neg] == 1).sum() / neg.sum())


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tag: str = "",
    threshold: float = 0.5,
) -> Dict:
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    ece_val = ece(y_true, y_prob)
    fpr = fpr_at_threshold(y_true, y_prob, threshold)
    LOG.info(
        f"[{tag}]  AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  "
        f"ECE={ece_val:.4f}  FPR@{threshold:.2f}={fpr:.4f}"
    )
    return {"auc": auc, "pr_auc": pr_auc, "ece": ece_val, f"fpr_at_{threshold}": fpr}


# ---------------------------------------------------------------------------
# Balanced-bagging helper
# ---------------------------------------------------------------------------

def balanced_bags(
    X: np.ndarray,
    y: np.ndarray,
    n_bags: int = 3,
    ratio: float = 1.0,
    seed: int = 42,
    sample_weights: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Return list of (X_bag, y_bag, w_bag) tuples."""
    rng = np.random.default_rng(seed)
    minority_idx = np.where(y == 1)[0]
    majority_idx = np.where(y == 0)[0]
    n_minority = len(minority_idx)
    n_majority = min(len(majority_idx), int(n_minority * ratio))

    bags = []
    for i in range(n_bags):
        maj_sample = rng.choice(majority_idx, size=n_majority, replace=False)
        idx = np.concatenate([minority_idx, maj_sample])
        rng.shuffle(idx)
        w = sample_weights[idx] if sample_weights is not None else None
        bags.append((X[idx], y[idx], w))
    return bags


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------

def train_xgb_bags(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    params: Dict,
    n_bags: int = 3,
    sample_weights: Optional[np.ndarray] = None,
):
    if xgb is None:
        raise ImportError("xgboost not installed")
    models = []
    bags = balanced_bags(X_tr, y_tr, n_bags=n_bags, sample_weights=sample_weights)
    for i, (Xb, yb, wb) in enumerate(bags):
        m = xgb.XGBClassifier(**params)
        fit_kw = dict(eval_set=[(X_va, y_va)], verbose=False)
        if wb is not None:
            fit_kw["sample_weight"] = wb
        m.fit(Xb, yb, **fit_kw)
        models.append(m)
    return models


def train_lgbm_bags(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    params: Dict,
    n_bags: int = 3,
    sample_weights: Optional[np.ndarray] = None,
):
    if lgb is None:
        raise ImportError("lightgbm not installed")
    models = []
    bags = balanced_bags(X_tr, y_tr, n_bags=n_bags, sample_weights=sample_weights)
    for i, (Xb, yb, wb) in enumerate(bags):
        m = lgb.LGBMClassifier(**params)
        fit_kw = dict(
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        if wb is not None:
            fit_kw["sample_weight"] = wb
        m.fit(Xb, yb, **fit_kw)
        models.append(m)
    return models


def train_catboost_bags(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    params: Dict,
    n_bags: int = 3,
    sample_weights: Optional[np.ndarray] = None,
):
    if cb is None:
        raise ImportError("catboost not installed")
    models = []
    bags = balanced_bags(X_tr, y_tr, n_bags=n_bags, sample_weights=sample_weights)
    for i, (Xb, yb, wb) in enumerate(bags):
        m = cb.CatBoostClassifier(**params)
        fit_kw = dict(eval_set=(X_va, y_va), verbose=False)
        if wb is not None:
            fit_kw["sample_weight"] = wb
        m.fit(Xb, yb, **fit_kw)
        models.append(m)
    return models


def predict_ensemble(models_dict: Dict[str, list], X: np.ndarray) -> np.ndarray:
    """Average predict_proba across all models in all families."""
    all_probs = []
    for family, models in models_dict.items():
        for m in models:
            p = m.predict_proba(X)[:, 1]
            all_probs.append(p)
    return np.mean(all_probs, axis=0)


# ---------------------------------------------------------------------------
# GroupDRO (iterative worst-domain reweighting)
# ---------------------------------------------------------------------------

def group_dro_lgbm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    params: Dict,
    n_rounds: int = 4,
    eta: float = 0.5,
    n_bags: int = 3,
) -> Tuple[list, list]:
    """
    Iterative worst-domain reweighting (approximation to GroupDRO).

    Algorithm:
      1. Initialize uniform domain weights w_g = 1/G for each domain g.
      2. For round t:
           a. Train LGBM bags with flow-level sample weights proportional to w_{g(i)}.
           b. Evaluate per-domain log-loss on train.
           c. Update w_g ∝ exp(eta * loss_g)  (exponentiated gradient ascent).
           d. Normalize w_g to sum to 1.
      3. Return final models and weight history.
    """
    from sklearn.metrics import log_loss

    if lgb is None:
        raise ImportError("lightgbm not installed")

    unique_groups = np.unique(groups_tr)
    G = len(unique_groups)
    w_g = np.ones(G) / G  # uniform init

    LOG.info(f"[GroupDRO] {G} domains, {n_rounds} rounds, eta={eta}")

    all_models = []
    weight_history = []

    for t in range(n_rounds):
        # Build per-sample weights from domain weights
        sample_w = np.ones(len(y_tr))
        for gi, g in enumerate(unique_groups):
            mask = groups_tr == g
            sample_w[mask] = w_g[gi]
        sample_w = sample_w / sample_w.mean()  # normalize to mean=1

        # Train bags
        round_models = train_lgbm_bags(
            X_tr, y_tr, X_va, y_va, params,
            n_bags=n_bags,
            sample_weights=sample_w,
        )
        all_models.append(round_models)

        # Evaluate per-domain log-loss on train
        y_pred_tr = np.mean(
            [m.predict_proba(X_tr)[:, 1] for m in round_models], axis=0
        )
        domain_losses = []
        for gi, g in enumerate(unique_groups):
            mask = groups_tr == g
            if mask.sum() < 2 or len(np.unique(y_tr[mask])) < 2:
                domain_losses.append(0.0)
            else:
                domain_losses.append(log_loss(y_tr[mask], y_pred_tr[mask]))

        domain_losses = np.array(domain_losses)
        LOG.info(
            f"[GroupDRO] round {t+1}/{n_rounds}  "
            + "  ".join(f"{g}={l:.4f}" for g, l in zip(unique_groups, domain_losses))
        )

        # Exponentiated gradient ascent on domain weights
        w_g = w_g * np.exp(eta * domain_losses)
        w_g = w_g / w_g.sum()
        weight_history.append({"round": t + 1, "domain_weights": dict(zip(unique_groups.tolist(), w_g.tolist()))})

    # Use last round's models
    final_models = all_models[-1]
    return final_models, weight_history


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def fit_calibration(y_va: np.ndarray, p_va: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, y_va)
    platt = LogisticRegression(C=1.0, max_iter=1000).fit(p_va.reshape(-1, 1), y_va)
    return iso, platt


def apply_calibration(iso, platt, p: np.ndarray):
    p_iso = iso.predict(p)
    p_platt = platt.predict_proba(p.reshape(-1, 1))[:, 1]
    return p_iso, p_platt


# ---------------------------------------------------------------------------
# LODO helpers
# ---------------------------------------------------------------------------

def make_lodo_df(df: pd.DataFrame, holdout: str) -> pd.DataFrame:
    other = df["dataset"] != holdout
    held = df["dataset"] == holdout

    parts = [
        df.loc[other & (df["split"] == "train")],
        df.loc[other & (df["split"] == "val")],
    ]
    held_df = df.loc[held].copy()
    held_df["split"] = "test"
    parts.append(held_df)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# One-fold training + evaluation
# ---------------------------------------------------------------------------

def run_fold(
    df_fold: pd.DataFrame,
    feature_cols: List[str],
    xgb_params: Dict,
    lgbm_params: Dict,
    cat_params: Dict,
    tag: str,
    use_groupdro: bool = True,
) -> Dict:
    train_df = df_fold[df_fold["split"] == "train"]
    val_df = df_fold[df_fold["split"] == "val"]
    test_df = df_fold[df_fold["split"] == "test"]

    LOG.info(
        f"[{tag}] train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )

    X_tr, X_va, X_te, qt = preprocess_features(train_df, val_df, test_df, feature_cols)
    y_tr = train_df["label"].to_numpy(int)
    y_va = val_df["label"].to_numpy(int)
    y_te = test_df["label"].to_numpy(int)

    # ── Baseline ensemble ──────────────────────────────────────────────────
    LOG.info(f"[{tag}] Training baseline ensemble ...")
    models_dict = {}
    if xgb is not None:
        models_dict["xgb"] = train_xgb_bags(X_tr, y_tr, X_va, y_va, xgb_params)
    if lgb is not None:
        models_dict["lgbm"] = train_lgbm_bags(X_tr, y_tr, X_va, y_va, lgbm_params)
    if cb is not None:
        models_dict["cat"] = train_catboost_bags(X_tr, y_tr, X_va, y_va, cat_params)

    p_va_raw = predict_ensemble(models_dict, X_va)
    p_te_raw = predict_ensemble(models_dict, X_te)

    iso, platt = fit_calibration(y_va, p_va_raw)
    p_va_iso, p_va_platt = apply_calibration(iso, platt, p_va_raw)
    p_te_iso, p_te_platt = apply_calibration(iso, platt, p_te_raw)

    result: Dict = {}
    for cal, p_va_c, p_te_c in [
        ("raw", p_va_raw, p_te_raw),
        ("isotonic", p_va_iso, p_te_iso),
        ("platt", p_va_platt, p_te_platt),
    ]:
        result[f"baseline/{cal}/val"] = compute_metrics(y_va, p_va_c, tag=f"{tag}/baseline/{cal}/val")
        result[f"baseline/{cal}/test"] = compute_metrics(y_te, p_te_c, tag=f"{tag}/baseline/{cal}/test")

        # Per-domain AUC on test
        domain_aucs = {}
        for ds in test_df["dataset"].unique():
            mask = (test_df["dataset"] == ds).to_numpy()
            if mask.sum() > 0 and len(np.unique(y_te[mask])) > 1:
                domain_aucs[ds] = float(roc_auc_score(y_te[mask], p_te_c[mask]))
        result[f"baseline/{cal}/test/domain_auc"] = domain_aucs

    # ── GroupDRO (LGBM-based) ──────────────────────────────────────────────
    if use_groupdro and lgb is not None:
        LOG.info(f"[{tag}] Training GroupDRO (LGBM) ...")
        groups_tr = train_df["dataset"].to_numpy(str)
        gdro_models, gdro_history = group_dro_lgbm(
            X_tr, y_tr, groups_tr, X_va, y_va, lgbm_params,
            n_rounds=4, eta=0.5, n_bags=3,
        )

        p_va_gdro = np.mean([m.predict_proba(X_va)[:, 1] for m in gdro_models], axis=0)
        p_te_gdro = np.mean([m.predict_proba(X_te)[:, 1] for m in gdro_models], axis=0)

        iso_g, platt_g = fit_calibration(y_va, p_va_gdro)
        p_va_giso, p_va_gplatt = apply_calibration(iso_g, platt_g, p_va_gdro)
        p_te_giso, p_te_gplatt = apply_calibration(iso_g, platt_g, p_te_gdro)

        for cal, p_va_c, p_te_c in [
            ("raw", p_va_gdro, p_te_gdro),
            ("isotonic", p_va_giso, p_te_giso),
            ("platt", p_va_gplatt, p_te_gplatt),
        ]:
            result[f"groupdro/{cal}/val"] = compute_metrics(y_va, p_va_c, tag=f"{tag}/gdro/{cal}/val")
            result[f"groupdro/{cal}/test"] = compute_metrics(y_te, p_te_c, tag=f"{tag}/gdro/{cal}/test")

            domain_aucs = {}
            for ds in test_df["dataset"].unique():
                mask = (test_df["dataset"] == ds).to_numpy()
                if mask.sum() > 0 and len(np.unique(y_te[mask])) > 1:
                    domain_aucs[ds] = float(roc_auc_score(y_te[mask], p_te_c[mask]))
            result[f"groupdro/{cal}/test/domain_auc"] = domain_aucs
        result["groupdro_weight_history"] = gdro_history

    return result


# ---------------------------------------------------------------------------
# Default hyperparameters (lightweight for this experiment)
# ---------------------------------------------------------------------------

def _default_xgb_params(paths) -> Dict:
    p = paths.artifacts_dir / "optuna_xgboost_firewall_best_params.json"
    if p.exists():
        with open(p) as f:
            params = json.load(f)
    else:
        params = {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8}
    params.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "booster": "gbtree",
        "tree_method": "hist",
        "n_estimators": 500,
        "random_state": 42,
        "n_jobs": 1,
        "early_stopping_rounds": 50,
        "verbosity": 0,
    })
    return params


def _default_lgbm_params(paths) -> Dict:
    p = paths.artifacts_dir / "optuna_lgbm_firewall_best_params.json"
    if p.exists():
        with open(p) as f:
            params = json.load(f)
    else:
        params = {"num_leaves": 63, "learning_rate": 0.05, "subsample": 0.8}
    params.update({
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": 1,
    })
    return params


def _default_cat_params(paths) -> Dict:
    p = paths.artifacts_dir / "optuna_catboost_firewall_best_params.json"
    if p.exists():
        with open(p) as f:
            params = json.load(f)
    else:
        params = {"depth": 6, "learning_rate": 0.05}
    params.update({
        "iterations": 500,
        "random_seed": 42,
        "thread_count": 1,
        "verbose": False,
        "allow_writing_files": False,
        "early_stopping_rounds": 100,
    })
    return params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_logger(level="INFO")
    paths = load_paths()

    out_dir: Path = paths.artifacts_dir / "relative_shape_v2_groupdro"
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("=" * 80)
    LOG.info("relative_shape_v2 + GroupDRO experiment")
    LOG.info("=" * 80)

    # 1. Load raw data
    LOG.info("Loading all 3 datasets ...")
    df_raw = load_and_prepare_data(vnat_only=False)
    LOG.info(f"Loaded {len(df_raw):,} flows  datasets={sorted(df_raw['dataset'].unique())}")
    LOG.info(f"Split × dataset:\n{pd.crosstab(df_raw['dataset'], df_raw['split'])}")

    # 2. Engineer relative_shape_v2 features
    LOG.info("Engineering relative_shape_v2 features ...")
    df = build_relative_shape_v2(df_raw)

    # Verify all features exist
    missing = [c for c in RELATIVE_SHAPE_V2_FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing relative_shape_v2 features after engineering: {missing}")
    LOG.info(f"Feature set: {RELATIVE_SHAPE_V2_FEATURES}")

    # Quick sanity: any NaN?
    nan_counts = {c: int(df[c].isna().sum()) for c in RELATIVE_SHAPE_V2_FEATURES}
    LOG.info(f"NaN counts per feature: {nan_counts}")

    # 3. Load hyperparams
    xgb_params = _default_xgb_params(paths)
    lgbm_params = _default_lgbm_params(paths)
    cat_params = _default_cat_params(paths)

    # 4. Pooled training (all 3 datasets)
    LOG.info("\n" + "=" * 60)
    LOG.info("PHASE 1: Pooled 3-dataset training")
    LOG.info("=" * 60)
    t0 = time.time()
    pooled_results = run_fold(
        df_fold=df,
        feature_cols=RELATIVE_SHAPE_V2_FEATURES,
        xgb_params=xgb_params,
        lgbm_params=lgbm_params,
        cat_params=cat_params,
        tag="POOLED",
        use_groupdro=True,
    )
    LOG.info(f"Pooled training done in {(time.time()-t0)/60:.1f} min")

    with open(out_dir / "pooled_results.json", "w") as f:
        json.dump(pooled_results, f, indent=2, default=float)

    # 5. LODO evaluation
    LOG.info("\n" + "=" * 60)
    LOG.info("PHASE 2: LODO evaluation")
    LOG.info("=" * 60)
    lodo_results = {}
    datasets = sorted(df["dataset"].unique())

    for holdout in datasets:
        LOG.info(f"\n--- LODO: hold_{holdout} ---")
        df_lodo = make_lodo_df(df, holdout)
        for split in ("train", "val", "test"):
            sub = df_lodo[df_lodo["split"] == split]
            if sub["label"].nunique() < 2 and split in ("val", "test"):
                raise RuntimeError(
                    f"LODO hold_{holdout}: split={split} lacks both classes"
                )

        t0 = time.time()
        fold_res = run_fold(
            df_fold=df_lodo,
            feature_cols=RELATIVE_SHAPE_V2_FEATURES,
            xgb_params=xgb_params,
            lgbm_params=lgbm_params,
            cat_params=cat_params,
            tag=f"LODO-hold_{holdout}",
            use_groupdro=True,
        )
        LOG.info(f"  Fold done in {(time.time()-t0)/60:.1f} min")
        lodo_results[holdout] = fold_res

        with open(out_dir / f"lodo_hold_{holdout}.json", "w") as f:
            json.dump(fold_res, f, indent=2, default=float)

    with open(out_dir / "lodo_all.json", "w") as f:
        json.dump(lodo_results, f, indent=2, default=float)

    # 6. Summary
    print("\n" + "=" * 88)
    print("RELATIVE_SHAPE_V2 + GroupDRO  —  RESULTS SUMMARY")
    print("=" * 88)

    print("\n── POOLED TEST (all 3 datasets) ──")
    for variant in ("baseline", "groupdro"):
        for cal in ("raw", "isotonic", "platt"):
            key = f"{variant}/{cal}/test"
            if key in pooled_results:
                m = pooled_results[key]
                da = pooled_results.get(f"{key}/domain_auc", {})
                da_str = "  ".join(f"{k}={v:.4f}" for k, v in sorted(da.items()))
                print(
                    f"  [{variant:9s}][{cal:8s}]  AUC={m['auc']:.4f}  "
                    f"PR-AUC={m['pr_auc']:.4f}  ECE={m['ece']:.4f}  {da_str}"
                )

    print("\n── LODO (trained on 2 datasets, tested on held-out) ──")
    for variant in ("baseline", "groupdro"):
        cal = "isotonic"
        holdout_aucs = []
        for holdout in datasets:
            key = f"{variant}/{cal}/test"
            if key in lodo_results.get(holdout, {}):
                a = lodo_results[holdout][key]["auc"]
                holdout_aucs.append(a)
                da = lodo_results[holdout].get(f"{key}/domain_auc", {})
                print(
                    f"  [{variant:9s}][{cal}]  hold_{holdout:7s}  "
                    f"holdout_AUC={a:.4f}  "
                    + "  ".join(f"{k}={v:.4f}" for k, v in sorted(da.items()))
                )
        if holdout_aucs:
            lodo_min = min(holdout_aucs)
            lodo_mean = float(np.mean(holdout_aucs))
            print(
                f"\n  [{variant:9s}][{cal}]  LODO-min={lodo_min:.4f}  "
                f"LODO-mean={lodo_mean:.4f}  "
                f"(baseline threshold: {BASELINE_LODO_MIN_AUC})"
            )
            beat = "✓ BEATS BASELINE" if lodo_min > BASELINE_LODO_MIN_AUC else "✗ does not beat baseline"
            print(f"  {beat}")

    print("\n── RECOMMENDATION ──")
    # Determine best LODO-min across variants
    best_variant = None
    best_lodo_min = -1.0
    for variant in ("baseline", "groupdro"):
        cal = "isotonic"
        aucs = []
        for holdout in datasets:
            key = f"{variant}/{cal}/test"
            if key in lodo_results.get(holdout, {}):
                aucs.append(lodo_results[holdout][key]["auc"])
        if aucs:
            lm = min(aucs)
            if lm > best_lodo_min:
                best_lodo_min = lm
                best_variant = variant

    if best_lodo_min > BASELINE_LODO_MIN_AUC:
        print(
            f"  → relative_shape_v2/{best_variant} achieves LODO-min={best_lodo_min:.4f} "
            f"> {BASELINE_LODO_MIN_AUC} — PROMOTE this model."
        )
    else:
        print(
            f"  → Best LODO-min={best_lodo_min:.4f} does NOT exceed {BASELINE_LODO_MIN_AUC}. "
            f"KEEP full_canonical__lgbm as final."
        )

    # Save compact summary
    summary = {
        "feature_family": "relative_shape_v2",
        "features": RELATIVE_SHAPE_V2_FEATURES,
        "baseline_lodo_min_auc": BASELINE_LODO_MIN_AUC,
        "best_variant": best_variant,
        "best_lodo_min_auc": best_lodo_min,
        "beats_baseline": best_lodo_min > BASELINE_LODO_MIN_AUC,
        "lodo_details": {},
    }
    for variant in ("baseline", "groupdro"):
        for holdout in datasets:
            key = f"{variant}/isotonic/test"
            if key in lodo_results.get(holdout, {}):
                summary["lodo_details"][f"{variant}/hold_{holdout}"] = lodo_results[holdout][key]

    pooled_key = "baseline/isotonic/test"
    if pooled_key in pooled_results:
        summary["pooled_test"] = pooled_results[pooled_key]

    with open(out_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    LOG.info(f"\nAll artifacts saved → {out_dir}")
    print(f"\nArtifacts: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
