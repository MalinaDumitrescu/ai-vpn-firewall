from __future__ import annotations

import argparse
import json
import math
import textwrap
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeClassifier

from src.clean_pipeline.config import default_config
from src.clean_pipeline.feature_extractor import extract_flow_features
from src.clean_pipeline.feature_families import FEATURE_REGISTRY, FeatureSafety, get_family
from src.clean_pipeline.iscx_loader import iter_iscx_flows
from src.clean_pipeline.usbvpn_parser import iter_usbvpn_all_files
from src.clean_pipeline.vnat_loader import iter_vnat_flows


DATASETS: Tuple[str, ...] = ("iscx", "usbvpn", "vnat")
DEFAULT_FAMILY = "safe_core_plus_temporal"
DEFAULT_BOOTSTRAP = 200
EPS = 1e-12

SIGN_COLUMNS = {
    "diff_mean": "sign_mean_diff",
    "diff_median": "sign_median_diff",
    "cohen_d": "sign_smd",
    "cliffs_delta": "sign_cliff",
    "spearman_r": "sign_spearman",
    "pearson_r": "sign_pearson",
    "logistic_coef": "sign_logistic",
    "signed_auc": "sign_auc",
}

METRIC_THRESHOLDS = {
    "diff_mean": 0.0,
    "diff_median": 0.0,
    "cohen_d": 0.2,
    "cliffs_delta": 0.147,
    "spearman_r": 0.1,
    "pearson_r": 0.1,
    "logistic_coef": 0.0,
    "signed_auc": 0.1,
}

PLOT_TOP_K = 6


@dataclass
class AuditConfig:
    repo_root: Path
    output_dir: Path
    feature_family: str = DEFAULT_FAMILY
    max_packets: int = 300
    min_packets: int = 3
    seed: int = 42
    n_bootstrap: int = DEFAULT_BOOTSTRAP
    use_cache: bool = True
    force_recompute: bool = False
    include_full_length_check: bool = True
    compare_existing_clean_artifact: bool = True


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _sign(value: float, eps: float = 1e-12) -> int:
    if not np.isfinite(value) or abs(value) <= eps:
        return 0
    return 1 if value > 0 else -1


def _sign_label(value: float, eps: float = 1e-12) -> str:
    s = _sign(value, eps)
    return {1: "positive", -1: "negative", 0: "neutral"}[s]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _safe_series(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return arr


def _pooled_std(vpn: np.ndarray, nonvpn: np.ndarray) -> float:
    n1 = len(vpn)
    n0 = len(nonvpn)
    if n1 <= 1 or n0 <= 1:
        return 0.0
    s1 = float(np.var(vpn, ddof=1))
    s0 = float(np.var(nonvpn, ddof=1))
    denom = max(n1 + n0 - 2, 1)
    pooled = ((n1 - 1) * s1 + (n0 - 1) * s0) / denom
    return float(math.sqrt(max(pooled, 0.0)))


def _cohen_d(vpn: np.ndarray, nonvpn: np.ndarray) -> float:
    ps = _pooled_std(vpn, nonvpn)
    if ps <= EPS:
        return 0.0
    return float((vpn.mean() - nonvpn.mean()) / ps)


def _cliffs_delta(vpn: np.ndarray, nonvpn: np.ndarray) -> float:
    if len(vpn) == 0 or len(nonvpn) == 0:
        return 0.0
    try:
        u = stats.mannwhitneyu(vpn, nonvpn, alternative="two-sided", method="asymptotic").statistic
    except TypeError:
        u = stats.mannwhitneyu(vpn, nonvpn, alternative="two-sided").statistic
    return float((2.0 * u / (len(vpn) * len(nonvpn))) - 1.0)


def _safe_corr(fn, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return 0.0, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r, p = fn(x, y)
        except Exception:
            return 0.0, 1.0
    if not np.isfinite(r):
        r = 0.0
    if not np.isfinite(p):
        p = 1.0
    return float(r), float(p)


def _signed_auc(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return 0.5, 0.0
    auc = float(roc_auc_score(y, x))
    return auc, float((2.0 * auc) - 1.0)


def _fit_univariate_logistic(x: np.ndarray, y: np.ndarray, seed: int) -> Dict[str, Any]:
    result = {
        "logistic_coef": 0.0,
        "logistic_intercept": 0.0,
        "logistic_success": False,
        "logistic_scale_mode": "raw",
        "logistic_fit_note": "",
    }
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        result["logistic_fit_note"] = "constant_feature_or_single_class"
        return result

    x2 = x.reshape(-1, 1)
    try:
        model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
        model.fit(x2, y)
        result.update(
            logistic_coef=float(model.coef_[0, 0]),
            logistic_intercept=float(model.intercept_[0]),
            logistic_success=True,
        )
        return result
    except Exception as exc:
        pass

    x_mu = float(np.mean(x))
    x_std = float(np.std(x))
    x_std = max(x_std, EPS)
    try:
        model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
        model.fit(((x - x_mu) / x_std).reshape(-1, 1), y)
        coef = float(model.coef_[0, 0]) / x_std
        intercept = float(model.intercept_[0] - (model.coef_[0, 0] * x_mu / x_std))
        result.update(
            logistic_coef=coef,
            logistic_intercept=intercept,
            logistic_success=True,
            logistic_scale_mode="standardized_fallback",
            logistic_fit_note="fallback_standardized_then_backprojected",
        )
    except Exception as exc:
        result["logistic_fit_note"] = f"fit_failed:{type(exc).__name__}"
    return result


def _fit_decision_stump(x: np.ndarray, y: np.ndarray, seed: int) -> Dict[str, Any]:
    result = {
        "stump_auc": 0.5,
        "stump_threshold": np.nan,
        "stump_left_prob": np.nan,
        "stump_right_prob": np.nan,
        "stump_success": False,
    }
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return result
    try:
        clf = DecisionTreeClassifier(max_depth=1, random_state=seed)
        clf.fit(x.reshape(-1, 1), y)
        probs = clf.predict_proba(x.reshape(-1, 1))[:, 1]
        result.update(
            stump_auc=float(roc_auc_score(y, probs)),
            stump_threshold=float(clf.tree_.threshold[0]),
            stump_left_prob=float(clf.tree_.value[1][0, 1] / max(clf.tree_.value[1][0].sum(), 1.0)),
            stump_right_prob=float(clf.tree_.value[2][0, 1] / max(clf.tree_.value[2][0].sum(), 1.0)),
            stump_success=True,
        )
    except Exception:
        pass
    return result


def _effect_metrics_for_arrays(x: np.ndarray, y: np.ndarray, seed: int) -> Dict[str, Any]:
    vpn = x[y == 1]
    nonvpn = x[y == 0]

    mean_vpn = float(np.mean(vpn)) if len(vpn) else 0.0
    mean_nonvpn = float(np.mean(nonvpn)) if len(nonvpn) else 0.0
    median_vpn = float(np.median(vpn)) if len(vpn) else 0.0
    median_nonvpn = float(np.median(nonvpn)) if len(nonvpn) else 0.0
    pooled_std = _pooled_std(vpn, nonvpn)
    pooled_iqr = float(stats.iqr(x)) if len(x) else 0.0

    smd = _cohen_d(vpn, nonvpn)
    cliff = _cliffs_delta(vpn, nonvpn)
    spearman_r, spearman_p = _safe_corr(stats.spearmanr, x, y)
    pearson_r, pearson_p = _safe_corr(stats.pearsonr, x, y)
    auc, signed_auc = _signed_auc(x, y)
    logistic = _fit_univariate_logistic(x, y, seed)
    stump = _fit_decision_stump(x, y, seed)

    out = {
        "n_total": int(len(x)),
        "n_vpn": int((y == 1).sum()),
        "n_nonvpn": int((y == 0).sum()),
        "mean_vpn": mean_vpn,
        "mean_nonvpn": mean_nonvpn,
        "median_vpn": median_vpn,
        "median_nonvpn": median_nonvpn,
        "diff_mean": mean_vpn - mean_nonvpn,
        "diff_median": median_vpn - median_nonvpn,
        "cohen_d": smd,
        "cliffs_delta": cliff,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "auc": auc,
        "signed_auc": signed_auc,
        "pooled_std": pooled_std,
        "pooled_iqr": pooled_iqr,
        "mean_diff_weak_zone": pooled_std * 0.1,
        "median_diff_weak_zone": pooled_iqr * 0.1,
    }
    out.update(logistic)
    out.update(stump)
    for metric, sign_col in SIGN_COLUMNS.items():
        out[sign_col] = _sign(out[metric])
    return out


def _metric_reversal(signs: Iterable[int]) -> bool:
    signs = [int(s) for s in signs if int(s) != 0]
    return bool(signs and (1 in signs) and (-1 in signs))


def _ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": output_dir,
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "intermediate": output_dir / "intermediate",
        "preprocessing": output_dir / "tables" / "preprocessing_variants",
        "publication_tables": output_dir / "tables" / "publication",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# ---------------------------------------------------------------------------
# Raw feature reconstruction
# ---------------------------------------------------------------------------


def _extract_feature_row(
    flow: Dict[str, Any],
    *,
    dataset: str,
    label: int,
    family_features: Sequence[str],
    max_packets: Optional[int],
    source_file: str,
    capture_id: str,
    app: str,
    raw_label_source: str,
    raw_label_value: str,
    vpn_protocol: str = "",
) -> Optional[Dict[str, Any]]:
    ts = np.asarray(flow["timestamps"], dtype=np.float64)
    sz = np.asarray(flow["sizes"], dtype=np.float64)
    dr = np.asarray(flow["directions"], dtype=np.int32)
    n_full = min(len(ts), len(sz), len(dr))
    if n_full < 3:
        return None

    effective_max = n_full if max_packets is None else min(n_full, max_packets)
    feat_all = extract_flow_features(ts[:n_full], sz[:n_full], dr[:n_full], max_packets=effective_max)
    feat = {k: feat_all[k] for k in family_features}

    flow_id = str(flow["flow_id"])
    raw_duration_full = float(np.max(ts[:n_full]) - np.min(ts[:n_full])) if n_full >= 2 else 0.0
    row: Dict[str, Any] = {
        "flow_id": flow_id,
        "capture_id": str(capture_id),
        "dataset": dataset,
        "label": int(label),
        "source_file": str(source_file),
        "app": str(app),
        "vpn_protocol": str(vpn_protocol),
        "raw_label_source": raw_label_source,
        "raw_label_value": raw_label_value,
        "raw_packet_count_full": int(n_full),
        "window_packets_used": int(effective_max),
        "was_truncated": bool(max_packets is not None and n_full > max_packets),
        "raw_flow_duration_full": raw_duration_full,
    }
    row.update(feat)
    return row


def build_canonical_feature_table(cfg: AuditConfig, *, max_packets: Optional[int], cache_name: str) -> pd.DataFrame:
    paths = _ensure_dirs(cfg.output_dir)
    cache_path = paths["intermediate"] / cache_name
    if cfg.use_cache and not cfg.force_recompute and cache_path.exists():
        return pd.read_parquet(cache_path)

    clean_cfg = default_config()
    family_features = list(get_family(cfg.feature_family))
    rows: List[Dict[str, Any]] = []

    for flow in iter_vnat_flows(clean_cfg.vnat_h5, min_packets=cfg.min_packets):
        source_file = str(flow.get("source_file", ""))
        lower = source_file.lower()
        raw_label_value = "vpn" if lower.startswith("vpn") else "nonvpn" if lower.startswith("nonvpn") else "unknown"
        row = _extract_feature_row(
            flow,
            dataset="vnat",
            label=int(flow["label"]),
            family_features=family_features,
            max_packets=max_packets,
            source_file=source_file,
            capture_id=str(flow["capture_id"]),
            app=str(flow.get("app", "")),
            raw_label_source="file_name_prefix",
            raw_label_value=raw_label_value,
        )
        if row is not None:
            rows.append(row)

    for flow in iter_iscx_flows(clean_cfg.iscx_parquet, min_packets=cfg.min_packets):
        source_file = str(flow.get("source_file", ""))
        lower = source_file.lower()
        raw_prefix = "vpn" if lower.startswith("vpn") else "nonvpn" if lower.startswith("nonvpn") else "unknown"
        row = _extract_feature_row(
            flow,
            dataset="iscx",
            label=int(flow["label"]),
            family_features=family_features,
            max_packets=max_packets,
            source_file=source_file,
            capture_id=str(flow["capture_id"]),
            app=str(flow.get("app", "")),
            raw_label_source="explicit_label_column_plus_file_prefix",
            raw_label_value=f"label={int(flow['label'])};prefix={raw_prefix}",
        )
        if row is not None:
            rows.append(row)

    for flow, meta in iter_usbvpn_all_files(clean_cfg.usbvpn_raw_dir, min_packets=cfg.min_packets):
        source_file = str(meta.get("source_file", ""))
        raw_dir_label = source_file.split("/", 1)[0] if source_file else "unknown"
        flow = {**flow, "flow_id": meta["flow_id"]}
        row = _extract_feature_row(
            flow,
            dataset="usbvpn",
            label=int(meta["label"]),
            family_features=family_features,
            max_packets=max_packets,
            source_file=source_file,
            capture_id=str(meta["capture_id"]),
            app=str(meta.get("app", "")),
            raw_label_source="directory_name",
            raw_label_value=raw_dir_label,
            vpn_protocol=str(meta.get("vpn_protocol", "")),
        )
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Canonical raw feature table is empty.")

    if df["flow_id"].duplicated().any():
        df["flow_id"] = df["flow_id"].astype(str) + "_" + df.index.astype(str)

    df.to_parquet(cache_path, index=False)
    return df


def compare_with_existing_clean_artifact(cfg: AuditConfig, df: pd.DataFrame) -> pd.DataFrame:
    out_path = cfg.output_dir / "tables" / "recomputed_vs_existing_clean_artifact.csv"
    clean_path = cfg.repo_root / "artifacts" / "clean_pipeline" / "features.parquet"
    if not clean_path.exists():
        empty = pd.DataFrame([{ "status": "missing_clean_artifact", "artifact_path": str(clean_path) }])
        empty.to_csv(out_path, index=False)
        return empty

    existing = pd.read_parquet(clean_path)
    features = list(get_family(cfg.feature_family))
    join_cols = ["flow_id", "dataset", "capture_id", "label"]
    shared = df[join_cols + features].merge(
        existing[join_cols + features],
        on=join_cols,
        how="inner",
        suffixes=("_recomputed", "_existing"),
    )

    rows = []
    for feature in features:
        diff = (shared[f"{feature}_recomputed"] - shared[f"{feature}_existing"]).abs()
        rows.append(
            {
                "feature": feature,
                "n_matched_rows": int(len(shared)),
                "max_abs_diff": float(diff.max()) if len(diff) else np.nan,
                "mean_abs_diff": float(diff.mean()) if len(diff) else np.nan,
                "exact_match_rate": float((diff <= 1e-12).mean()) if len(diff) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


# ---------------------------------------------------------------------------
# Part 1: data inventory / quality / labels
# ---------------------------------------------------------------------------


def build_dataset_inventory(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inventory = (
        df.groupby("dataset")
        .agg(
            n_flows=("flow_id", "size"),
            n_captures=("capture_id", "nunique"),
            vpn_flows=("label", lambda s: int((s == 1).sum())),
            nonvpn_flows=("label", lambda s: int((s == 0).sum())),
            truncation_rate=("was_truncated", "mean"),
        )
        .reset_index()
    )
    inventory["vpn_proportion"] = inventory["vpn_flows"] / inventory["n_flows"]
    inventory["nonvpn_proportion"] = inventory["nonvpn_flows"] / inventory["n_flows"]

    balance = (
        df.groupby(["dataset", "label"]).size().rename("n_flows").reset_index()
        .assign(label_name=lambda d: d["label"].map({1: "VPN", 0: "nonVPN"}))
    )
    totals = balance.groupby("dataset")["n_flows"].transform("sum")
    balance["proportion"] = balance["n_flows"] / totals
    return inventory, balance


def build_label_mapping_report(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], str]:
    records: List[Dict[str, Any]] = []
    notes: List[str] = []
    fail_messages: List[str] = []

    for dataset in DATASETS:
        sub = df[df["dataset"] == dataset].copy()
        raw_values = sorted(map(str, sub["raw_label_value"].dropna().unique().tolist()))
        labels = sorted(sub["label"].dropna().unique().tolist())
        mapping_ok = set(labels) == {0, 1}

        if dataset == "vnat":
            mapping_note = "Binary label is derived from `file_names` / `source_file` prefix: `vpn* -> 1`, `nonvpn* -> 0`."
            prefix_ok = ((sub["source_file"].str.lower().str.startswith("vpn") & (sub["label"] == 1)) |
                         (sub["source_file"].str.lower().str.startswith("nonvpn") & (sub["label"] == 0))).all()
            if not prefix_ok:
                fail_messages.append("VNAT filename-prefix label mapping is inconsistent.")
        elif dataset == "iscx":
            mapping_note = "Binary label comes from the raw ISCX `label` column; file prefix was cross-checked for agreement."
            prefix_series = sub["source_file"].str.lower().map(
                lambda x: 1 if str(x).startswith("vpn") else 0 if str(x).startswith("nonvpn") else np.nan
            )
            prefix_ok = ((prefix_series.isna()) | (prefix_series.astype(float) == sub["label"].astype(float))).all()
            if not prefix_ok:
                fail_messages.append("ISCX explicit label and filename prefix disagree.")
        else:
            mapping_note = "Binary label is derived from USBVPN directory structure: `vpn/... -> 1`, `nonvpn/... -> 0`."
            dir_series = sub["source_file"].str.split("/").str[0].map({"vpn": 1, "nonvpn": 0})
            prefix_ok = (dir_series == sub["label"]).all()
            if not prefix_ok:
                fail_messages.append("USBVPN directory-based label mapping is inconsistent.")

        records.append(
            {
                "dataset": dataset,
                "unique_raw_label_values": raw_values,
                "unique_binary_values": labels,
                "vpn_binary_value": 1,
                "nonvpn_binary_value": 0,
                "mapping_ok": bool(mapping_ok and prefix_ok),
                "mapping_note": mapping_note,
            }
        )

    md_lines = ["# Label Mapping Audit", ""]
    for record in records:
        md_lines.extend(
            [
                f"## {record['dataset']}",
                f"- unique raw label values: `{record['unique_raw_label_values']}`",
                f"- binary values observed: `{record['unique_binary_values']}`",
                f"- VPN = `{record['vpn_binary_value']}`",
                f"- nonVPN = `{record['nonvpn_binary_value']}`",
                f"- mapping ok: **{record['mapping_ok']}**",
                f"- note: {record['mapping_note']}",
                "",
            ]
        )

    if fail_messages:
        md_lines.append("## FAILURES")
        for message in fail_messages:
            md_lines.append(f"- {message}")
        raise ValueError(" | ".join(fail_messages))

    return records, "\n".join(md_lines)


def build_feature_quality_report(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    dataset_dup = (
        df.groupby("dataset")
        .apply(
            lambda g: pd.Series(
                {
                    "duplicate_row_rate": float(g.duplicated(subset=list(features) + ["label", "capture_id"]).mean()),
                    "duplicate_flow_id_count": int(g["flow_id"].duplicated().sum()),
                }
            )
        )
        .reset_index()
    )

    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        sub = df[df["dataset"] == dataset].copy()
        dup_row = dataset_dup[dataset_dup["dataset"] == dataset].iloc[0]
        for feature in features:
            values = sub[feature]
            rows.append(
                {
                    "dataset": dataset,
                    "feature": feature,
                    "dtype": str(values.dtype),
                    "missing_count": int(values.isna().sum()),
                    "missing_rate": float(values.isna().mean()),
                    "inf_count": int(np.isinf(values.to_numpy(dtype=float)).sum()),
                    "n_unique": int(values.nunique(dropna=True)),
                    "variance": float(values.var(ddof=0)),
                    "near_constant": bool(values.nunique(dropna=True) <= 1 or float(values.var(ddof=0)) < 1e-12),
                    "duplicate_row_rate_dataset": float(dup_row["duplicate_row_rate"]),
                    "duplicate_flow_id_count_dataset": int(dup_row["duplicate_flow_id_count"]),
                }
            )
    out = pd.DataFrame(rows)
    dtype_summary = (
        out.groupby("feature")["dtype"].nunique().rename("dtype_count_across_datasets").reset_index()
    )
    out = out.merge(dtype_summary, on="feature", how="left")
    out["dtype_mismatch_across_datasets"] = out["dtype_count_across_datasets"] > 1
    return out


def build_truncation_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, sub in df.groupby("dataset"):
        for label, cls in sub.groupby("label"):
            rows.append(
                {
                    "dataset": dataset,
                    "label": int(label),
                    "label_name": "VPN" if int(label) == 1 else "nonVPN",
                    "n_flows": int(len(cls)),
                    "truncated_flows": int(cls["was_truncated"].sum()),
                    "truncation_rate": float(cls["was_truncated"].mean()),
                    "median_raw_packet_count": float(cls["raw_packet_count_full"].median()),
                    "median_window_packets_used": float(cls["window_packets_used"].median()),
                    "median_raw_flow_duration_full": float(cls["raw_flow_duration_full"].median()),
                    "median_window_flow_duration": float(cls["flow_duration"].median()),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 2/3/4/7: effects, preprocessing, balancing, models
# ---------------------------------------------------------------------------


def compute_effects_table(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    analysis_name: str,
    transform_name: str,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        sub = df[df["dataset"] == dataset].copy()
        y = sub["label"].to_numpy(dtype=int)
        for feature in features:
            x = sub[feature].to_numpy(dtype=float)
            metrics = _effect_metrics_for_arrays(x, y, seed)
            metrics.update(
                {
                    "analysis_name": analysis_name,
                    "transform_name": transform_name,
                    "dataset": dataset,
                    "feature": feature,
                }
            )
            rows.append(metrics)
    return pd.DataFrame(rows)


def save_sign_matrices(effects: pd.DataFrame, output_dir: Path, suffix: str = "") -> Dict[str, pd.DataFrame]:
    matrices: Dict[str, pd.DataFrame] = {}
    for metric, sign_col in SIGN_COLUMNS.items():
        mat = effects.pivot(index="feature", columns="dataset", values=sign_col).reindex(columns=list(DATASETS))
        matrices[metric] = mat
        name_map = {
            "diff_mean": "mean",
            "diff_median": "median",
            "cohen_d": "smd",
            "cliffs_delta": "cliff",
            "spearman_r": "spearman",
            "pearson_r": "pearson",
            "logistic_coef": "logistic",
            "signed_auc": "auc",
        }
        mat.to_csv(output_dir / f"feature_sign_matrix_{name_map[metric]}{suffix}.csv")
    return matrices


def summarize_reversals(effects: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for feature, sub in effects.groupby("feature"):
        row: Dict[str, Any] = {"feature": feature}
        loose_count = 0
        for metric, sign_col in SIGN_COLUMNS.items():
            signs = sub.set_index("dataset")[sign_col].reindex(list(DATASETS)).fillna(0).astype(int)
            row[f"{metric}_positive_datasets"] = int((signs > 0).sum())
            row[f"{metric}_negative_datasets"] = int((signs < 0).sum())
            row[f"{metric}_neutral_datasets"] = int((signs == 0).sum())
            reversal = _metric_reversal(signs.tolist())
            row[f"{metric}_reversal"] = bool(reversal)
            if reversal:
                loose_count += 1
        row["loose_reversal_metric_count"] = loose_count
        row["consensus_reversal"] = loose_count >= 3
        row["reversal_any_metric"] = loose_count > 0
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["consensus_reversal", "loose_reversal_metric_count", "feature"], ascending=[False, False, True])
    return out


def build_compact_sign_table(effects: pd.DataFrame, reversal_summary: pd.DataFrame) -> pd.DataFrame:
    smd = effects.set_index(["feature", "dataset"])["sign_smd"].unstack("dataset").reindex(columns=list(DATASETS))
    auc = effects.set_index(["feature", "dataset"])["sign_auc"].unstack("dataset").reindex(columns=list(DATASETS))
    out = smd.reset_index().rename(columns={ds: f"{ds}_smd_sign" for ds in DATASETS})
    out = out.merge(auc.reset_index().rename(columns={ds: f"{ds}_auc_sign" for ds in DATASETS}), on="feature", how="left")
    out = out.merge(reversal_summary[["feature", "reversal_any_metric", "consensus_reversal", "loose_reversal_metric_count"]], on="feature", how="left")
    return out


def apply_transform_variant(df: pd.DataFrame, features: Sequence[str], variant: str, seed: int) -> pd.DataFrame:
    out = df.copy()
    X = out[list(features)].astype(float).copy()

    def _zscore(frame: pd.DataFrame) -> pd.DataFrame:
        mu = frame.mean(axis=0)
        sigma = frame.std(axis=0, ddof=0).replace(0.0, 1.0)
        return (frame - mu) / sigma

    def _robust(frame: pd.DataFrame) -> pd.DataFrame:
        med = frame.median(axis=0)
        iqr = (frame.quantile(0.75) - frame.quantile(0.25)).replace(0.0, 1.0)
        return (frame - med) / iqr

    def _quantile(frame: pd.DataFrame) -> pd.DataFrame:
        qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(10, min(1000, len(frame))),
            random_state=seed,
        )
        arr = qt.fit_transform(frame.to_numpy(dtype=float))
        return pd.DataFrame(arr, columns=frame.columns, index=frame.index)

    if variant == "raw":
        return out
    if variant == "log1p":
        out[list(features)] = np.log1p(X.clip(lower=0.0))
        return out
    if variant == "global_zscore":
        out[list(features)] = _zscore(X)
        return out
    if variant == "per_dataset_zscore":
        out[list(features)] = (
            X.groupby(out["dataset"], group_keys=False).apply(_zscore)
        )
        return out
    if variant == "per_dataset_robust":
        out[list(features)] = (
            X.groupby(out["dataset"], group_keys=False).apply(_robust)
        )
        return out
    if variant == "global_quantile_normal":
        out[list(features)] = _quantile(X)
        return out
    if variant == "per_dataset_quantile_normal":
        out[list(features)] = (
            X.groupby(out["dataset"], group_keys=False).apply(_quantile)
        )
        return out
    raise ValueError(f"Unknown transform variant: {variant}")


def balance_flows_by_class(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for dataset, sub in df.groupby("dataset"):
        counts = sub["label"].value_counts()
        if len(counts) < 2:
            continue
        take = int(counts.min())
        for label, cls in sub.groupby("label"):
            idx = rng.choice(cls.index.to_numpy(), size=take, replace=False)
            rows.append(df.loc[idx])
    return pd.concat(rows, axis=0).sort_index().reset_index(drop=True)


def balance_captures_by_class(df: pd.DataFrame, features: Sequence[str], seed: int) -> pd.DataFrame:
    cap = (
        df.groupby(["dataset", "capture_id", "label"], as_index=False)[list(features)]
        .mean()
    )
    cap_meta = df.groupby(["dataset", "capture_id", "label"], as_index=False).agg(
        n_flows=("flow_id", "size"),
        app=("app", "first"),
        source_file=("source_file", "first"),
    )
    cap = cap.merge(cap_meta, on=["dataset", "capture_id", "label"], how="left")

    rng = np.random.default_rng(seed)
    rows = []
    for dataset, sub in cap.groupby("dataset"):
        counts = sub["label"].value_counts()
        if len(counts) < 2:
            continue
        take = int(counts.min())
        for label, cls in sub.groupby("label"):
            idx = rng.choice(cls.index.to_numpy(), size=take, replace=False)
            rows.append(cap.loc[idx])
    return pd.concat(rows, axis=0).sort_index().reset_index(drop=True)


def capture_balanced_table(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    cap = (
        df.groupby(["dataset", "capture_id", "label"], as_index=False)[list(features)]
        .mean()
    )
    cap_meta = df.groupby(["dataset", "capture_id", "label"], as_index=False).agg(
        app=("app", "first"),
        source_file=("source_file", "first"),
        n_flows=("flow_id", "size"),
    )
    return cap.merge(cap_meta, on=["dataset", "capture_id", "label"], how="left")


def build_capture_purity_report(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["dataset", "capture_id"], as_index=False)
        .agg(
            n_flows=("flow_id", "size"),
            vpn_flows=("label", lambda s: int((s == 1).sum())),
            nonvpn_flows=("label", lambda s: int((s == 0).sum())),
            app=("app", "first"),
            source_file=("source_file", "first"),
        )
    )
    grouped["purity"] = grouped[["vpn_flows", "nonvpn_flows"]].max(axis=1) / grouped["n_flows"]
    grouped["contains_both_classes"] = (grouped["vpn_flows"] > 0) & (grouped["nonvpn_flows"] > 0)
    grouped["dominant_class"] = np.where(grouped["vpn_flows"] >= grouped["nonvpn_flows"], "VPN", "nonVPN")
    return grouped


# ---------------------------------------------------------------------------
# Part 5/6: feature construction, direction semantics, class definitions
# ---------------------------------------------------------------------------


def build_feature_construction_audit(features: Sequence[str], max_packets: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for feature in features:
        spec = FEATURE_REGISTRY[feature]
        source_fields = set(spec.source_fields)
        size_based = "sizes" in source_fields
        time_based = "timestamps" in source_fields
        direction_based = "directions" in source_fields
        notes = [spec.notes]
        if size_based:
            notes.append("Packet sizes are converted to absolute values before feature computation.")
        if time_based:
            notes.append("Timestamps are sorted before IAT computation; negative diffs are clamped to 0.")
        if feature in {"flow_duration", "packet_rate", "byte_rate"}:
            notes.append("Duration is computed on the truncated analysis window, not the full original flow.")
        else:
            notes.append("Feature is computed after window truncation to the first max_packets packets.")
        rows.append(
            {
                "feature": feature,
                "formula": spec.formula,
                "source_fields": ", ".join(spec.source_fields),
                "feature_safety": spec.safety.value,
                "direction_safe": "yes" if spec.safety == FeatureSafety.SAFE and not direction_based else "no",
                "depends_on_direction_directly": "yes" if direction_based else "no",
                "depends_on_direction_indirectly": "no",
                "absolute_value_applied_to_sizes": "yes" if size_based else "no",
                "truncation_before_feature_computation": "yes",
                "window_max_packets": max_packets,
                "duration_window_based": "yes" if feature in {"flow_duration", "packet_rate", "byte_rate"} else "no",
                "computed_identically_across_datasets": "yes",
                "notes": " ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def direction_semantics_audit_text() -> str:
    return textwrap.dedent(
        """
        # Direction Semantics Audit

        ## Summary
        - `VNAT` and `ISCX` use canonical endpoint sorting when constructing bidirectional flows.
          In `src/flow/builder.py`, direction `1` means canonical `A -> B`, where endpoint `A` is the lexicographically smaller endpoint.
        - `USBVPN` does **not** use the same convention. In `src/clean_pipeline/usbvpn_parser.py`, direction is inferred from the sign of raw packet bytes:
          - positive bytes -> direction `1`
          - negative bytes -> direction `0`
        - These semantics are therefore **not interchangeable** across datasets.

        ## Why this matters
        - Any feature that uses raw forward/backward direction labels can pick up dataset-specific convention mismatch.
        - The audited 21-feature family (`safe_core_plus_temporal`) does **not** directly use packet direction. It uses packet counts, absolute sizes, and timing only.
        - Size features are made direction-safe by applying `abs()` before all computations.
        - Timing features are based on sorted timestamps only.

        ## Additional verified construction details
        - Window truncation happens **before** feature computation in `src/clean_pipeline/feature_extractor.py`.
        - `flow_duration`, `packet_rate`, and `byte_rate` are based on the truncated observation window, not the full flow.
        - The `configs/clean_pipeline.yaml` file contains `apply_quantile_scaling`, but `src/clean_pipeline/run_pipeline.py` does not apply any feature scaling before saving `features.parquet`.
          Therefore the clean artifact itself is raw feature-space, not scaled feature-space.

        ## Verdict
        For the 21-feature audit family, packet-direction convention mismatch is unlikely to explain sign reversal directly.
        Direction mismatch remains a serious risk for direction-labelled features outside the audited 21-feature family.
        """
    ).strip() + "\n"


def build_class_definition_audit(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    breakdown = (
        df.groupby(["dataset", "label", "app", "vpn_protocol"], dropna=False)
        .agg(n_flows=("flow_id", "size"), n_captures=("capture_id", "nunique"))
        .reset_index()
    )
    totals = breakdown.groupby(["dataset", "label"])["n_flows"].transform("sum")
    breakdown["proportion_within_dataset_class"] = breakdown["n_flows"] / totals
    breakdown["label_name"] = breakdown["label"].map({1: "VPN", 0: "nonVPN"})

    rows: List[Dict[str, Any]] = []
    for (dataset, label), sub in breakdown.groupby(["dataset", "label"]):
        sub_sorted = sub.sort_values("n_flows", ascending=False)
        top_apps = ", ".join(
            [f"{row.app}:{int(row.n_flows)}" for row in sub_sorted.head(5).itertuples()]
        )
        shares = sub_sorted["proportion_within_dataset_class"].to_numpy(dtype=float)
        entropy = float(stats.entropy(shares + EPS)) if len(shares) else 0.0
        rows.append(
            {
                "dataset": dataset,
                "label": int(label),
                "label_name": "VPN" if int(label) == 1 else "nonVPN",
                "n_flows": int(sub["n_flows"].sum()),
                "n_captures": int(sub["n_captures"].sum()),
                "n_fine_grained_labels": int(sub["app"].nunique()),
                "dominant_fine_grained_share": float(sub_sorted.iloc[0]["proportion_within_dataset_class"]) if len(sub_sorted) else 0.0,
                "shannon_entropy": entropy,
                "top_fine_grained_labels": top_apps,
                "notes": "USBVPN VPN traffic also varies by `vpn_protocol`; VNAT labels are filename-derived; ISCX labels are explicit.",
            }
        )
    return pd.DataFrame(rows), breakdown.sort_values(["dataset", "label", "n_flows"], ascending=[True, True, False])


# ---------------------------------------------------------------------------
# Part 8: bootstrap / strength
# ---------------------------------------------------------------------------


def _group_bootstrap_values(sub: pd.DataFrame, feature: str, seed: int, n_bootstrap: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    capture_groups = {
        lbl: [g[feature].to_numpy(dtype=float) for _, g in grp.groupby("capture_id")]
        for lbl, grp in sub.groupby("label")
    }
    if 0 not in capture_groups or 1 not in capture_groups:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for i in range(n_bootstrap):
        sampled_arrays = []
        sampled_labels = []
        for lbl in (0, 1):
            groups = capture_groups[lbl]
            picks = rng.integers(0, len(groups), size=len(groups))
            for idx in picks:
                arr = groups[int(idx)]
                sampled_arrays.append(arr)
                sampled_labels.append(np.full(arr.shape[0], lbl, dtype=int))
        x = np.concatenate(sampled_arrays).astype(float)
        y = np.concatenate(sampled_labels).astype(int)
        metrics = _effect_metrics_for_arrays(x, y, seed + i)
        for metric in SIGN_COLUMNS:
            rows.append(
                {
                    "bootstrap_id": i,
                    "metric": metric,
                    "estimate": float(metrics[metric]),
                }
            )
    return pd.DataFrame(rows)


def _strength_tag(metric: str, estimate: float, ci_low: float, ci_high: float, weak_zone: float) -> str:
    if ci_low <= 0.0 <= ci_high:
        return "neutral/uncertain"
    strong = abs(estimate) > weak_zone
    if estimate > 0:
        return "positive strong" if strong else "positive weak"
    return "negative strong" if strong else "negative weak"


def build_bootstrap_reports(df: pd.DataFrame, features: Sequence[str], seed: int, n_bootstrap: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []
    base_effects = compute_effects_table(df, features, analysis_name="all_flows", transform_name="raw", seed=seed)

    for dataset in DATASETS:
        sub = df[df["dataset"] == dataset].copy()
        for feature in features:
            boot = _group_bootstrap_values(sub, feature, seed=seed, n_bootstrap=n_bootstrap)
            if boot.empty:
                continue
            boot["dataset"] = dataset
            boot["feature"] = feature
            sample_rows.append(boot)

            effect_row = base_effects[(base_effects["dataset"] == dataset) & (base_effects["feature"] == feature)].iloc[0]
            for metric, metric_sub in boot.groupby("metric"):
                est = float(effect_row[metric])
                ci_low = float(metric_sub["estimate"].quantile(0.025))
                ci_high = float(metric_sub["estimate"].quantile(0.975))
                weak_zone = METRIC_THRESHOLDS[metric]
                if metric == "diff_mean":
                    weak_zone = float(effect_row["mean_diff_weak_zone"])
                elif metric == "diff_median":
                    weak_zone = float(effect_row["median_diff_weak_zone"])
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "feature": feature,
                        "metric": metric,
                        "estimate": est,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "weak_zone": weak_zone,
                        "strength_tag": _strength_tag(metric, est, ci_low, ci_high, weak_zone),
                    }
                )
    samples = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return samples, summary


def build_strict_loose_reversal_report(strength_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for feature, sub in strength_summary.groupby("feature"):
        row: Dict[str, Any] = {"feature": feature}
        loose_count = 0
        strict_count = 0
        for metric, metric_sub in sub.groupby("metric"):
            tags = metric_sub.set_index("dataset")["strength_tag"].reindex(list(DATASETS)).fillna("neutral/uncertain")
            signs = [1 if tag.startswith("positive") else -1 if tag.startswith("negative") else 0 for tag in tags]
            strong_signs = [
                1 if tag == "positive strong" else -1 if tag == "negative strong" else 0
                for tag in tags
            ]
            loose = _metric_reversal(signs)
            strict = _metric_reversal(strong_signs)
            row[f"{metric}_loose_reversal"] = bool(loose)
            row[f"{metric}_strict_reversal"] = bool(strict)
            if loose:
                loose_count += 1
            if strict:
                strict_count += 1
        row["loose_reversal_metric_count"] = loose_count
        row["strict_reversal_metric_count"] = strict_count
        row["loose_reversal"] = loose_count > 0
        row["strict_reversal"] = strict_count > 0
        row["consensus_reversal"] = strict_count >= 3
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["consensus_reversal", "strict_reversal_metric_count", "feature"], ascending=[False, False, True])


# ---------------------------------------------------------------------------
# Part 9/10: domain fingerprint + verdicts
# ---------------------------------------------------------------------------


def build_domain_fingerprint_report(
    df: pd.DataFrame,
    features: Sequence[str],
    strict_loose: pd.DataFrame,
) -> pd.DataFrame:
    capture_df = (
        df.groupby(["dataset", "capture_id"], as_index=False)[list(features)]
        .mean()
    )
    rows: List[Dict[str, Any]] = []
    for feature in features:
        pairwise_scores = []
        for ds_a, ds_b in combinations(DATASETS, 2):
            sub = capture_df[capture_df["dataset"].isin([ds_a, ds_b])].copy()
            y = (sub["dataset"] == ds_b).astype(int).to_numpy()
            x = sub[feature].to_numpy(dtype=float)
            if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
                auc = 0.5
            else:
                auc = float(roc_auc_score(y, x))
            pairwise_scores.append(max(auc, 1.0 - auc))
        rows.append(
            {
                "feature": feature,
                "domain_auc_pairwise_mean": float(np.mean(pairwise_scores)),
                "domain_auc_pairwise_min": float(np.min(pairwise_scores)),
                "domain_auc_pairwise_max": float(np.max(pairwise_scores)),
            }
        )
    out = pd.DataFrame(rows).merge(
        strict_loose[["feature", "loose_reversal_metric_count", "strict_reversal_metric_count", "consensus_reversal"]],
        on="feature",
        how="left",
    )
    out["absolute_effect_instability"] = out["strict_reversal_metric_count"]
    out["is_top5_domain"] = False
    out["is_top5_reversing"] = False
    top5_domain = set(out.nlargest(5, "domain_auc_pairwise_mean")["feature"])
    top5_rev = set(out.nlargest(5, "strict_reversal_metric_count")["feature"])
    out.loc[out["feature"].isin(top5_domain), "is_top5_domain"] = True
    out.loc[out["feature"].isin(top5_rev), "is_top5_reversing"] = True
    out["top5_overlap"] = out["feature"].isin(top5_domain & top5_rev)
    return out.sort_values(["domain_auc_pairwise_mean", "strict_reversal_metric_count"], ascending=[False, False])


def build_preprocessing_comparison(
    variant_summaries: Dict[str, pd.DataFrame],
    strict_loose_raw: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    rows = []
    raw_map = strict_loose_raw.set_index("feature")
    for feature in features:
        row = {"feature": feature}
        raw_rev = bool(raw_map.at[feature, "consensus_reversal"]) if feature in raw_map.index else False
        row["reversal_raw_space"] = raw_rev
        scaled_reversals = []
        for variant, summary in variant_summaries.items():
            if variant == "raw":
                continue
            is_rev = bool(summary.set_index("feature").at[feature, "consensus_reversal"])
            row[f"reversal_{variant}"] = is_rev
            scaled_reversals.append(is_rev)
        row["reversal_after_global_scaling"] = bool(row.get("reversal_global_zscore", False))
        row["reversal_after_per_dataset_scaling"] = bool(
            row.get("reversal_per_dataset_zscore", False) or row.get("reversal_per_dataset_robust", False)
        )
        row["reversal_introduced_only_after_scaling"] = (not raw_rev) and any(scaled_reversals)
        row["reversal_stable_across_preprocessing_variants"] = raw_rev and all(scaled_reversals)
        rows.append(row)
    return pd.DataFrame(rows)


def classify_feature_verdict(
    feature: str,
    strict_loose: pd.DataFrame,
    balancing: pd.DataFrame,
    preprocessing: pd.DataFrame,
    construction: pd.DataFrame,
) -> str:
    strict_row = strict_loose.set_index("feature").loc[feature]
    balance_row = balancing.set_index("feature").loc[feature]
    prep_row = preprocessing.set_index("feature").loc[feature]
    cons_row = construction.set_index("feature").loc[feature]

    raw_consensus = bool(strict_row["consensus_reversal"])
    strict_count = int(strict_row["strict_reversal_metric_count"])
    robust_balancing = all(
        bool(balance_row[col])
        for col in [
            "reversal_all_flows",
            "reversal_capture_balanced",
            "reversal_class_balanced",
            "reversal_capture_and_class_balanced",
        ]
    )
    scaling_artifact = bool(prep_row["reversal_introduced_only_after_scaling"])
    direction_risk = cons_row["direction_safe"] != "yes"

    if scaling_artifact or direction_risk:
        return "POSSIBLE ARTIFACT"
    if raw_consensus and robust_balancing and strict_count >= 3:
        return "VERIFIED REVERSAL"
    if raw_consensus and strict_count >= 2:
        return "LIKELY REVERSAL BUT SENSITIVE"
    if strict_count == 0 and not bool(strict_row["loose_reversal"]):
        return "NO REAL REVERSAL"
    return "INCONCLUSIVE"


def classify_thesis_verdict(final_verdict: pd.DataFrame) -> str:
    counts = final_verdict["final_category"].value_counts().to_dict()
    verified = counts.get("VERIFIED REVERSAL", 0)
    likely = counts.get("LIKELY REVERSAL BUT SENSITIVE", 0)
    artifact = counts.get("POSSIBLE ARTIFACT", 0)
    total = len(final_verdict)
    if verified >= max(6, total // 3) and artifact == 0:
        return "A. strongly supported"
    if verified + likely >= max(6, total // 3):
        return "B. partially supported"
    if artifact < total / 2:
        return "C. weak / unstable"
    return "D. likely artifact"


# ---------------------------------------------------------------------------
# Figures / report
# ---------------------------------------------------------------------------


def plot_heatmaps(matrices: Dict[str, pd.DataFrame], fig_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    sign_mat = matrices["cohen_d"].copy()
    plt.figure(figsize=(8, max(6, len(sign_mat) * 0.35)))
    sns.heatmap(sign_mat, cmap="coolwarm", center=0, cbar_kws={"label": "SMD sign"}, linewidths=0.5)
    plt.title("Raw-space sign direction by feature and dataset (SMD)")
    plt.tight_layout()
    plt.savefig(fig_dir / "heatmap_sign_direction_by_feature_dataset.png", dpi=180)
    plt.close()

    reversal_rows = []
    for metric, mat in matrices.items():
        reversal_rows.append({"metric": metric, **{f: int(_metric_reversal(mat.loc[f].fillna(0).tolist())) for f in mat.index}})
    reversal_df = pd.DataFrame(reversal_rows).set_index("metric").T
    plt.figure(figsize=(8, max(6, len(reversal_df) * 0.35)))
    sns.heatmap(reversal_df, cmap="viridis", vmin=0, vmax=1, cbar_kws={"label": "reversal"}, linewidths=0.5)
    plt.title("Reversal consensus across metrics")
    plt.tight_layout()
    plt.savefig(fig_dir / "heatmap_reversal_consensus_across_metrics.png", dpi=180)
    plt.close()


def plot_domain_scatter(domain_df: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        data=domain_df,
        x="domain_auc_pairwise_mean",
        y="strict_reversal_metric_count",
        hue="consensus_reversal",
        style="consensus_reversal",
        s=90,
    )
    for row in domain_df.itertuples():
        ax.text(row.domain_auc_pairwise_mean + 0.001, row.strict_reversal_metric_count + 0.05, row.feature, fontsize=8)
    plt.xlabel("Single-feature domain AUC (capture-balanced pairwise mean)")
    plt.ylabel("Strict reversal metric count")
    plt.title("Reversal strength vs domain informativeness")
    plt.tight_layout()
    plt.savefig(fig_dir / "scatter_reversal_strength_vs_domain_auc.png", dpi=180)
    plt.close()


def plot_top_feature_distributions(df: pd.DataFrame, final_verdict: pd.DataFrame, fig_dir: Path) -> None:
    top_features = final_verdict.sort_values(["strict_reversal_metric_count", "feature"], ascending=[False, True])["feature"].head(PLOT_TOP_K).tolist()
    if not top_features:
        return

    ncols = 2
    nrows = math.ceil(len(top_features) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, feature in zip(axes, top_features):
        sns.violinplot(data=df, x="dataset", y=feature, hue="label", split=False, inner="quart", ax=ax)
        ax.set_title(feature)
        ax.legend_.set_title("label")
    for ax in axes[len(top_features):]:
        ax.axis("off")
    fig.suptitle("Top reversing features: raw distributions by dataset and class", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "top_reversing_features_violin.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for feature in top_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="dataset", y=feature, hue="label")
        plt.title(f"Raw distribution: {feature}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"distribution_{feature}.png", dpi=180)
        plt.close()


def write_readme(output_dir: Path) -> None:
    text = textwrap.dedent(
        """
        # Sign Reversal Forensic Audit

        Generated by `src/eval/sign_reversal_forensic_audit.py`.

        Key files:
        - `sign_reversal_audit_report.md`
        - `sign_reversal_final_verdict.csv`
        - `tables/` for machine-readable outputs
        - `figures/` for thesis-ready figures

        Rerun command:
        ```powershell
        python -m src.eval.sign_reversal_forensic_audit --output-dir artifacts/sign_reversal_forensic_audit --bootstrap 200 --force-recompute
        ```
        """
    ).strip() + "\n"
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def write_markdown_report(
    cfg: AuditConfig,
    inventory: pd.DataFrame,
    mapping_records: List[Dict[str, Any]],
    compare_df: pd.DataFrame,
    truncation_df: pd.DataFrame,
    preprocessing_df: pd.DataFrame,
    balancing_df: pd.DataFrame,
    class_def_df: pd.DataFrame,
    strict_loose: pd.DataFrame,
    domain_df: pd.DataFrame,
    final_verdict: pd.DataFrame,
    thesis_verdict: str,
) -> None:
    report_path = cfg.output_dir / "sign_reversal_audit_report.md"
    verified = int((final_verdict["final_category"] == "VERIFIED REVERSAL").sum())
    likely = int((final_verdict["final_category"] == "LIKELY REVERSAL BUT SENSITIVE").sum())
    artifact = int((final_verdict["final_category"] == "POSSIBLE ARTIFACT").sum())
    none = int((final_verdict["final_category"] == "NO REAL REVERSAL").sum())
    inconclusive = int((final_verdict["final_category"] == "INCONCLUSIVE").sum())

    raw_rev = int(strict_loose["consensus_reversal"].sum())
    scaling_artifacts = preprocessing_df[preprocessing_df["reversal_introduced_only_after_scaling"]]["feature"].tolist()
    balance_stable = balancing_df[
        balancing_df[[
            "reversal_all_flows",
            "reversal_capture_balanced",
            "reversal_class_balanced",
            "reversal_capture_and_class_balanced",
        ]].all(axis=1)
    ]["feature"].tolist()

    inventory_md = inventory.to_markdown(index=False)
    final_md = final_verdict.to_markdown(index=False)
    class_md = class_def_df.to_markdown(index=False)

    executive = [
        "## Executive Summary",
        "",
        f"1. **Is the sign-reversal claim real?** {thesis_verdict}",
        f"2. **If yes, for how many features and under what definition?** Raw-space strict consensus reversal was found for **{raw_rev} / {len(final_verdict)}** features; **{verified}** were classified as `VERIFIED REVERSAL` and **{likely}** as `LIKELY REVERSAL BUT SENSITIVE`.",
        f"3. **If no / partly no, what caused the illusion?** Features flagged as scaling-only artifacts: `{scaling_artifacts}`.",
        f"4. **Can it safely be presented as a thesis contribution?** {('Yes, with caveats about class-definition heterogeneity and truncation.' if thesis_verdict.startswith('A') or thesis_verdict.startswith('B') else 'Not as a strong standalone claim without the listed caveats.')}",
        "",
    ]

    body = textwrap.dedent(
        f"""
        # Sign Reversal Forensic Audit Report

        This report was generated from a raw-source recomputation of the `{cfg.feature_family}` feature family.
        It is intentionally skeptical: prior notebook claims were treated as hypotheses, not evidence.

        {chr(10).join(executive)}

        ## Reproducibility
        - repo root: `{cfg.repo_root}`
        - output dir: `{cfg.output_dir}`
        - feature family: `{cfg.feature_family}`
        - max packets: `{cfg.max_packets}`
        - min packets: `{cfg.min_packets}`
        - bootstrap resamples: `{cfg.n_bootstrap}`
        - seed: `{cfg.seed}`

        ## Part 1 — Dataset Inventory

        {inventory_md}

        ### Label mapping checks
        {chr(10).join([f"- **{r['dataset']}**: mapping_ok={r['mapping_ok']} | raw_values={r['unique_raw_label_values']} | note={r['mapping_note']}" for r in mapping_records])}

        ### Recomputed vs existing clean artifact
        - This audit recomputed the 21-feature table directly from raw loaders and compared it with the existing clean artifact where possible.
        - Mean/max absolute differences per feature are saved to `tables/recomputed_vs_existing_clean_artifact.csv`.
        - Largest observed comparison row:\n\n{compare_df.sort_values('max_abs_diff', ascending=False).head(5).to_markdown(index=False) if 'max_abs_diff' in compare_df.columns else compare_df.to_markdown(index=False)}

        ### Truncation audit
        - Window truncation is active at `{cfg.max_packets}` packets.
        - Class- and dataset-specific truncation rates are saved to `tables/truncation_audit.csv`.

        {truncation_df.to_markdown(index=False)}

        ## Part 2 / 8 — Raw-space reversal strength
        - Strict consensus reversal = strong positive in at least one dataset and strong negative in another, under at least 3 metrics.
        - Loose reversal = any sign disagreement across datasets.

        ## Part 3 — Preprocessing sensitivity
        - The clean pipeline artifact itself is raw-space; the `apply_quantile_scaling` config option exists but is not applied in `src/clean_pipeline/run_pipeline.py`.
        - Preprocessing comparison results are saved to `tables/reversal_preprocessing_comparison.csv`.
        - Features where reversal appears only after scaling: `{scaling_artifacts}`.

        ## Part 4 — Capture-aware / balanced analysis
        - Features reversing under all four variants (all flows, capture-balanced, class-balanced, both): `{balance_stable}`.

        ## Part 5 — Direction / construction audit
        - The audited 21-feature family is direction-safe by construction.
        - Packet-direction convention mismatch remains relevant for direction-labelled features outside this family.

        ## Part 6 — Label integrity / class definition audit
        {class_md}

        ## Part 9 — Domain fingerprint relation
        - Single-feature domain informativeness was measured on capture-aggregated features via pairwise dataset AUC.
        - The joined per-feature table is saved to `tables/reversal_vs_domain_fingerprint.csv`.

        ## Part 10 — Final verdict
        - `VERIFIED REVERSAL`: {verified}
        - `LIKELY REVERSAL BUT SENSITIVE`: {likely}
        - `POSSIBLE ARTIFACT`: {artifact}
        - `NO REAL REVERSAL`: {none}
        - `INCONCLUSIVE`: {inconclusive}

        ### Thesis-level verdict
        **{thesis_verdict}**

        ### Feature-level verdict table
        {final_md}

        ## What could not be independently verified
        - True semantic ground truth of VPN/nonVPN beyond dataset-provided labels, directory structure, and filename conventions cannot be independently proven from packet contents alone in this audit.
        - Fine-grained application labels are dataset-specific and not fully harmonized across corpora.
        - The audit can show whether reversal persists under raw-space, balancing, and preprocessing checks; it cannot prove that differing application mixtures are irrelevant.
        """
    ).strip() + "\n"

    report_path.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_audit(cfg: AuditConfig) -> Dict[str, pd.DataFrame]:
    paths = _ensure_dirs(cfg.output_dir)
    write_readme(cfg.output_dir)
    features = list(get_family(cfg.feature_family))

    (cfg.output_dir / "audit_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=_json_default),
        encoding="utf-8",
    )

    df_raw = build_canonical_feature_table(
        cfg,
        max_packets=cfg.max_packets,
        cache_name=f"canonical_{cfg.feature_family}_{cfg.max_packets}.parquet",
    )
    df_raw.to_parquet(paths["intermediate"] / "canonical_feature_table_raw.parquet", index=False)

    compare_df = compare_with_existing_clean_artifact(cfg, df_raw) if cfg.compare_existing_clean_artifact else pd.DataFrame()

    inventory, balance = build_dataset_inventory(df_raw)
    mapping_records, mapping_md = build_label_mapping_report(df_raw)
    quality = build_feature_quality_report(df_raw, features)
    truncation = build_truncation_report(df_raw)

    inventory.to_csv(cfg.output_dir / "dataset_inventory.csv", index=False)
    balance.to_csv(cfg.output_dir / "class_balance_summary.csv", index=False)
    quality.to_csv(cfg.output_dir / "feature_quality_report.csv", index=False)
    truncation.to_csv(cfg.output_dir / "tables" / "truncation_audit.csv", index=False)
    (cfg.output_dir / "label_mapping_report.md").write_text(mapping_md, encoding="utf-8")

    effects_raw = compute_effects_table(df_raw, features, analysis_name="all_flows", transform_name="raw", seed=cfg.seed)
    effects_raw.to_csv(cfg.output_dir / "feature_effects_by_dataset.csv", index=False)
    raw_matrices = save_sign_matrices(effects_raw, cfg.output_dir)
    reversal_summary = summarize_reversals(effects_raw)
    reversal_summary.to_csv(cfg.output_dir / "feature_reversal_summary.csv", index=False)
    compact = build_compact_sign_table(effects_raw, reversal_summary)
    compact.to_csv(paths["publication_tables"] / "table_sign_directions_across_datasets.csv", index=False)

    variant_summaries: Dict[str, pd.DataFrame] = {"raw": reversal_summary}
    variant_effects_to_save: Dict[str, pd.DataFrame] = {"raw": effects_raw}
    for variant in [
        "raw",
        "log1p",
        "global_zscore",
        "per_dataset_zscore",
        "per_dataset_robust",
        "global_quantile_normal",
        "per_dataset_quantile_normal",
    ]:
        if variant == "raw":
            continue
        transformed = apply_transform_variant(df_raw, features, variant, cfg.seed)
        effects = compute_effects_table(transformed, features, analysis_name="all_flows", transform_name=variant, seed=cfg.seed)
        effects.to_csv(paths["preprocessing"] / f"feature_effects_by_dataset_{variant}.csv", index=False)
        save_sign_matrices(effects, paths["preprocessing"], suffix=f"_{variant}")
        summary = build_strict_loose_reversal_report(*build_bootstrap_reports(transformed, features, cfg.seed, min(50, cfg.n_bootstrap))[1:]) if False else summarize_reversals(effects)
        summary.to_csv(paths["preprocessing"] / f"feature_reversal_summary_{variant}.csv", index=False)
        variant_summaries[variant] = summary
        variant_effects_to_save[variant] = effects

    capture_purity = build_capture_purity_report(df_raw)
    capture_purity.to_csv(cfg.output_dir / "capture_purity_report.csv", index=False)

    capture_df = capture_balanced_table(df_raw, features)
    capture_effects = compute_effects_table(capture_df, features, analysis_name="capture_balanced", transform_name="raw", seed=cfg.seed)
    capture_effects.to_csv(cfg.output_dir / "capture_level_feature_effects.csv", index=False)
    capture_summary = summarize_reversals(capture_effects)

    class_balanced_df = balance_flows_by_class(df_raw, cfg.seed)
    class_effects = compute_effects_table(class_balanced_df, features, analysis_name="class_balanced", transform_name="raw", seed=cfg.seed)
    class_summary = summarize_reversals(class_effects)

    both_balanced_df = balance_captures_by_class(df_raw, features, cfg.seed)
    both_effects = compute_effects_table(both_balanced_df, features, analysis_name="capture_and_class_balanced", transform_name="raw", seed=cfg.seed)
    both_summary = summarize_reversals(both_effects)

    balancing = (
        reversal_summary[["feature", "consensus_reversal"]].rename(columns={"consensus_reversal": "reversal_all_flows"})
        .merge(capture_summary[["feature", "consensus_reversal"]].rename(columns={"consensus_reversal": "reversal_capture_balanced"}), on="feature")
        .merge(class_summary[["feature", "consensus_reversal"]].rename(columns={"consensus_reversal": "reversal_class_balanced"}), on="feature")
        .merge(both_summary[["feature", "consensus_reversal"]].rename(columns={"consensus_reversal": "reversal_capture_and_class_balanced"}), on="feature")
    )
    balancing["robustness_tag"] = np.select(
        [
            balancing[[
                "reversal_all_flows",
                "reversal_capture_balanced",
                "reversal_class_balanced",
                "reversal_capture_and_class_balanced",
            ]].all(axis=1),
            balancing[[
                "reversal_all_flows",
                "reversal_capture_balanced",
                "reversal_class_balanced",
                "reversal_capture_and_class_balanced",
            ]].sum(axis=1) >= 2,
        ],
        ["robust", "sensitive"],
        default="unstable",
    )
    balancing.to_csv(cfg.output_dir / "reversal_robustness_balancing.csv", index=False)

    construction = build_feature_construction_audit(features, cfg.max_packets)
    construction.to_csv(cfg.output_dir / "feature_construction_audit.csv", index=False)
    (cfg.output_dir / "direction_semantics_audit.md").write_text(direction_semantics_audit_text(), encoding="utf-8")

    class_def, fine_breakdown = build_class_definition_audit(df_raw)
    class_def.to_csv(cfg.output_dir / "class_definition_audit.csv", index=False)
    fine_breakdown.to_csv(cfg.output_dir / "fine_grained_label_breakdown.csv", index=False)

    model_report = effects_raw[
        [
            "dataset",
            "feature",
            "logistic_coef",
            "logistic_intercept",
            "logistic_success",
            "logistic_scale_mode",
            "logistic_fit_note",
            "auc",
            "signed_auc",
            "stump_auc",
            "stump_threshold",
            "stump_left_prob",
            "stump_right_prob",
            "stump_success",
        ]
    ].copy()
    model_report.to_csv(cfg.output_dir / "univariate_model_report.csv", index=False)

    bootstrap_samples, strength = build_bootstrap_reports(df_raw, features, cfg.seed, cfg.n_bootstrap)
    bootstrap_samples.to_csv(paths["intermediate"] / "bootstrap_metric_samples.csv", index=False)
    strength.to_csv(cfg.output_dir / "feature_effect_strength_report.csv", index=False)

    strict_loose = build_strict_loose_reversal_report(strength)
    strict_loose.to_csv(cfg.output_dir / "strict_vs_loose_reversal_report.csv", index=False)

    preprocessing = build_preprocessing_comparison(variant_summaries, strict_loose, features)
    preprocessing.to_csv(cfg.output_dir / "reversal_preprocessing_comparison.csv", index=False)

    domain = build_domain_fingerprint_report(df_raw, features, strict_loose)
    domain.to_csv(cfg.output_dir / "reversal_vs_domain_fingerprint.csv", index=False)

    final_verdict = strict_loose[["feature", "strict_reversal_metric_count", "loose_reversal_metric_count", "consensus_reversal"]].copy()
    final_verdict = final_verdict.merge(preprocessing, on="feature", how="left")
    final_verdict = final_verdict.merge(balancing, on="feature", how="left")
    final_verdict = final_verdict.merge(construction[["feature", "direction_safe", "computed_identically_across_datasets"]], on="feature", how="left")
    final_verdict["final_category"] = final_verdict["feature"].map(
        lambda f: classify_feature_verdict(f, strict_loose, balancing, preprocessing, construction)
    )
    thesis_verdict = classify_thesis_verdict(final_verdict)
    final_verdict["thesis_level_claim_verdict"] = thesis_verdict
    final_verdict.to_csv(cfg.output_dir / "sign_reversal_final_verdict.csv", index=False)

    publication_tables = {
        "table_robust_reversing_features.csv": final_verdict[final_verdict["final_category"] == "VERIFIED REVERSAL"],
        "table_preprocessing_artifact_candidates.csv": final_verdict[final_verdict["final_category"] == "POSSIBLE ARTIFACT"],
        "table_direction_semantics_risk_features.csv": construction[construction["direction_safe"] != "yes"],
        "table_raw_vs_scaled_reversal_comparison.csv": preprocessing,
    }
    for name, frame in publication_tables.items():
        frame.to_csv(paths["publication_tables"] / name, index=False)

    plot_heatmaps(raw_matrices, cfg.output_dir / "figures")
    plot_domain_scatter(domain, cfg.output_dir / "figures")
    plot_top_feature_distributions(df_raw, final_verdict, cfg.output_dir / "figures")

    if cfg.include_full_length_check:
        df_full = build_canonical_feature_table(
            cfg,
            max_packets=None,
            cache_name=f"canonical_{cfg.feature_family}_full_length.parquet",
        )
        effects_full = compute_effects_table(df_full, features, analysis_name="all_flows", transform_name="raw_full_length", seed=cfg.seed)
        full_summary = summarize_reversals(effects_full)
        full_summary = full_summary[["feature", "consensus_reversal", "loose_reversal_metric_count"]].rename(
            columns={
                "consensus_reversal": "reversal_full_length_raw",
                "loose_reversal_metric_count": "full_length_loose_reversal_metric_count",
            }
        )
        trunc_compare = preprocessing.merge(full_summary, on="feature", how="left")
        trunc_compare["reversal_changed_vs_full_length"] = trunc_compare["reversal_raw_space"] != trunc_compare["reversal_full_length_raw"]
        trunc_compare.to_csv(cfg.output_dir / "tables" / "truncation_sensitivity_report.csv", index=False)

    write_markdown_report(
        cfg=cfg,
        inventory=inventory,
        mapping_records=mapping_records,
        compare_df=compare_df,
        truncation_df=truncation,
        preprocessing_df=preprocessing,
        balancing_df=balancing,
        class_def_df=class_def,
        strict_loose=strict_loose,
        domain_df=domain,
        final_verdict=final_verdict,
        thesis_verdict=thesis_verdict,
    )

    return {
        "inventory": inventory,
        "effects_raw": effects_raw,
        "strict_loose": strict_loose,
        "final_verdict": final_verdict,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full forensic audit of cross-dataset feature sign reversal")
    parser.add_argument("--output-dir", type=str, default="artifacts/sign_reversal_forensic_audit")
    parser.add_argument("--family", type=str, default=DEFAULT_FAMILY)
    parser.add_argument("--max-packets", type=int, default=300)
    parser.add_argument("--min-packets", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip-full-length-check", action="store_true")
    parser.add_argument("--skip-clean-compare", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    cfg = AuditConfig(
        repo_root=repo_root,
        output_dir=(repo_root / args.output_dir).resolve(),
        feature_family=args.family,
        max_packets=args.max_packets,
        min_packets=args.min_packets,
        seed=args.seed,
        n_bootstrap=args.bootstrap,
        use_cache=not args.no_cache,
        force_recompute=args.force_recompute,
        include_full_length_check=not args.skip_full_length_check,
        compare_existing_clean_artifact=not args.skip_clean_compare,
    )
    run_audit(cfg)


if __name__ == "__main__":
    main()
