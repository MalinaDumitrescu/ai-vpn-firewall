#!/usr/bin/env python
"""tools/build_frontend_demo_csvs.py

Builds two frontend-ready demo CSVs from processed feature files (ISCX, VNAT, USBVPN):
  1. frontend_demo_robust9.csv    — 9 robust9 features + passthrough columns
  2. frontend_demo_multimodel.csv — 35 multimodel columns + passthrough columns

These CSV files are curated demo samples from the processed ISCX, VNAT, and USBVPN datasets.
They are NOT live VM captures. They are intended to test and demonstrate model behaviour on
training-distribution-like VPN and non-VPN feature rows.  Live VM captures are demonstrated
separately using basic/vpnlike/warp/openvpnlab profiles.

Usage:
  python tools/build_frontend_demo_csvs.py
  python tools/build_frontend_demo_csvs.py --rows-per-group 100
  python tools/build_frontend_demo_csvs.py --output-dir exports/frontend_demo
  python tools/build_frontend_demo_csvs.py --score-with-models
  python tools/build_frontend_demo_csvs.py --allow-unknown

Outputs:
  exports/frontend_demo/frontend_demo_robust9.csv
  exports/frontend_demo/frontend_demo_multimodel.csv
  exports/frontend_demo/frontend_demo_report.txt
  exports/frontend_demo/frontend_demo_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root — assumed to be two levels up from this file (tools/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

ROBUST9_FEATURES: list[str] = [
    "sz_all_mean",
    "sz_cv",
    "sz_all_p25",
    "sz_all_median",
    "sz_all_p75",
    "sz_mean_max",
    "sz_mean_min",
    "sz_std_max",
    "sz_std_min",
]

MULTIMODEL_FEATURES: list[str] = [
    "direction_balance_bytes",
    "direction_balance_packets",
    "dispersion_symmetry",
    "iat_all_mean",
    "iat_all_median",
    "iat_all_p25",
    "iat_all_p75",
    "iat_all_std",
    "iat_mean_max",
    "iat_mean_min",
    "iat_std_max",
    "iat_std_min",
    "session_consecutive_high_runs",
    "session_fraction_high",
    "session_mean_prob",
    "session_top_k_mean_prob",
    "session_var_prob",
    "sz_all_mean",
    "sz_all_median",
    "sz_all_p25",
    "sz_all_p75",
    "sz_all_std",
    "sz_coef_variation",
    "sz_cv",
    "sz_iqr_norm_median",
    "sz_mean_max",
    "sz_mean_min",
    "sz_p25_median_ratio",
    "sz_p75_median_ratio",
    "sz_std_max",
    "sz_std_min",
]

PASSTHROUGH_COLS: list[str] = ["session_id", "flow_id", "dataset", "label"]

ROBUST9_OUTPUT_COLS: list[str] = PASSTHROUGH_COLS + ROBUST9_FEATURES
MULTIMODEL_OUTPUT_COLS: list[str] = PASSTHROUGH_COLS + MULTIMODEL_FEATURES

# ---------------------------------------------------------------------------
# Column aliases: canonical -> [candidate source names in priority order]
# ---------------------------------------------------------------------------
COLUMN_ALIASES: dict[str, list[str]] = {
    # size coefficient of variation
    "sz_cv": ["sz_cv", "sz_coef_variation", "pkt_len_cv", "size_cv", "pkt_size_cv"],
    "sz_coef_variation": ["sz_coef_variation", "sz_cv", "pkt_len_cv", "size_cv", "pkt_size_cv"],
    # all-direction packet size statistics
    "sz_all_mean": ["sz_all_mean", "mean_pkt_len"],
    "sz_all_std": ["sz_all_std", "std_pkt_len"],
    "sz_all_median": ["sz_all_median", "median_pkt_len"],
    "sz_all_p25": ["sz_all_p25", "p25_pkt_len"],
    "sz_all_p75": ["sz_all_p75", "p75_pkt_len"],
    # inter-arrival time (bulk statistics name variations)
    "iat_all_mean": ["iat_all_mean", "iat_mean", "inter_arrival_mean"],
    "iat_all_std": ["iat_all_std", "iat_std", "inter_arrival_std"],
    "iat_all_median": ["iat_all_median", "iat_median", "inter_arrival_median"],
    "iat_all_p25": ["iat_all_p25", "iat_p25"],
    "iat_all_p75": ["iat_all_p75", "iat_p75"],
}

# Session-derived features that do NOT exist in raw feature files —
# they come from second-stage session modeling.  We always fill them 0.0
# for sources that lack them and record a warning.
SESSION_FILL_COLS: list[str] = [
    "session_consecutive_high_runs",
    "session_fraction_high",
    "session_mean_prob",
    "session_top_k_mean_prob",
    "session_var_prob",
]

# ---------------------------------------------------------------------------
# Dataset name detection from path / column values
# ---------------------------------------------------------------------------
DATASET_PATTERNS: dict[str, list[str]] = {
    "iscx": ["iscx"],
    "vnat": ["vnat"],
    "usbvpn": ["usbvpn", "usb-vpn", "usb_vpn"],
}

# Label normalisation
VPN_INDICATORS = {
    "vpn", "openvpn", "wireguard", "tor", "tunnel", "vpnlike", "vpn-like",
    "1", "true",
}
NONVPN_INDICATORS = {
    "nonvpn", "non-vpn", "non_vpn", "benign", "normal", "regular",
    "non_tunnel", "nontunnel", "0", "false",
}

# ---------------------------------------------------------------------------
# Search paths (relative to project root)
# ---------------------------------------------------------------------------
SEARCH_DIRS: list[str] = [
    "data/processed",
    "artifacts/clean_pipeline",
    "artifacts/clean_pipeline_test_iscx",
    "exports/app_runtime_bundle/demo_data",
    "data/splits",
    "data",
    "artifacts/features",
    "captures",
    "datasets",
    "processed",
]

# ---------------------------------------------------------------------------
# Logging / warning accumulator
# ---------------------------------------------------------------------------

_warnings: list[str] = []
_report_lines: list[str] = []


def _log(msg: str, *, warn: bool = False) -> None:
    print(msg)
    _report_lines.append(msg)
    if warn:
        _warnings.append(msg)


# ===========================================================================
# Helper functions
# ===========================================================================


def detect_dataset_from_path(path: Path) -> str | None:
    """Return 'iscx', 'vnat', or 'usbvpn' from the file path string, else None."""
    path_lower = str(path).lower()
    for name, patterns in DATASET_PATTERNS.items():
        if any(p in path_lower for p in patterns):
            return name
    return None


def detect_dataset_from_df(df: pd.DataFrame) -> str | None:
    """Return dataset name from an existing 'dataset'/'source'/'dataset_name' column."""
    for col in ("dataset", "source", "dataset_name"):
        if col in df.columns:
            vals = df[col].dropna().unique()
            for v in vals:
                v_lower = str(v).lower()
                for name, patterns in DATASET_PATTERNS.items():
                    if any(p in v_lower for p in patterns):
                        return name
    return None


def normalize_label(raw_value: Any, path_hint: str = "") -> str:
    """Normalise a raw label value to 'VPN', 'NONVPN', or 'UNKNOWN'."""
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        # Fall back to path hint
        h = path_hint.lower()
        if any(p in h for p in ("vpn_", "vpn/", "/vpn", "_vpn")):
            return "VPN"
        if any(p in h for p in ("nonvpn", "non_vpn", "non-vpn", "benign", "normal")):
            return "NONVPN"
        return "UNKNOWN"

    s = str(raw_value).strip().lower()

    # Numeric 1/0
    if s in VPN_INDICATORS:
        return "VPN"
    if s in NONVPN_INDICATORS:
        return "NONVPN"

    # Partial match for compound labels like 'vpn_youtube', 'nonvpn_sftp'
    if s.startswith("vpn") or "openvpn" in s or "wireguard" in s or "tor" in s:
        return "VPN"
    if s.startswith("nonvpn") or s.startswith("non_vpn") or s.startswith("non-vpn"):
        return "NONVPN"
    if s in {"benign", "normal", "regular", "non_tunnel"}:
        return "NONVPN"

    return "UNKNOWN"


def apply_column_aliases(df: pd.DataFrame, warn_prefix: str = "") -> tuple[pd.DataFrame, list[str]]:
    """For each canonical column name that is absent, try known aliases.

    Returns the (possibly enriched) DataFrame and list of alias substitutions made.
    """
    applied: list[str] = []
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical not in df.columns:
            for alias in aliases:
                if alias in df.columns and alias != canonical:
                    df[canonical] = df[alias].copy()
                    applied.append(f"{warn_prefix}: mapped '{alias}' -> '{canonical}'")
                    break
    return df, applied


def load_file(path: Path) -> pd.DataFrame | None:
    """Load a CSV or parquet file. Returns None on failure."""
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        _log(f"  SKIP {path}: could not read — {exc}", warn=True)
    return None


def has_required_cols(df: pd.DataFrame, required: list[str]) -> tuple[bool, list[str]]:
    """Return (all_present, missing_list)."""
    missing = [c for c in required if c not in df.columns]
    return len(missing) == 0, missing


# ===========================================================================
# Data loading strategy
# ===========================================================================


def _try_load_and_resolve(path: Path, required: list[str]) -> pd.DataFrame | None:
    """Load a file, apply aliases, return df only if it has all required cols."""
    df = load_file(path)
    if df is None:
        return None
    df, alias_notes = apply_column_aliases(df, warn_prefix=str(path))
    for note in alias_notes:
        _log(f"  ALIAS: {note}")
    ok, missing = has_required_cols(df, required)
    if not ok:
        return None
    return df


def build_iscx_joined_df() -> pd.DataFrame | None:
    """
    ISCX has two complementary feature files:
      - data/processed/iscx/features.parquet:
            has sz_mean_max/min, sz_std_max/min, sz_coef_variation, iat_all_*
            but MISSING sz_all_mean/std/median/p25/p75
      - artifacts/clean_pipeline_test_iscx/features.parquet:
            has mean_pkt_len (=sz_all_mean), std_pkt_len (=sz_all_std), etc.
            but only 11 801 rows (subset of iscx1)

    We join them on flow_id to construct a full-feature ISCX slice.
    """
    p1 = PROJECT_ROOT / "data" / "processed" / "iscx" / "features.parquet"
    p2 = PROJECT_ROOT / "artifacts" / "clean_pipeline_test_iscx" / "features.parquet"

    if not p1.exists():
        _log(f"  ISCX join: primary file missing: {p1}", warn=True)
        return None
    if not p2.exists():
        _log(f"  ISCX join: secondary file missing: {p2}", warn=True)
        return None

    _log(f"  ISCX: loading primary features from {p1.relative_to(PROJECT_ROOT)}")
    df1 = pd.read_parquet(p1)
    _log(f"         shape {df1.shape}")

    _log(f"  ISCX: loading secondary features from {p2.relative_to(PROJECT_ROOT)}")
    df2 = pd.read_parquet(p2)
    _log(f"         shape {df2.shape}")

    # Apply aliases on df2, so mean_pkt_len → sz_all_mean etc.
    df2, alias_notes = apply_column_aliases(df2, warn_prefix="ISCX/secondary")
    for n in alias_notes:
        _log(f"  ALIAS: {n}")

    # Columns to take from df2 (prefer those not already in df1)
    sz_all_cols_from_df2 = [
        c for c in ["sz_all_mean", "sz_all_std", "sz_all_median", "sz_all_p25", "sz_all_p75"]
        if c in df2.columns and c not in df1.columns
    ]
    extra_cols_from_df2 = sz_all_cols_from_df2

    if not extra_cols_from_df2:
        _log("  ISCX join: no extra columns from secondary — using df1 only", warn=True)
        df1["dataset"] = "iscx"
        return df1

    merge_cols = ["flow_id"] + extra_cols_from_df2
    df_merged = df1.merge(df2[merge_cols], on="flow_id", how="inner")
    _log(f"  ISCX join result: {len(df_merged)} rows (inner join on flow_id)")

    # Ensure dataset column
    df_merged["dataset"] = "iscx"

    # Apply aliases again on the merged result
    df_merged, alias_notes = apply_column_aliases(df_merged, warn_prefix="ISCX/merged")
    for n in alias_notes:
        _log(f"  ALIAS: {n}")

    return df_merged


def discover_candidate_files() -> list[Path]:
    """
    Recursively search SEARCH_DIRS for .csv and .parquet files.
    Returns sorted list of found paths (relative to project root).
    """
    found: list[Path] = []
    seen: set[Path] = set()
    for rel_dir in SEARCH_DIRS:
        d = PROJECT_ROOT / rel_dir
        if not d.exists():
            continue
        for suffix in ("*.csv", "*.parquet"):
            for p in d.rglob(suffix):
                if p not in seen:
                    found.append(p)
                    seen.add(p)
    return sorted(found)


# ===========================================================================
# Per-dataset DataFrame builders
# ===========================================================================


def build_usbvpn_df() -> pd.DataFrame | None:
    """Load USBVPN flows — has all features."""
    p = PROJECT_ROOT / "data" / "processed" / "usbvpn" / "flows.parquet"
    if not p.exists():
        _log(f"  USBVPN: file not found at {p}", warn=True)
        return None
    _log(f"  USBVPN: loading {p.relative_to(PROJECT_ROOT)}")
    df = pd.read_parquet(p)
    df, alias_notes = apply_column_aliases(df, warn_prefix="USBVPN")
    for n in alias_notes:
        _log(f"  ALIAS: {n}")
    _log(f"         shape {df.shape}")
    df["dataset"] = "usbvpn"
    return df


def build_vnat_df() -> pd.DataFrame | None:
    """Load VNAT — has only a limited set of features."""
    p = PROJECT_ROOT / "data" / "processed" / "vnat" / "features_compact_eval.parquet"
    if not p.exists():
        # Fall back to the standard features file
        p = PROJECT_ROOT / "data" / "processed" / "vnat" / "features.parquet"
    if not p.exists():
        _log("  VNAT: no feature file found", warn=True)
        return None
    _log(f"  VNAT: loading {p.relative_to(PROJECT_ROOT)}")
    df = pd.read_parquet(p)
    df, alias_notes = apply_column_aliases(df, warn_prefix="VNAT")
    for n in alias_notes:
        _log(f"  ALIAS: {n}")
    _log(f"         shape {df.shape}")
    df["dataset"] = "vnat"
    return df


# ===========================================================================
# Label normalisation for a whole DataFrame
# ===========================================================================


def add_normalised_labels(df: pd.DataFrame, dataset: str, allow_unknown: bool) -> pd.DataFrame:
    """
    Add a 'label_norm' column ('VPN' | 'NONVPN' | 'UNKNOWN') from:
      1. Existing label/class/target column (numeric 0/1 or string)
      2. flow_id / source_file for finer-grained resolution
    Drops UNKNOWN rows unless allow_unknown is True.
    """
    label_col = None
    for c in ("label", "class", "target", "category"):
        if c in df.columns:
            label_col = c
            break

    if label_col is not None:
        raw = df[label_col]
        # Numeric 1→VPN, 0→NONVPN
        if pd.api.types.is_numeric_dtype(raw):
            df["label_norm"] = raw.map({1: "VPN", 0: "NONVPN"})
            unknown_mask = df["label_norm"].isna()
        else:
            # Use path hint from flow_id if present
            fid = df.get("flow_id", pd.Series([""] * len(df), index=df.index))
            df["label_norm"] = [
                normalize_label(v, f) for v, f in zip(raw, fid)
            ]
            unknown_mask = df["label_norm"] == "UNKNOWN"
    else:
        # Fall back: derive from flow_id / source_file
        fid = df.get(
            "flow_id",
            df.get("source_file", pd.Series([""] * len(df), index=df.index)),
        )
        df["label_norm"] = [normalize_label(None, str(f)) for f in fid]
        unknown_mask = df["label_norm"] == "UNKNOWN"

    n_unknown = int(unknown_mask.sum())
    if n_unknown > 0:
        msg = (
            f"  WARNING [{dataset}]: {n_unknown} rows have UNKNOWN label. "
            f"{'Keeping (--allow-unknown)' if allow_unknown else 'Dropping them.'}"
        )
        _log(msg, warn=True)
        if not allow_unknown:
            df = df[~unknown_mask].copy()

    return df


# ===========================================================================
# Feature cleaning
# ===========================================================================


def clean_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Replace inf / -inf with NaN, then drop rows where any *required* feature is NaN.
    Also deduplicates.
    """
    df = df.copy()
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Drop rows missing required features
    required_in_df = [c for c in feature_cols if c in df.columns]
    before = len(df)
    df = df.dropna(subset=required_in_df)
    after = len(df)
    if before != after:
        _log(f"  Dropped {before - after} rows with NaN in required features.")

    # Deduplication
    dup_before = len(df)
    df = df.drop_duplicates(subset=required_in_df)
    if len(df) < dup_before:
        _log(f"  Deduplicated: removed {dup_before - len(df)} duplicate rows.")
    return df


def fill_missing_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    fill_value: float = 0.0,
) -> tuple[pd.DataFrame, list[str]]:
    """
    For feature_cols not present in df, fill with fill_value.
    Returns (df, list_of_filled_columns).
    """
    filled: list[str] = []
    for col in feature_cols:
        if col not in df.columns:
            df[col] = fill_value
            filled.append(col)
    return df, filled


# ===========================================================================
# Diversity sampling
# ===========================================================================


def sample_diverse(
    dfs_by_group: dict[tuple[str, str], pd.DataFrame],
    rows_per_group: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Sample up to rows_per_group rows from each (dataset, label) group.
    Warn if a group has fewer rows.
    Returns concatenated and shuffled DataFrame.
    """
    sampled_parts: list[pd.DataFrame] = []

    for (dataset, label), group_df in sorted(dfs_by_group.items()):
        available = len(group_df)
        target = rows_per_group
        if available == 0:
            _log(
                f"  WARN: {dataset}/{label} — 0 rows available, skipping.",
                warn=True,
            )
            continue
        if available < target:
            _log(
                f"  WARN: {dataset}/{label} — only {available} rows available "
                f"(target {target}). Using all.",
                warn=True,
            )
            sampled_parts.append(group_df)
        else:
            sampled_parts.append(
                group_df.sample(n=target, random_state=random_state, replace=False)
            )

    if not sampled_parts:
        return pd.DataFrame()

    combined = pd.concat(sampled_parts, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return combined


# ===========================================================================
# Session features from predictions (optional enrichment)
# ===========================================================================


def _load_predictions_index() -> pd.DataFrame | None:
    """
    Try to load the 3-dataset REFRESH predictions to get session-level probability
    aggregates for USBVPN / ISCX / VNAT flows.  Returns a df indexed by flow_id
    with session-aggregate columns if found, else None.
    """
    candidate = (
        PROJECT_ROOT
        / "artifacts"
        / "balanced_bagging_firewall_tuned_ensemble_3dataset_REFRESH"
        / "predictions.csv"
    )
    if not candidate.exists():
        return None
    _log(f"  Loading predictions for session enrichment: {candidate.relative_to(PROJECT_ROOT)}")
    pred = pd.read_csv(candidate)
    # Build session-level aggregates grouped by capture_id + dataset
    if "capture_id" not in pred.columns:
        return None
    agg = (
        pred.groupby(["capture_id", "dataset"])["prob"]
        .agg(
            session_mean_prob="mean",
            session_var_prob="var",
            session_fraction_high=lambda x: (x > 0.5).mean(),
        )
        .reset_index()
    )
    # top-k mean (k=3)
    def top_k_mean(s: pd.Series, k: int = 3) -> float:
        top = s.nlargest(k)
        return float(top.mean()) if len(top) > 0 else 0.0

    topk = pred.groupby(["capture_id", "dataset"])["prob"].apply(top_k_mean).reset_index()
    topk.columns = ["capture_id", "dataset", "session_top_k_mean_prob"]
    agg = agg.merge(topk, on=["capture_id", "dataset"], how="left")

    # consecutive high runs: max run of prob>0.5 within each session
    def max_consec_high(s: pd.Series) -> int:
        high = (s > 0.5).astype(int).tolist()
        max_run = cur_run = 0
        for v in high:
            cur_run = cur_run + 1 if v else 0
            max_run = max(max_run, cur_run)
        return max_run

    runs = pred.groupby(["capture_id", "dataset"])["prob"].apply(max_consec_high).reset_index()
    runs.columns = ["capture_id", "dataset", "session_consecutive_high_runs"]
    agg = agg.merge(runs, on=["capture_id", "dataset"], how="left")
    agg["session_var_prob"] = agg["session_var_prob"].fillna(0.0)

    return agg


def enrich_with_session_features(df: pd.DataFrame, session_agg: pd.DataFrame | None) -> pd.DataFrame:
    """Merge session-level probability aggregates onto per-flow df by capture_id / dataset."""
    if session_agg is None:
        return df
    if "capture_id" not in df.columns:
        return df
    merge_key = ["capture_id"]
    if "dataset" in df.columns and "dataset" in session_agg.columns:
        merge_key.append("dataset")
    df = df.merge(session_agg, on=merge_key, how="left", suffixes=("", "_sess"))
    for col in SESSION_FILL_COLS:
        dup = col + "_sess"
        if dup in df.columns:
            df[col] = df[dup].fillna(df.get(col, pd.Series(0.0, index=df.index)))
            df.drop(columns=[dup], inplace=True)
    return df


# ===========================================================================
# Notebook-consistent ROBUST9 demo CSV
# ---------------------------------------------------------------------------
# This is the canonical, notebook-aligned path for ``frontend_demo_robust9.csv``.
#
# Why this exists
# ---------------
# The legacy sampling path used downstream feature parquets that were either
# re-extracted with a different pipeline (``clean_pipeline_test_iscx``) or
# heavily subsampled (200 flows over 113 captures, sometimes 1 flow per
# capture).  p80 over a single flow collapses to that single flow's score,
# producing a CSV that PASSes even for VPN sessions and makes the model look
# broken at demo time.
#
# This new path instead joins the notebook's own ``predictions.csv`` (which
# stores ``flow_id, capture_id, dataset, split, label, prob_raw, prob_iso``
# for every flow scored by the trained ensemble) against the *original* raw
# flow data under ``data/processed/{usbvpn,iscx,vnat}/`` and emits whole
# captures only.  Selected captures are chosen by their notebook-level
# p80(prob_iso) outcome so the demo represents real TP/TN/FN behaviour.
# ===========================================================================

# Robust9 strict deployment threshold (notebook 29 / package_robust9_firewall).
ROBUST9_THRESHOLD: float = 0.8717948717948718

# Required columns for the notebook-consistent CSV (frontend contract).
NOTEBOOK_CSV_COLS: list[str] = [
    "capture_id", "session_id", "flow_id", "dataset", "label",
    "sz_all_mean", "sz_cv", "sz_all_p25", "sz_all_median", "sz_all_p75",
    "sz_mean_max", "sz_mean_min", "sz_std_max", "sz_std_min",
]


def _classify_capture_outcome(label: int, p80: float, threshold: float = ROBUST9_THRESHOLD) -> str:
    """Return one of TP / FN / TN / FP from notebook label + p80(prob_iso)."""
    blocked = p80 >= threshold
    is_vpn = int(label) == 1
    if is_vpn and blocked:
        return "TP"
    if is_vpn and not blocked:
        return "FN"
    if (not is_vpn) and blocked:
        return "FP"
    return "TN"


def _per_flow_robust9_from_raw(sizes: np.ndarray, directions: np.ndarray) -> dict[str, float]:
    """Compute the 9 robust9 features from raw packet sizes/directions.

    Mirror of ``scripts.pkt_size_mode_parity_check.compute_robust9_per_flow``.
    Used as a fallback for ISCX/VNAT where the raw ``flows.parquet`` stores
    packet ``sizes`` / ``directions`` arrays but NOT pre-computed sz_all_* /
    sz_*_max columns.  USBVPN already has these columns, so it is read
    directly from its flows.parquet.
    """
    if len(sizes) == 0:
        return {f: 0.0 for f in ROBUST9_FEATURES}
    s_all = np.asarray(np.abs(sizes), dtype=float)
    mean_all = float(s_all.mean())
    std_all = float(s_all.std(ddof=0))
    feats: dict[str, float] = {
        "sz_all_mean": mean_all,
        "sz_cv": std_all / max(mean_all, 1e-12),
        "sz_all_p25": float(np.percentile(s_all, 25)),
        "sz_all_median": float(np.percentile(s_all, 50)),
        "sz_all_p75": float(np.percentile(s_all, 75)),
    }
    d = np.asarray(directions)
    fwd = s_all[d > 0]
    bwd = s_all[d < 0]
    if len(fwd) > 0 and len(bwd) > 0:
        fwd_mean, bwd_mean = float(fwd.mean()), float(bwd.mean())
        fwd_std, bwd_std = float(fwd.std(ddof=0)), float(bwd.std(ddof=0))
    elif len(fwd) > 0:
        fwd_mean = bwd_mean = float(fwd.mean())
        fwd_std = bwd_std = float(fwd.std(ddof=0))
    elif len(bwd) > 0:
        fwd_mean = bwd_mean = float(bwd.mean())
        fwd_std = bwd_std = float(bwd.std(ddof=0))
    else:
        fwd_mean = bwd_mean = fwd_std = bwd_std = 0.0
    feats["sz_mean_max"] = max(fwd_mean, bwd_mean)
    feats["sz_mean_min"] = min(fwd_mean, bwd_mean)
    feats["sz_std_max"] = max(fwd_std, bwd_std)
    feats["sz_std_min"] = min(fwd_std, bwd_std)
    return feats


def _load_robust9_features_for_captures(
    project_root: Path,
    target_caps_by_ds: dict[str, list[str]],
) -> pd.DataFrame:
    """Load robust9 features for the supplied captures grouped by dataset.

    Returns a DataFrame with columns:
        flow_id, capture_id, dataset, label, *ROBUST9_FEATURES
    """
    out_frames: list[pd.DataFrame] = []

    # ------ USBVPN: features are pre-computed in flows.parquet --------------
    if target_caps_by_ds.get("usbvpn"):
        usb_path = project_root / "data" / "processed" / "usbvpn" / "flows.parquet"
        if usb_path.exists():
            cols_needed = [
                "flow_id", "capture_id", "label",
                "sz_all_mean", "sz_coef_variation", "sz_all_p25", "sz_all_median",
                "sz_all_p75", "sz_mean_max", "sz_mean_min", "sz_std_max", "sz_std_min",
            ]
            usb = pd.read_parquet(usb_path, columns=cols_needed)
            usb["capture_id"] = usb["capture_id"].astype(str)
            usb["flow_id"] = usb["flow_id"].astype(str)
            usb = usb[usb["capture_id"].isin(target_caps_by_ds["usbvpn"])].copy()
            usb["sz_cv"] = usb["sz_coef_variation"]
            usb["dataset"] = "usbvpn"
            usb = usb[["flow_id", "capture_id", "dataset", "label", *ROBUST9_FEATURES]]
            _log(f"  USBVPN: loaded {len(usb)} feature rows for "
                 f"{usb['capture_id'].nunique()} captures from flows.parquet")
            out_frames.append(usb)
        else:
            _log(f"  USBVPN flows.parquet not found at {usb_path}", warn=True)

    # ------ ISCX / VNAT: re-derive features from raw sizes/directions -------
    for ds in ("iscx", "vnat"):
        if not target_caps_by_ds.get(ds):
            continue
        flows_path = project_root / "data" / "processed" / ds / "flows.parquet"
        if not flows_path.exists():
            _log(f"  {ds.upper()} flows.parquet not found at {flows_path}", warn=True)
            continue
        df_raw = pd.read_parquet(
            flows_path,
            columns=["flow_id", "capture_id", "label", "sizes", "directions"],
        )
        df_raw["capture_id"] = df_raw["capture_id"].astype(str)
        df_raw["flow_id"] = df_raw["flow_id"].astype(str)
        df_raw = df_raw[df_raw["capture_id"].isin(target_caps_by_ds[ds])].copy()
        if df_raw.empty:
            _log(f"  {ds.upper()}: no rows found for requested captures", warn=True)
            continue
        feat_rows: list[dict[str, Any]] = []
        for _, r in df_raw.iterrows():
            feats = _per_flow_robust9_from_raw(
                np.asarray(r["sizes"]), np.asarray(r["directions"])
            )
            feats.update(
                {
                    "flow_id": r["flow_id"],
                    "capture_id": r["capture_id"],
                    "label": int(r["label"]),
                    "dataset": ds,
                }
            )
            feat_rows.append(feats)
        df_feats = pd.DataFrame(feat_rows)[
            ["flow_id", "capture_id", "dataset", "label", *ROBUST9_FEATURES]
        ]
        _log(f"  {ds.upper()}: derived {len(df_feats)} feature rows for "
             f"{df_feats['capture_id'].nunique()} captures from raw flows.parquet "
             f"(sizes/directions → robust9)")
        out_frames.append(df_feats)

    if not out_frames:
        return pd.DataFrame(columns=["flow_id", "capture_id", "dataset", "label", *ROBUST9_FEATURES])
    return pd.concat(out_frames, ignore_index=True)


def _select_demo_captures(
    cap_table: pd.DataFrame,
    n_tp: int = 3,
    n_tn: int = 3,
    n_fn: int = 1,
    min_flows: int = 5,
) -> dict[str, list[str]]:
    """Select demo captures from the per-capture outcome table.

    Selection policy:
      - TP VPN: highest p80(prob_iso) first (most confident blocks).  Prefer
        captures with ``n_flows >= min_flows`` so p80 is meaningful; if none
        qualify, fall back to the available pool.
      - TN NONVPN: lowest p80(prob_iso) first (cleanest passes).
      - FN VPN: a single VPN capture (label=1, p80<threshold) for honesty.
        May be empty if the notebook test split has no FN.
    """
    def _pick(df: pd.DataFrame, k: int, ascending: bool) -> list[str]:
        if df.empty or k <= 0:
            return []
        rich = df[df["n_flows"] >= min_flows]
        chosen = rich if len(rich) >= k else df
        ordered = chosen.sort_values(["p80_iso", "n_flows"], ascending=[ascending, False])
        return ordered["capture_id"].head(k).tolist()

    tp_caps = _pick(cap_table[cap_table["outcome"] == "TP"], n_tp, ascending=False)
    tn_caps = _pick(cap_table[cap_table["outcome"] == "TN"], n_tn, ascending=True)
    fn_caps = _pick(cap_table[cap_table["outcome"] == "FN"], n_fn, ascending=False)
    return {"TP": tp_caps, "TN": tn_caps, "FN": fn_caps}


def _verify_runtime_decisions_against_notebook(
    project_root: Path,
    export: pd.DataFrame,
    sel_table: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    """Re-score the freshly built CSV with the deployed robust9 ensemble.

    Reports, per selected capture, the runtime BLOCK / PASS decision
    alongside the notebook decision.  Probabilities are NOT expected to
    match exactly for ISCX/VNAT because their training-time feature
    pipeline cannot be perfectly reconstructed from the raw arrays we have
    available — the per-capture *decision* should however match for
    confidently-classified captures.  Any drift is explicitly logged so
    reviewers can see provenance honestly.
    """
    out: dict[str, Any] = {"available": False, "captures": []}
    try:
        sys.path.insert(0, str(project_root))
        from scripts.robust9_csv_inference import Robust9Inferencer  # type: ignore
    except Exception as exc:  # pragma: no cover - diagnostic best-effort
        _log(f"  Runtime verification skipped: {exc}", warn=True)
        return out

    artifact_dir = project_root / "artifacts" / "ensemble" / "diverse_bagging_robust9"
    if not artifact_dir.exists():
        _log(f"  Runtime verification skipped — artifact dir missing: {artifact_dir}", warn=True)
        return out

    inf = Robust9Inferencer(artifact_dir=artifact_dir, verbose=False)
    inf.load()

    scored = inf.predict_flows(export.copy())
    sessions = inf.decide_sessions(scored, session_col="capture_id")

    notebook_decision = dict(
        zip(
            sel_table["capture_id"].astype(str),
            np.where(sel_table["p80_iso"].astype(float) >= threshold, "BLOCK", "PASS"),
        )
    )
    notebook_outcome = dict(zip(sel_table["capture_id"].astype(str), sel_table["outcome"].astype(str)))

    _log("\n  ───── Runtime verification (rescore + p80) ───��─")
    _log(f"    {'capture_id':<40} {'expected':<8} {'runtime':<8} {'p80_runtime':<12} {'match'}")
    out["available"] = True
    n_match = 0
    p80_col_candidates = ("session_score", "p80_iso", "p80", "p80_prob_iso")
    for _, row in sessions.iterrows():
        cap = str(row.get("capture_id", row.get("session_id", "")))
        runtime_action = str(row["action"])
        runtime_p80 = float("nan")
        for c in p80_col_candidates:
            if c in row.index:
                runtime_p80 = float(row[c])
                break
        expected = notebook_decision.get(cap, "?")
        match = "OK" if runtime_action == expected else "MISMATCH"
        if runtime_action == expected:
            n_match += 1
        _log(f"    {cap:<40} {expected:<8} {runtime_action:<8} {runtime_p80:<12.4f} {match}")
        out["captures"].append(
            {
                "capture_id": cap,
                "notebook_outcome": notebook_outcome.get(cap, "?"),
                "notebook_decision": expected,
                "runtime_decision": runtime_action,
                "runtime_p80_iso": round(runtime_p80, 6),
                "decisions_match": runtime_action == expected,
            }
        )
    out["decisions_match"] = n_match
    out["decisions_total"] = int(len(sessions))
    _log(f"    decisions matching notebook: {n_match} / {len(sessions)}")
    return out


def build_robust9_csv_from_notebook(
    project_root: Path,
    output_csv: Path,
    n_tp: int = 3,
    n_tn: int = 3,
    n_fn: int = 1,
    threshold: float = ROBUST9_THRESHOLD,
    run_runtime_verification: bool = True,
) -> dict[str, Any]:
    """Build ``frontend_demo_robust9.csv`` from notebook predictions + raw flows.

    Pipeline
    --------
    1.  Load ``artifacts/ensemble/diverse_bagging_robust9/predictions.csv``
        → notebook ground truth for every scored flow.
    2.  Filter ``split == "test"``.
    3.  Per-capture aggregate ``p80(prob_iso)`` and classify TP/FN/TN/FP
        using the strict deployment threshold ``0.8717948717948718``.
    4.  Select demo captures: ``n_tp`` highest-confidence TP VPN captures,
        ``n_tn`` cleanest TN NONVPN captures, and (optionally) ``n_fn``
        FN VPN capture(s) for honesty.
    5.  Pull features for every flow in every selected capture:
          - USBVPN → from ``data/processed/usbvpn/flows.parquet``
            (sz_all_* / sz_*_max columns are already materialised there;
            ``sz_coef_variation`` is exposed as ``sz_cv``).
          - ISCX & VNAT → recompute the 9 robust9 features directly from
            ``data/processed/{iscx,vnat}/flows.parquet`` raw
            ``sizes`` / ``directions`` arrays (the only available source).
    6.  Write the CSV with the columns required by the frontend.
        ``session_id`` is set equal to ``capture_id`` for frontend
        compatibility while preserving the underlying capture identity.
    7.  Validate (no mixed-label captures/sessions, no all-zero feature
        rows) and emit a diagnostics block.
    8.  Optionally re-score with the deployed Robust9Inferencer and report
        per-capture BLOCK/PASS decisions side-by-side with notebook
        decisions so any provenance drift is visible.

    Returns
    -------
    A diagnostics dictionary suitable for inclusion in the demo summary.
    """
    pred_path = project_root / "artifacts" / "ensemble" / "diverse_bagging_robust9" / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"predictions.csv not found at {pred_path}.  "
            "Cannot build notebook-consistent robust9 demo CSV."
        )

    _log("=" * 72)
    _log("BUILD NOTEBOOK-CONSISTENT ROBUST9 DEMO CSV")
    _log("=" * 72)
    _log(f"  predictions: {pred_path.relative_to(project_root)}")
    _log(f"  output     : {output_csv.relative_to(project_root)}")
    _log(f"  threshold  : {threshold:.16f}")
    _log(f"  target TP / TN / FN captures: {n_tp} / {n_tn} / {n_fn}")

    # -- 1. Load notebook predictions ----------------------------------------
    nb = pd.read_csv(pred_path)
    required = {"flow_id", "capture_id", "split", "label", "prob_raw", "prob_iso", "dataset"}
    missing = required - set(nb.columns)
    if missing:
        raise ValueError(f"predictions.csv missing required columns: {sorted(missing)}")

    nb["dataset"] = nb["dataset"].astype(str)
    nb["capture_id"] = nb["capture_id"].astype(str)
    nb["flow_id"] = nb["flow_id"].astype(str)

    # -- 2. Filter to notebook test split ------------------------------------
    test = nb[nb["split"].astype(str) == "test"].copy()
    if test.empty:
        raise RuntimeError("predictions.csv has no rows with split=='test'")
    _log(f"\n  Notebook test split: {len(test)} flows over "
         f"{test['capture_id'].nunique()} captures across "
         f"datasets={sorted(test['dataset'].unique().tolist())}")

    # -- 3. Per-capture notebook outcomes ------------------------------------
    grp = test.groupby(["capture_id", "dataset"], as_index=False).agg(
        label=("label", "first"),
        n_flows=("flow_id", "size"),
        p80_iso=("prob_iso", lambda x: float(np.percentile(x, 80))),
        mean_iso=("prob_iso", "mean"),
    )
    grp["outcome"] = [
        _classify_capture_outcome(int(lab), float(p80), threshold)
        for lab, p80 in zip(grp["label"], grp["p80_iso"])
    ]
    out_counts = grp["outcome"].value_counts().to_dict()
    _log(f"  Per-capture notebook outcomes (test split): {out_counts}")

    # -- 4. Select demo captures ---------------------------------------------
    picks = _select_demo_captures(grp, n_tp=n_tp, n_tn=n_tn, n_fn=n_fn)
    _log(f"\n  Selected TP VPN captures   (n={len(picks['TP'])}): {picks['TP']}")
    _log(f"  Selected TN NONVPN captures (n={len(picks['TN'])}): {picks['TN']}")
    _log(f"  Selected FN VPN captures   (n={len(picks['FN'])}): {picks['FN']}")

    selected = set(picks["TP"]) | set(picks["TN"]) | set(picks["FN"])
    if not selected:
        raise RuntimeError(
            "No demo captures could be selected — the notebook test split "
            "produced 0 TP / TN / FN candidates after filtering."
        )

    sel_table = grp[grp["capture_id"].isin(selected)].copy()
    caps_by_ds: dict[str, list[str]] = {}
    for ds, sub in sel_table.groupby("dataset"):
        caps_by_ds[str(ds)] = sub["capture_id"].astype(str).tolist()
    _log(f"  Selected captures by dataset: { {k: len(v) for k, v in caps_by_ds.items()} }")

    # -- 5. Pull features for every flow in every selected capture ----------
    _log("\n  Loading feature rows for selected captures …")
    feat = _load_robust9_features_for_captures(project_root, caps_by_ds)
    if feat.empty:
        raise RuntimeError("No feature rows could be loaded for the selected captures.")

    # Restrict to the exact (flow_id, capture_id) pairs the notebook scored
    test_keys = test[["flow_id", "capture_id"]].drop_duplicates()
    before = len(feat)
    feat = feat.merge(test_keys, on=["flow_id", "capture_id"], how="inner")
    _log(f"  Intersected with notebook test flow_ids: {before} -> {len(feat)} rows")

    # Drop any duplicate flow rows (USBVPN parquet contains duplicate flow_ids
    # for windowed flows; keep the first occurrence to preserve cardinality).
    pre_dedup = len(feat)
    feat = feat.drop_duplicates(subset=["flow_id", "capture_id"], keep="first")
    if len(feat) != pre_dedup:
        _log(f"  Dropped {pre_dedup - len(feat)} duplicate flow rows after intersect")

    # -- 6. Build the export DataFrame ---------------------------------------
    feat["session_id"] = feat["capture_id"]
    feat["label"] = feat["label"].astype(int).map({1: "VPN", 0: "NONVPN"}).fillna("UNKNOWN")
    export = feat[NOTEBOOK_CSV_COLS].copy()

    # Order rows by (outcome bucket, capture_id, flow_id) for reproducibility
    cap_to_outcome = dict(zip(sel_table["capture_id"], sel_table["outcome"]))
    bucket_order = {"TP": 0, "FN": 1, "TN": 2, "FP": 3}
    export["_bucket"] = export["capture_id"].map(cap_to_outcome).map(bucket_order).fillna(99)
    export = export.sort_values(["_bucket", "capture_id", "flow_id"]).drop(columns=["_bucket"])
    export = export.reset_index(drop=True)

    # -- 7. Validate & diagnostics -------------------------------------------
    diag: dict[str, Any] = {
        "row_count": int(len(export)),
        "capture_count": int(export["capture_id"].nunique()),
        "session_count": int(export["session_id"].nunique()),
        "flow_label_counts": export["label"].value_counts().to_dict(),
        "selected_captures": {k: list(map(str, v)) for k, v in picks.items()},
        "capture_id_present": "capture_id" in export.columns,
        "notebook_outcome_breakdown_full_test": out_counts,
        "per_capture_notebook_p80": {},
    }

    # Mixed-label capture / session detection
    mixed_caps = (
        export.groupby("capture_id")["label"].nunique().pipe(lambda s: s[s > 1]).index.tolist()
    )
    mixed_sessions = (
        export.groupby("session_id")["label"].nunique().pipe(lambda s: s[s > 1]).index.tolist()
    )
    diag["mixed_label_capture_count"] = len(mixed_caps)
    diag["mixed_label_session_count"] = len(mixed_sessions)
    if mixed_caps:
        _log(f"  WARNING: mixed-label captures detected: {mixed_caps}", warn=True)
    if mixed_sessions:
        _log(f"  WARNING: mixed-label sessions detected: {mixed_sessions}", warn=True)

    # All-zero feature rows
    is_all_zero = (export[ROBUST9_FEATURES].abs().sum(axis=1) == 0.0)
    diag["all_zero_feature_row_count"] = int(is_all_zero.sum())

    # Flows per capture
    flows_per_cap = export.groupby("capture_id").size()
    diag["flows_per_capture"] = {
        "min": int(flows_per_cap.min()),
        "median": int(flows_per_cap.median()),
        "max": int(flows_per_cap.max()),
    }

    # Capture-level label counts
    cap_label_counts = (
        export.drop_duplicates("capture_id")["label"].value_counts().to_dict()
    )
    diag["capture_label_counts"] = cap_label_counts

    # Per-capture notebook stats for the chosen demo set
    for _, sub in sel_table.iterrows():
        diag["per_capture_notebook_p80"][str(sub["capture_id"])] = {
            "dataset": str(sub["dataset"]),
            "label": int(sub["label"]),
            "n_flows_notebook": int(sub["n_flows"]),
            "n_flows_csv": int((export["capture_id"] == sub["capture_id"]).sum()),
            "p80_iso_notebook": round(float(sub["p80_iso"]), 6),
            "mean_iso_notebook": round(float(sub["mean_iso"]), 6),
            "outcome_notebook": str(sub["outcome"]),
            "decision_notebook": "BLOCK" if float(sub["p80_iso"]) >= threshold else "PASS",
        }

    # -- 8. Write CSV --------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(output_csv, index=False)
    _log(f"\n  Written notebook-consistent CSV: {output_csv}  ({len(export)} rows)")

    # -- 9. Diagnostics block ------------------------------------------------
    _log("\n  ===== Diagnostics =====")
    _log(f"    row count                : {diag['row_count']}")
    _log(f"    capture count            : {diag['capture_count']}")
    _log(f"    session count            : {diag['session_count']}")
    _log(f"    flow label counts        : {diag['flow_label_counts']}")
    _log(f"    capture label counts     : {cap_label_counts}")
    _log(f"    capture_id present       : {diag['capture_id_present']}")
    _log(f"    mixed-label captures     : {diag['mixed_label_capture_count']}")
    _log(f"    mixed-label sessions     : {diag['mixed_label_session_count']}")
    _log(f"    all-zero feature rows    : {diag['all_zero_feature_row_count']}")
    _log(f"    flows per capture        : {diag['flows_per_capture']}")
    _log(f"    selected TP captures     : {diag['selected_captures']['TP']}")
    _log(f"    selected TN captures     : {diag['selected_captures']['TN']}")
    _log(f"    selected FN captures     : {diag['selected_captures']['FN']}")

    # -- 10. Runtime verification (optional, recommended) -------------------
    if run_runtime_verification:
        diag["runtime_verification"] = _verify_runtime_decisions_against_notebook(
            project_root=project_root,
            export=export,
            sel_table=sel_table,
            threshold=threshold,
        )

    return diag


# ===========================================================================
# Optional model scoring
# ===========================================================================


def score_with_models(
    df_robust9: pd.DataFrame,
    project_root: Path,
) -> None:
    """
    Score the robust9 CSV using the Robust9Inferencer (correct inference path).

    Uses scripts/robust9_csv_inference.py which:
      - Loads models directly from artifacts/ensemble/diverse_bagging_robust9/
      - Passes features as named DataFrame (NOT numpy) to prevent column mis-ordering
      - Validates feature schema against feature_order.json
      - Applies isotonic calibration correctly
      - Computes p80(prob_iso) per session and compares to threshold 0.8718

    NOTE: The old approach (rglob for model_cat_bag0.pkl) was wrong because it would
    find OTHER model files first and score xwith wrong model/feature order.
    """
    try:
        sys.path.insert(0, str(project_root))
        from scripts.robust9_csv_inference import Robust9Inferencer, STRICT_THRESHOLD, EXPECTED_FEATURES  # type: ignore

        artifact_dir = project_root / "artifacts" / "ensemble" / "diverse_bagging_robust9"
        if not artifact_dir.exists():
            _log(f"  --score-with-models: artifact dir not found: {artifact_dir}", warn=True)
            return

        _log(f"  Loading robust9 inferencer from: {artifact_dir.relative_to(project_root)}")
        inferencer = Robust9Inferencer(artifact_dir=artifact_dir, verbose=False)
        inferencer.load()

        # Check which robust9 features are present
        missing_feats = [f for f in EXPECTED_FEATURES if f not in df_robust9.columns]
        if missing_feats:
            _log(f"  --score-with-models: missing features: {missing_feats}. Skipping.", warn=True)
            return

        # Score flows
        df_scored = inferencer.predict_flows(df_robust9.copy())

        # Session decisions
        session_col = "session_id" if "session_id" in df_scored.columns else "flow_id"
        if session_col in df_scored.columns:
            sessions = inferencer.decide_sessions(df_scored, session_col=session_col)
            _log(f"\n  Session decisions ({len(sessions)} total):")
            n_block = (sessions["action"] == "BLOCK").sum()
            n_pass = (sessions["action"] == "PASS").sum()
            _log(f"    BLOCK: {n_block}  PASS: {n_pass}  threshold={STRICT_THRESHOLD:.4f}")

        # Flow-level statistics by dataset/label
        stats = (
            df_scored.groupby(["dataset", "label"])[["prob_iso", "prob_raw"]]
            .agg(["mean", "min", "max"])
            .round(4)
        )
        _log("\n  Score statistics by dataset/label:")
        _log(stats.to_string())

    except Exception as exc:
        _log(f"  --score-with-models: scoring failed — {exc}", warn=True)
        import traceback as _tb
        _log(_tb.format_exc(), warn=True)


# ===========================================================================
# Smoke tests
# ===========================================================================


def smoke_test(
    out_robust9: Path,
    out_multimodel: Path,
) -> None:
    """
    Read both output CSVs and verify correctness.
    """
    _log("\n" + "=" * 60)
    _log("SMOKE TESTS")
    _log("=" * 60)
    passed = 0
    failed = 0

    def _check(condition: bool, msg_ok: str, msg_fail: str) -> None:
        nonlocal passed, failed
        if condition:
            _log(f"  PASS: {msg_ok}")
            passed += 1
        else:
            _log(f"  FAIL: {msg_fail}", warn=True)
            failed += 1

    # --- Robust9 ---
    _log(f"\nChecking {out_robust9.name} …")
    try:
        r9 = pd.read_csv(out_robust9)
        _check(len(r9) > 0, f"Non-empty ({len(r9)} rows)", "File is empty")
        for col in ROBUST9_FEATURES:
            _check(
                col in r9.columns,
                f"Column '{col}' present",
                f"Column '{col}' MISSING",
            )
        feat_cols_present = [c for c in ROBUST9_FEATURES if c in r9.columns]
        has_nan = r9[feat_cols_present].isin([np.inf, -np.inf]).any().any() or r9[feat_cols_present].isna().any().any()
        _check(not has_nan, "No NaN/Inf in feature columns", "NaN or Inf found in feature columns")
        labels_ok = set(r9["label"].unique()).issubset({"VPN", "NONVPN"})
        _check(labels_ok, f"All labels VPN/NONVPN ({set(r9['label'].unique())})", f"Unexpected labels: {set(r9['label'].unique())}")
        datasets_in = set(r9["dataset"].unique()) if "dataset" in r9.columns else set()
        _check(len(datasets_in) > 0, f"Datasets present: {datasets_in}", "No 'dataset' column")
    except Exception as exc:
        _log(f"  ERROR reading {out_robust9}: {exc}", warn=True)
        failed += 1

    # --- Multimodel ---
    _log(f"\nChecking {out_multimodel.name} …")
    try:
        mm = pd.read_csv(out_multimodel)
        _check(len(mm) > 0, f"Non-empty ({len(mm)} rows)", "File is empty")
        for col in MULTIMODEL_FEATURES:
            _check(
                col in mm.columns,
                f"Column '{col}' present",
                f"Column '{col}' MISSING",
            )
        feat_cols_present = [c for c in MULTIMODEL_FEATURES if c in mm.columns]
        has_nan = mm[feat_cols_present].isin([np.inf, -np.inf]).any().any() or mm[feat_cols_present].isna().any().any()
        _check(not has_nan, "No NaN/Inf in feature columns", "NaN or Inf found in feature columns")
        labels_ok = set(mm["label"].unique()).issubset({"VPN", "NONVPN"})
        _check(labels_ok, f"All labels VPN/NONVPN ({set(mm['label'].unique())})", f"Unexpected labels: {set(mm['label'].unique())}")
        datasets_in = set(mm["dataset"].unique()) if "dataset" in mm.columns else set()
        _check(len(datasets_in) > 0, f"Datasets present: {datasets_in}", "No 'dataset' column")
    except Exception as exc:
        _log(f"  ERROR reading {out_multimodel}: {exc}", warn=True)
        failed += 1

    _log(f"\nSmoke tests: {passed} passed, {failed} failed.")


# ===========================================================================
# Report generation
# ===========================================================================


def write_report(
    output_dir: Path,
    files_scanned: list[Path],
    files_used: list[Path],
    files_skipped: list[tuple[Path, str]],
    rows_before: int,
    rows_after_clean: int,
    output_row_counts: dict[str, int],
    distribution: dict[str, Any],
    missing_filled: dict[str, list[str]],
    final_cols: dict[str, list[str]],
    out_paths: dict[str, Path],
) -> None:
    _log("\n" + "=" * 60)
    _log("FINAL REPORT")
    _log("=" * 60)
    _log(f"Files scanned : {len(files_scanned)}")
    _log(f"Files used    : {len(files_used)}")
    _log(f"Files skipped : {len(files_skipped)}")
    for p, reason in files_skipped:
        _log(f"  SKIP {p}: {reason}")
    _log(f"Rows before cleaning  : {rows_before}")
    _log(f"Rows after cleaning   : {rows_after_clean}")
    for name, n in output_row_counts.items():
        _log(f"Output rows [{name}]  : {n}")
    _log("\nClass distribution by dataset:")
    for k, v in distribution.items():
        _log(f"  {k}: {v}")
    _log("\nMissing columns filled with 0.0 (multimodel):")
    for ds, cols in missing_filled.items():
        if cols:
            _log(f"  {ds}: {cols}")
    _log("\nFinal column lists:")
    for name, cols in final_cols.items():
        _log(f"  [{name}]: {cols}")
    _log("\nCreated files:")
    for name, p in out_paths.items():
        _log(f"  {name}: {p}")

    report_path = output_dir / "frontend_demo_report.txt"
    report_path.write_text("\n".join(_report_lines), encoding="utf-8")
    _log(f"\nReport written to: {report_path}")


def write_summary_json(
    output_dir: Path,
    out_robust9: Path,
    out_multimodel: Path,
    distribution: dict[str, Any],
    output_row_counts: dict[str, int],
) -> None:
    summary = {
        "robust9_csv": str(out_robust9),
        "multimodel_csv": str(out_multimodel),
        "rows": output_row_counts,
        "distribution": distribution,
        "warnings": _warnings,
    }
    summary_path = output_dir / "frontend_demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _log(f"Summary JSON written to: {summary_path}")


# ===========================================================================
# Main pipeline
# ===========================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frontend demo CSVs from processed feature files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rows-per-group",
        type=int,
        default=50,
        help="Target rows per (dataset, label) group in each output CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports/frontend_demo",
        help="Output directory for demo CSVs (relative to project root or absolute).",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        default=False,
        help="Keep rows with UNKNOWN label (default: drop them).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling and shuffling.",
    )
    parser.add_argument(
        "--score-with-models",
        action="store_true",
        default=False,
        help="If set, attempt to score robust9 CSV with saved model artifacts.",
    )
    parser.add_argument(
        "--robust9-source",
        choices=["notebook", "legacy"],
        default="notebook",
        help=(
            "How to build frontend_demo_robust9.csv:  "
            "'notebook' (default, recommended) = join notebook predictions.csv "
            "with raw flows.parquet, whole-capture selection from the test "
            "split.  'legacy' = old (dataset, label) sampling path used "
            "before the demo-CSV fix."
        ),
    )
    parser.add_argument(
        "--robust9-tp", type=int, default=3,
        help="Number of TP VPN captures to include in notebook-consistent CSV.",
    )
    parser.add_argument(
        "--robust9-tn", type=int, default=3,
        help="Number of TN NONVPN captures to include in notebook-consistent CSV.",
    )
    parser.add_argument(
        "--robust9-fn", type=int, default=1,
        help="Number of FN VPN captures to include (set 0 to skip).",
    )
    parser.add_argument(
        "--robust9-skip-runtime-verify",
        action="store_true",
        default=False,
        help="Skip the runtime re-scoring verification step (faster).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Resolve output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _log("=" * 60)
    _log("BUILD FRONTEND DEMO CSVs")
    _log("=" * 60)
    _log(f"Project root     : {PROJECT_ROOT}")
    _log(f"Output dir       : {output_dir}")
    _log(f"Rows per group   : {args.rows_per_group}")
    _log(f"Random state     : {args.random_state}")
    _log(f"Allow unknown    : {args.allow_unknown}")
    _log(f"Robust9 source   : {args.robust9_source}")
    _log("")

    out_robust9 = output_dir / "frontend_demo_robust9.csv"
    out_multimodel = output_dir / "frontend_demo_multimodel.csv"

    # -----------------------------------------------------------------------
    # ROBUST9: notebook-consistent path (default)
    # -----------------------------------------------------------------------
    robust9_diag: dict[str, Any] = {}
    if args.robust9_source == "notebook":
        try:
            robust9_diag = build_robust9_csv_from_notebook(
                project_root=PROJECT_ROOT,
                output_csv=out_robust9,
                n_tp=args.robust9_tp,
                n_tn=args.robust9_tn,
                n_fn=args.robust9_fn,
                threshold=ROBUST9_THRESHOLD,
                run_runtime_verification=not args.robust9_skip_runtime_verify,
            )
            # Persist a sidecar JSON with full diagnostics
            diag_path = output_dir / "frontend_demo_robust9_diagnostics.json"
            diag_path.write_text(
                json.dumps(robust9_diag, indent=2, default=str), encoding="utf-8"
            )
            _log(f"\n  Robust9 diagnostics JSON: {diag_path}")
        except Exception as exc:
            _log(
                f"FATAL: notebook-consistent robust9 builder failed: {exc}",
                warn=True,
            )
            _log(traceback.format_exc(), warn=True)
            _log("Falling back to legacy sampling path for robust9.", warn=True)
            args.robust9_source = "legacy"

    # -----------------------------------------------------------------------
    # A. Data discovery
    # -----------------------------------------------------------------------
    _log("--- A. Data Discovery ---")
    files_scanned = discover_candidate_files()
    _log(f"Found {len(files_scanned)} candidate files across search paths.")

    files_used: list[Path] = []
    files_skipped: list[tuple[Path, str]] = []

    # -----------------------------------------------------------------------
    # B. Load known-good datasets
    # -----------------------------------------------------------------------
    _log("\n--- B. Loading Datasets ---")

    dataset_dfs: dict[str, pd.DataFrame] = {}

    # -- USBVPN --
    _log("\n[USBVPN]")
    usbvpn_df = build_usbvpn_df()
    if usbvpn_df is not None:
        usbvpn_df = add_normalised_labels(usbvpn_df, "usbvpn", args.allow_unknown)
        dataset_dfs["usbvpn"] = usbvpn_df
        files_used.append(PROJECT_ROOT / "data" / "processed" / "usbvpn" / "flows.parquet")
    else:
        _log("  USBVPN data not available.", warn=True)

    # -- ISCX (joined) --
    _log("\n[ISCX]")
    iscx_df = build_iscx_joined_df()
    if iscx_df is not None:
        iscx_df = add_normalised_labels(iscx_df, "iscx", args.allow_unknown)
        dataset_dfs["iscx"] = iscx_df
        files_used.append(PROJECT_ROOT / "data" / "processed" / "iscx" / "features.parquet")
        files_used.append(PROJECT_ROOT / "artifacts" / "clean_pipeline_test_iscx" / "features.parquet")
    else:
        _log("  ISCX data not available.", warn=True)

    # -- VNAT --
    _log("\n[VNAT]")
    vnat_df = build_vnat_df()
    if vnat_df is not None:
        vnat_df = add_normalised_labels(vnat_df, "vnat", args.allow_unknown)
        dataset_dfs["vnat"] = vnat_df
        files_used.append(PROJECT_ROOT / "data" / "processed" / "vnat" / "features_compact_eval.parquet")
    else:
        _log("  VNAT data not available.", warn=True)

    # Mark files that were scanned but not used
    for f in files_scanned:
        if f not in files_used:
            files_skipped.append((f, "not a primary feature source or superseded"))

    if not dataset_dfs:
        _log("FATAL: No data loaded. Exiting.", warn=True)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # C. Optional: enrich with session-level features from predictions
    # -----------------------------------------------------------------------
    _log("\n--- C. Session Feature Enrichment ---")
    session_agg = _load_predictions_index()
    if session_agg is None:
        _log("  Session enrichment not available — session_* cols will be 0.0", warn=True)

    for ds_name, df in dataset_dfs.items():
        dataset_dfs[ds_name] = enrich_with_session_features(df, session_agg)

    # -----------------------------------------------------------------------
    # D. Build multimodel DataFrames per dataset
    # -----------------------------------------------------------------------
    _log("\n--- D. Building Multimodel DataFrames ---")

    rows_before = sum(len(df) for df in dataset_dfs.values())
    _log(f"Total rows loaded: {rows_before}")

    missing_filled_report: dict[str, list[str]] = {}
    multimodel_groups: dict[tuple[str, str], pd.DataFrame] = {}
    robust9_groups: dict[tuple[str, str], pd.DataFrame] = {}

    for ds_name, df in dataset_dfs.items():
        _log(f"\n  [{ds_name}] shape={df.shape}")

        # Apply aliases
        df, alias_notes = apply_column_aliases(df, warn_prefix=ds_name)
        for n in alias_notes:
            _log(f"  ALIAS: {n}")

        # --- MULTIMODEL preparation ---
        df_mm = df.copy()
        df_mm, filled_cols = fill_missing_features(df_mm, MULTIMODEL_FEATURES, fill_value=0.0)
        missing_filled_report[ds_name] = filled_cols
        if filled_cols:
            _log(
                f"  [{ds_name}] Multimodel: filled {len(filled_cols)} missing cols with 0.0: {filled_cols}",
                warn=True,
            )

        df_mm = clean_features(df_mm, MULTIMODEL_FEATURES)

        # Ensure passthrough columns
        if "session_id" not in df_mm.columns:
            df_mm["session_id"] = df_mm.get(
                "capture_id", pd.Series("", index=df_mm.index)
            )
        if "flow_id" not in df_mm.columns:
            df_mm["flow_id"] = df_mm.index.astype(str)
        df_mm["label"] = df_mm["label_norm"]

        # Group by label
        for label in ("VPN", "NONVPN"):
            grp = df_mm[df_mm["label_norm"] == label].copy()
            if len(grp) > 0:
                multimodel_groups[(ds_name, label)] = grp
                _log(f"  [{ds_name}/{label}] multimodel rows: {len(grp)}")
            else:
                _log(f"  [{ds_name}/{label}] multimodel rows: 0", warn=True)

        # --- ROBUST9 preparation ---
        # Check if all robust9 features are available after alias mapping
        ok_r9, missing_r9 = has_required_cols(df, ROBUST9_FEATURES)
        if not ok_r9:
            _log(
                f"  [{ds_name}] Cannot provide robust9 rows — missing: {missing_r9}",
                warn=True,
            )
            continue

        df_r9 = df.copy()
        df_r9 = clean_features(df_r9, ROBUST9_FEATURES)
        if "session_id" not in df_r9.columns:
            df_r9["session_id"] = df_r9.get(
                "capture_id", pd.Series("", index=df_r9.index)
            )
        if "flow_id" not in df_r9.columns:
            df_r9["flow_id"] = df_r9.index.astype(str)
        df_r9["label"] = df_r9["label_norm"]

        for label in ("VPN", "NONVPN"):
            grp = df_r9[df_r9["label_norm"] == label].copy()
            if len(grp) > 0:
                robust9_groups[(ds_name, label)] = grp
                _log(f"  [{ds_name}/{label}] robust9 rows: {len(grp)}")
            else:
                _log(f"  [{ds_name}/{label}] robust9 rows: 0", warn=True)

    rows_after_clean = (
        sum(len(df) for df in multimodel_groups.values())
    )

    # -----------------------------------------------------------------------
    # E. Diversity sampling
    # -----------------------------------------------------------------------
    _log("\n--- E. Diversity Sampling ---")

    _log(f"\nSampling robust9 CSV (target {args.rows_per_group} per group):")
    df_robust9_out = sample_diverse(robust9_groups, args.rows_per_group, args.random_state)

    _log(f"\nSampling multimodel CSV (target {args.rows_per_group} per group):")
    df_multimodel_out = sample_diverse(multimodel_groups, args.rows_per_group, args.random_state)

    # -----------------------------------------------------------------------
    # F & G. Finalise column order and export
    # -----------------------------------------------------------------------
    _log("\n--- F/G. Exporting CSVs ---")

    def _finalise(df: pd.DataFrame, output_cols: list[str]) -> pd.DataFrame:
        """Select & order output columns, ensuring all exist."""
        for col in output_cols:
            if col not in df.columns:
                df[col] = 0.0 if col not in PASSTHROUGH_COLS else ""
        return df[output_cols].copy()

    df_r9_final = _finalise(df_robust9_out, ROBUST9_OUTPUT_COLS)
    df_mm_final = _finalise(df_multimodel_out, MULTIMODEL_OUTPUT_COLS)

    # Robust9 CSV: only emit from this legacy path if the notebook-consistent
    # path was not used.  The notebook-consistent CSV is preferred and has
    # already been written above when args.robust9_source == "notebook".
    if args.robust9_source == "legacy":
        df_r9_final.to_csv(out_robust9, index=False)
        _log(f"  Written (LEGACY): {out_robust9}  ({len(df_r9_final)} rows)")
    else:
        _log(
            f"  Skipping legacy robust9 write — notebook-consistent CSV "
            f"already at {out_robust9}"
        )

    df_mm_final.to_csv(out_multimodel, index=False)
    _log(f"  Written: {out_multimodel}  ({len(df_mm_final)} rows)")

    # -----------------------------------------------------------------------
    # Compute distribution summary
    # -----------------------------------------------------------------------
    distribution: dict[str, Any] = {}
    # Read back the actual written robust9 CSV so the summary reflects whatever
    # source path produced it (notebook-consistent OR legacy).
    try:
        df_r9_on_disk = pd.read_csv(out_robust9)
    except Exception:
        df_r9_on_disk = df_r9_final
    for df_out, name in [(df_r9_on_disk, "robust9"), (df_mm_final, "multimodel")]:
        if "dataset" in df_out.columns and "label" in df_out.columns:
            dist = (
                df_out.groupby(["dataset", "label"])
                .size()
                .reset_index(name="count")
                .to_dict(orient="records")
            )
            distribution[name] = dist
    robust9_row_count = len(df_r9_on_disk)

    # -----------------------------------------------------------------------
    # H. Report
    # -----------------------------------------------------------------------
    write_report(
        output_dir=output_dir,
        files_scanned=files_scanned,
        files_used=files_used,
        files_skipped=files_skipped,
        rows_before=rows_before,
        rows_after_clean=rows_after_clean,
        output_row_counts={
            "robust9": robust9_row_count,
            "multimodel": len(df_mm_final),
        },
        distribution=distribution,
        missing_filled=missing_filled_report,
        final_cols={
            "robust9": list(df_r9_on_disk.columns),
            "multimodel": list(df_mm_final.columns),
        },
        out_paths={"robust9_csv": out_robust9, "multimodel_csv": out_multimodel},
    )
    write_summary_json(
        output_dir=output_dir,
        out_robust9=out_robust9,
        out_multimodel=out_multimodel,
        distribution=distribution,
        output_row_counts={
            "robust9": robust9_row_count,
            "multimodel": len(df_mm_final),
        },
    )

    # -----------------------------------------------------------------------
    # I. Smoke tests
    # -----------------------------------------------------------------------
    smoke_test(out_robust9, out_multimodel)

    # -----------------------------------------------------------------------
    # J. Optional model scoring
    # -----------------------------------------------------------------------
    if args.score_with_models:
        _log("\n--- J. Scoring with Models ---")
        score_with_models(df_r9_final, PROJECT_ROOT)

    _log("\nDone.")


if __name__ == "__main__":
    main()

