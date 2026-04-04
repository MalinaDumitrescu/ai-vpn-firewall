# src/clean_pipeline/feature_extractor.py
"""
Feature extraction for the CLEAN pipeline.

Computes every feature in the feature_families registry from raw
packet-level arrays (timestamps, sizes, directions).

Key invariant: every formula here operates identically on data from
VNAT, ISCX, and USBVPN -- the unified schema guarantees same semantics.

MEMORY-SAFE: Provides both batch (DataFrame) and single-flow extraction
so the pipeline can process large datasets in streaming fashion.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.clean_pipeline.feature_families import get_family


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------

_EPS = 1e-9


def _safe_stats(arr: np.ndarray) -> Dict[str, float]:
    """Compute standard summary stats safely (handles empty arrays)."""
    if arr.size == 0:
        return {
            "count": 0.0,
            "sum": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }
    return {
        "count": float(arr.size),
        "sum": float(arr.sum()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def _iat(timestamps: np.ndarray) -> np.ndarray:
    """Inter-arrival times from sorted timestamps."""
    if timestamps.size <= 1:
        return np.array([], dtype=np.float64)
    ts_sorted = np.sort(timestamps.astype(np.float64))
    d = np.diff(ts_sorted)
    return np.maximum(d, 0.0)  # clamp negatives from clock drift


# ------------------------------------------------------
# Single-flow feature computation
# ------------------------------------------------------

def extract_flow_features(
    timestamps: np.ndarray,
    sizes: np.ndarray,
    directions: np.ndarray,
    *,
    max_packets: int = 300,
) -> Dict[str, float]:
    """
    Compute ALL registered features for a single flow.

    Parameters
    ----------
    timestamps : array of float, epoch seconds
    sizes : array of int, absolute byte counts
    directions : array of int (0 or 1)
    max_packets : int
        Truncate flow to first N packets (window size).

    Returns
    -------
    dict[feature_name  float]
    """
    # Truncate to window
    n = min(len(timestamps), len(sizes), len(directions), max_packets)
    ts = np.asarray(timestamps[:n], dtype=np.float64)
    sz = np.abs(np.asarray(sizes[:n], dtype=np.float64))
    dr = np.asarray(directions[:n], dtype=np.int32)

    # Sort by timestamp for IAT correctness
    order = np.argsort(ts)
    ts = ts[order]
    sz = sz[order]
    dr = dr[order]

    # Split by direction
    fwd_mask = dr == 1
    bwd_mask = dr == 0
    fwd_sz = sz[fwd_mask]
    bwd_sz = sz[bwd_mask]

    # IAT
    iat_all = _iat(ts)
    iat_stats = _safe_stats(iat_all)
    sz_stats = _safe_stats(sz)

    # Duration
    duration = float(ts[-1] - ts[0]) if n >= 2 else 0.0

    feat: Dict[str, float] = {}

    # === Size features ===
    feat["total_packets"] = float(n)
    feat["total_bytes"] = sz_stats["sum"]
    feat["mean_pkt_len"] = sz_stats["mean"]
    feat["std_pkt_len"] = sz_stats["std"]
    feat["median_pkt_len"] = sz_stats["median"]
    feat["p25_pkt_len"] = sz_stats["p25"]
    feat["p75_pkt_len"] = sz_stats["p75"]
    feat["max_pkt_len"] = sz_stats["max"]
    feat["min_pkt_len"] = sz_stats["min"]

    # === IAT features ===
    feat["iat_mean"] = iat_stats["mean"]
    feat["iat_std"] = iat_stats["std"]
    feat["iat_median"] = iat_stats["median"]
    feat["iat_p25"] = iat_stats["p25"]
    feat["iat_p75"] = iat_stats["p75"]

    # === Duration / rate features ===
    feat["flow_duration"] = duration
    feat["packet_rate"] = float(n) / max(duration, _EPS)
    feat["byte_rate"] = sz_stats["sum"] / max(duration, _EPS)

    # === Derived temporal features ===
    feat["iat_cv"] = iat_stats["std"] / max(iat_stats["mean"], _EPS)
    feat["iat_iqr"] = iat_stats["p75"] - iat_stats["p25"]

    # === Derived size features ===
    feat["pkt_len_cv"] = sz_stats["std"] / max(sz_stats["mean"], _EPS)
    feat["pkt_len_iqr"] = sz_stats["p75"] - sz_stats["p25"]

    # === Direction features (per-direction stats) ===
    fwd_count = float(fwd_sz.size)
    bwd_count = float(bwd_sz.size)
    fwd_bytes = float(fwd_sz.sum()) if fwd_sz.size > 0 else 0.0
    bwd_bytes = float(bwd_sz.sum()) if bwd_sz.size > 0 else 0.0
    fwd_mean = float(fwd_sz.mean()) if fwd_sz.size > 0 else 0.0
    bwd_mean = float(bwd_sz.mean()) if bwd_sz.size > 0 else 0.0

    # SEMANTICALLY_RISKY direction-labeled features
    feat["fwd_packets"] = fwd_count
    feat["bwd_packets"] = bwd_count
    feat["fwd_bytes"] = fwd_bytes
    feat["bwd_bytes"] = bwd_bytes
    feat["fwd_mean_pkt_len"] = fwd_mean
    feat["bwd_mean_pkt_len"] = bwd_mean
    feat["packet_ratio"] = fwd_count / max(bwd_count, 1.0)
    feat["byte_ratio"] = fwd_bytes / max(bwd_bytes, _EPS)

    # SAFE direction-INVARIANT features (use min/max so label doesn't matter)
    feat["dir_pkt_ratio_minmax"] = (
        min(fwd_count, bwd_count) / max(fwd_count, bwd_count, 1.0)
    )
    feat["dir_bytes_ratio_minmax"] = (
        min(fwd_bytes, bwd_bytes) / max(fwd_bytes, bwd_bytes, _EPS)
    )
    feat["dir_mean_pkt_max"] = max(fwd_mean, bwd_mean)
    feat["dir_mean_pkt_min"] = min(fwd_mean, bwd_mean)

    return feat


# ------------------------------------------------------
# Single-flow extraction from raw lists (no DataFrame)
# ------------------------------------------------------

def extract_single_flow(
    timestamps: Sequence,
    sizes: Sequence,
    directions: Sequence,
    *,
    family: str = "direction_invariant_augmented",
    max_packets: int = 300,
) -> Optional[Dict[str, float]]:
    """
    Extract features for ONE flow from raw Python lists.

    This is the memory-safe single-flow entry point used during
    streaming processing.  No DataFrame is created.

    Returns
    -------
    dict of feature_name  float, or None if flow is too short.
    """
    family_cols = set(get_family(family))

    n = min(len(timestamps), len(sizes), len(directions))
    if n < 3:
        return None

    feat = extract_flow_features(
        np.asarray(timestamps[:n], dtype=np.float64),
        np.asarray(sizes[:n], dtype=np.float64),
        np.asarray(directions[:n], dtype=np.int32),
        max_packets=max_packets,
    )

    # Filter to requested family only
    return {k: v for k, v in feat.items() if k in family_cols}


# ------------------------------------------------------
# Batch extraction over a unified flows DataFrame
# ------------------------------------------------------

def extract_features_batch(
    flows_df: pd.DataFrame,
    *,
    family: str = "direction_invariant_augmented",
    max_packets: int = 300,
    min_packets: int = 3,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Extract features from a unified flows DataFrame.

    Parameters
    ----------
    flows_df : DataFrame
        Must have: flow_id, capture_id, dataset, label,
                   timestamps, sizes, directions
    family : str
        Feature family to select (from feature_families.py).
    max_packets : int
        Window size (truncate per flow).
    min_packets : int
        Flows with fewer packets are dropped.
    progress : bool
        Show tqdm progress bar.

    Returns
    -------
    DataFrame with: flow_id, capture_id, dataset, label, + feature columns
    """
    family_cols = list(get_family(family))

    # Validate incoming schema
    required = {"flow_id", "capture_id", "dataset", "label",
                "timestamps", "sizes", "directions"}
    missing = required - set(flows_df.columns)
    if missing:
        raise ValueError(f"flows_df missing required columns: {missing}")

    rows: List[Dict] = []

    iterator = flows_df.itertuples(index=False)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(flows_df), desc="Extracting features")
        except ImportError:
            pass

    for row in iterator:
        ts = row.timestamps
        sz = row.sizes
        dr = row.directions

        n = min(len(ts), len(sz), len(dr))
        if n < min_packets:
            continue

        feat = extract_flow_features(
            np.asarray(ts, dtype=np.float64),
            np.asarray(sz, dtype=np.float64),
            np.asarray(dr, dtype=np.int32),
            max_packets=max_packets,
        )

        # Metadata columns
        feat["flow_id"] = row.flow_id
        feat["capture_id"] = row.capture_id
        feat["dataset"] = row.dataset
        feat["label"] = int(row.label)

        if hasattr(row, "source_file"):
            feat["source_file"] = row.source_file
        if hasattr(row, "app"):
            feat["app"] = row.app

        rows.append(feat)

    df_out = pd.DataFrame(rows)

    # Select only the requested family + metadata
    meta_cols = ["flow_id", "capture_id", "dataset", "label"]
    optional_meta = ["source_file", "app"]
    for col in optional_meta:
        if col in df_out.columns:
            meta_cols.append(col)

    # Keep only family features that were actually computed
    available_feats = [c for c in family_cols if c in df_out.columns]
    missing_feats = [c for c in family_cols if c not in df_out.columns]
    if missing_feats:
        print(f"WARNING: {len(missing_feats)} family features not computed: {missing_feats}")

    df_out = df_out[meta_cols + available_feats].copy()

    # Ensure numeric
    for c in available_feats:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce").fillna(0.0)

    # Replace inf
    df_out = df_out.replace([np.inf, -np.inf], 0.0)

    print(f"Extracted {len(df_out)} flows x {len(available_feats)} features "
          f"(family={family})")

    return df_out


if __name__ == "__main__":
    # Quick test: extract from VNAT
    from pathlib import Path
    from src.clean_pipeline.vnat_loader import load_vnat_raw

    h5 = Path("data/raw/vnat/VNAT_Dataframe_release_1.h5")
    flows = load_vnat_raw(h5, min_packets=3)
    feats = extract_features_batch(flows, family="safe_core_plus_temporal", max_packets=100)
    print(feats.head())
    print(feats.describe())

