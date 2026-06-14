"""
src/features/unified_extractor.py
==================================
Unified feature extractor for the ``unified_feature_contract_v2`` experiment.

This module defines ONE canonical set of formulas used across every dataset
(ISCX, USBVPN, VNAT) and the live PCAP extractor.

Design principles
-----------------
* Every formula is written exactly once here and imported everywhere else.
* All derived features use only normalised / bounded formulas so that the
  numeric range is the same regardless of capture conditions.
* No feature that encodes dataset identity (e.g. flow_duration, total_bytes)
  is produced by this extractor unless explicitly requested as a diagnostic.

Packet-size convention
----------------------
Sizes are expected to be **IP total-length in bytes** (integer values).
For ISCX and VNAT the raw ``sizes`` array in flows.parquet already stores
IP-total-length values. For live capture the extractor should similarly
use IP total length (``ip.len``).
USBVPN does not supply raw packet arrays; its pre-computed ``sz_all_*``
statistics are accepted as-is and assumed to follow the same convention
(IP total length). The formula report documents this caveat.

Direction convention
--------------------
``direction == 1`` → upload / client → server (forward direction)
``direction == 0`` → download / server → client (reverse direction)

IAT units
---------
Timestamps are expected in **seconds** (floating-point Unix epoch or relative
time in seconds). Inter-arrival times (IAT) are computed via ``np.diff(ts)``
and floored to *eps* to avoid log-zero issues in downstream models.

Quantile method
---------------
All percentiles use ``numpy.percentile`` with the default ``linear``
interpolation (numpy < 2.0 default). This is identical to
``np.quantile(..., method="linear")``.

Zero-division handling
----------------------
All ratio and normalised features use a small epsilon (default ``1e-6``)
added to denominators.  No feature silently returns NaN; if a sub-array is
empty the feature returns ``0.0``.

Missing-value policy
--------------------
* If the raw sizes / timestamps / directions array is empty or shorter than
  ``min_packets`` the feature row is still produced but ``q_min_packets_ok``
  is set to 0.
* USBVPN-only path: if a required base-stat column is NaN the derived
  feature is set to ``0.0``.

Flow / session grouping rules
------------------------------
* The extractor operates on a **single flow window** of up to ``N`` packets.
* Session grouping (multiple flows → one session) is NOT performed here; that
  is a higher-level responsibility.

Extractor version: ``unified_v2.0``
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

EXTRACTOR_VERSION = "unified_v2.0"
PACKET_SIZE_MODE = "ip_total_length_bytes"
DIRECTION_CONVENTION = "1=upload/client_to_server 0=download/server_to_client"
IAT_UNIT = "seconds"
QUANTILE_METHOD = "linear_interpolation_numpy_percentile"
EPS = 1e-6
MISSING_VALUE_POLICY = "zero_fill_on_empty_or_nan"
MAX_WINDOW = 100          # N: maximum packets used per flow
MIN_PACKETS = 3           # flows with fewer packets are flagged (q_min_packets_ok=0)

# ---------------------------------------------------------------------------
# All unified feature names (canonical order)
# ---------------------------------------------------------------------------

UNIFIED_FEATURES: List[str] = [
    # Size — absolute statistics
    "sz_all_mean",
    "sz_all_std",
    "sz_all_p25",
    "sz_all_median",
    "sz_all_p75",
    # Size — per-direction cross features
    "sz_mean_max",
    "sz_mean_min",
    "sz_std_max",
    "sz_std_min",
    # Size — derived ratio/shape
    "sz_cv",            # coefficient of variation  (std / mean)
    "sz_iqr",           # IQR  (p75 - p25)
    "sz_qratio",        # quartile ratio  (p75 / p25)
    "sz_median_to_mean",# median / mean
    "sz_coef_variation",# alias: same as sz_cv, kept for legacy compatibility
    "sz_p25_median_ratio",
    "sz_p75_median_ratio",
    "sz_iqr_norm_median",
    # Timing — absolute statistics
    "iat_all_mean",
    "iat_all_std",
    "iat_all_p25",
    "iat_all_median",
    "iat_all_p75",
    # Timing — per-direction cross features
    "iat_mean_max",
    "iat_mean_min",
    "iat_std_max",
    "iat_std_min",
    # Timing — derived ratio/shape
    "iat_cv",           # coefficient of variation  (std / mean)
    "iat_iqr",          # IQR  (p75 - p25)
    "iat_median",       # alias for iat_all_median
    "iat_p25",          # alias for iat_all_p25
    "iat_p75",          # alias for iat_all_p75
    # Directional
    "direction_balance_bytes",
    "direction_balance_packets",
    "dispersion_symmetry",
]

# Features safe to use as model inputs (excludes construction-leak features)
CONSTRUCTION_LEAK_FEATURES: List[str] = [
    "flow_duration",
    "total_packets",
    "total_bytes",
    "packet_rate",
    "byte_rate",
]

# ---------------------------------------------------------------------------
# Feature families (defined here; used by the build script)
# ---------------------------------------------------------------------------

FAMILY_UNIFIED_FULL: List[str] = [f for f in UNIFIED_FEATURES
                                   if f != "sz_coef_variation"]  # deduplicated (sz_cv is canonical)

FAMILY_UNIFIED_SIZE_SHAPE: List[str] = [
    "sz_all_mean", "sz_all_std", "sz_all_p25", "sz_all_median", "sz_all_p75",
    "sz_mean_max", "sz_mean_min", "sz_std_max", "sz_std_min",
    "sz_cv", "sz_iqr", "sz_qratio", "sz_median_to_mean",
    "sz_p25_median_ratio", "sz_p75_median_ratio", "sz_iqr_norm_median",
]

FAMILY_UNIFIED_TIMING_SHAPE: List[str] = [
    "iat_all_mean", "iat_all_std", "iat_all_p25", "iat_all_median", "iat_all_p75",
    "iat_mean_max", "iat_mean_min", "iat_std_max", "iat_std_min",
    "iat_cv", "iat_iqr", "iat_median", "iat_p25", "iat_p75",
]

FAMILY_UNIFIED_DIRECTIONAL_SHAPE: List[str] = [
    "direction_balance_bytes",
    "direction_balance_packets",
    "dispersion_symmetry",
]

FAMILY_UNIFIED_DIRECTIONLESS: List[str] = [
    f for f in FAMILY_UNIFIED_FULL
    if f not in FAMILY_UNIFIED_DIRECTIONAL_SHAPE
]

FAMILY_UNIFIED_RELATIVE_SHAPE_V2: List[str] = [
    # Pure normalised/ratio features — robust to absolute scale changes
    "sz_cv", "sz_iqr", "sz_qratio", "sz_median_to_mean",
    "sz_p25_median_ratio", "sz_p75_median_ratio", "sz_iqr_norm_median",
    "iat_cv", "iat_iqr",
    "direction_balance_bytes", "direction_balance_packets",
    "dispersion_symmetry",
]

FAMILY_UNIFIED_SAFE_HYBRID_CANDIDATE_POOL: List[str] = [
    # Candidate pool for later anti-fingerprint feature selection.
    # Includes both absolute and relative features, but excludes
    # features that were confirmed to have different formulas in legacy datasets.
    "sz_all_mean", "sz_all_p25", "sz_all_median", "sz_all_p75",
    "sz_mean_max", "sz_mean_min",
    "sz_cv", "sz_iqr", "sz_qratio", "sz_median_to_mean",
    "sz_p25_median_ratio", "sz_p75_median_ratio", "sz_iqr_norm_median",
    "iat_all_mean", "iat_all_p25", "iat_all_median", "iat_all_p75",
    "iat_mean_max", "iat_mean_min",
    "iat_cv", "iat_iqr",
    "direction_balance_bytes", "direction_balance_packets",
    "dispersion_symmetry",
]

ALL_FAMILIES: Dict[str, List[str]] = {
    "unified_full": FAMILY_UNIFIED_FULL,
    "unified_size_shape": FAMILY_UNIFIED_SIZE_SHAPE,
    "unified_timing_shape": FAMILY_UNIFIED_TIMING_SHAPE,
    "unified_directional_shape": FAMILY_UNIFIED_DIRECTIONAL_SHAPE,
    "unified_directionless": FAMILY_UNIFIED_DIRECTIONLESS,
    "unified_relative_shape_v2": FAMILY_UNIFIED_RELATIVE_SHAPE_V2,
    "unified_safe_hybrid_candidate_pool": FAMILY_UNIFIED_SAFE_HYBRID_CANDIDATE_POOL,
}

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _safe_stats(x: np.ndarray) -> Dict[str, float]:
    """
    Compute descriptive statistics for a 1-D array.
    Returns zeros for empty arrays.
    """
    if x.size == 0:
        return {
            "count": 0.0, "sum": 0.0, "mean": 0.0, "std": 0.0,
            "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0,
        }
    return {
        "count": float(x.size),
        "sum": float(x.sum()),
        "mean": float(x.mean()),
        "std": float(np.std(x, ddof=0)),    # population std (ddof=0)
        "min": float(x.min()),
        "p25": float(np.percentile(x, 25)), # linear interpolation
        "median": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "max": float(x.max()),
    }


def _iat(ts: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Compute inter-arrival times from a sorted timestamp array (seconds).
    IATs are floored to eps to prevent zero/negative values.
    """
    if ts.size <= 1:
        return np.asarray([], dtype=float)
    d = np.diff(ts.astype(float))
    return np.maximum(d, eps)


def _split_by_direction(
    values: np.ndarray,
    dirs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split values into upload (dirs==1) and download (dirs==0) arrays."""
    up = values[dirs == 1]
    dn = values[dirs == 0]
    return up, dn


# ---------------------------------------------------------------------------
# Core unified formula set
# ---------------------------------------------------------------------------

def compute_direction_balance_bytes(bytes_up: float, bytes_down: float,
                                    eps: float = EPS) -> float:
    """
    Normalised directional byte balance.
    Range: [-1, 1]   0 = perfectly symmetric   1 = all upload   -1 = all download
    Formula:  (bytes_up - bytes_down) / (bytes_up + bytes_down + eps)
    """
    return float((bytes_up - bytes_down) / (bytes_up + bytes_down + eps))


def compute_direction_balance_packets(pkts_up: float, pkts_down: float,
                                      eps: float = EPS) -> float:
    """
    Normalised directional packet balance.
    Range: [-1, 1]
    Formula:  (pkts_up - pkts_down) / (pkts_up + pkts_down + eps)
    """
    return float((pkts_up - pkts_down) / (pkts_up + pkts_down + eps))


def compute_dispersion_symmetry(p25: float, median: float, p75: float,
                                eps: float = EPS) -> float:
    """
    Quantile-skewness measure for the all-packets size distribution.
    Measures whether the distribution is symmetric around its median.
    Range: [-1, 1]   0 = symmetric   >0 = right-skewed   <0 = left-skewed
    Formula:  clip( (p75 + p25 - 2*median) / (|p75 - p25| + eps), -1, 1 )
    """
    num = p75 + p25 - 2.0 * median
    den = abs(p75 - p25)
    return float(np.clip(num / (den + eps), -1.0, 1.0))


def compute_sz_cv(std: float, mean: float, eps: float = EPS) -> float:
    """
    Coefficient of variation for packet sizes.
    Formula: std / (mean + eps)
    """
    return float(std / (mean + eps))


def compute_sz_iqr(p25: float, p75: float) -> float:
    """IQR of all-packet sizes.  Formula: p75 - p25"""
    return float(p75 - p25)


def compute_sz_qratio(p25: float, p75: float, eps: float = EPS) -> float:
    """
    Quartile ratio of all-packet sizes.
    Formula: p75 / (p25 + eps)
    """
    return float(p75 / (p25 + eps))


def compute_sz_median_to_mean(median: float, mean: float, eps: float = EPS) -> float:
    """
    Median-to-mean ratio.  Values near 1 indicate symmetry.
    Formula: median / (mean + eps)
    """
    return float(median / (mean + eps))


def compute_iat_cv(std: float, mean: float, eps: float = EPS) -> float:
    """Coefficient of variation for IAT.  Formula: std / (mean + eps)"""
    return float(std / (mean + eps))


def compute_iat_iqr(p25: float, p75: float) -> float:
    """IQR of all-packet IAT.  Formula: p75 - p25"""
    return float(p75 - p25)


# ---------------------------------------------------------------------------
# Single-flow extraction from raw arrays
# ---------------------------------------------------------------------------

def extract_unified_features_from_arrays(
    timestamps: np.ndarray,
    sizes: np.ndarray,
    directions: np.ndarray,
    eps: float = EPS,
    max_n: int = MAX_WINDOW,
    min_packets: int = MIN_PACKETS,
) -> Dict[str, float]:
    """
    Extract all unified features from raw per-packet arrays for a single flow.

    Parameters
    ----------
    timestamps  : 1-D float array, sorted ascending, in seconds
    sizes       : 1-D int/float array of IP-total-length bytes per packet
    directions  : 1-D int array (1=upload, 0=download)
    eps         : zero-division guard (default 1e-6)
    max_n       : maximum packets to use (window truncation)
    min_packets : minimum for quality flag

    Returns
    -------
    dict mapping feature_name -> float
    """
    n = min(len(timestamps), len(sizes), len(directions), max_n)
    ts = np.asarray(timestamps[:n], dtype=float)
    sz = np.asarray(sizes[:n], dtype=float)
    dr = np.asarray(directions[:n], dtype=int)

    # Per-direction splits
    sz_up, sz_dn = _split_by_direction(sz, dr)
    ts_up = ts[dr == 1]
    ts_dn = ts[dr == 0]

    # Stats
    st_all = _safe_stats(sz)
    st_up  = _safe_stats(sz_up)
    st_dn  = _safe_stats(sz_dn)

    iat_all = _iat(ts, eps)
    iat_up  = _iat(ts_up, eps)
    iat_dn  = _iat(ts_dn, eps)

    st_iat_all = _safe_stats(iat_all)
    st_iat_up  = _safe_stats(iat_up)
    st_iat_dn  = _safe_stats(iat_dn)

    # Direction totals
    bytes_up   = float(sz_up.sum()) if sz_up.size > 0 else 0.0
    bytes_dn   = float(sz_dn.sum()) if sz_dn.size > 0 else 0.0
    pkts_up    = float(sz_up.size)
    pkts_dn    = float(sz_dn.size)

    feat: Dict[str, float] = {}

    # --- Size absolute ---
    feat["sz_all_mean"]   = st_all["mean"]
    feat["sz_all_std"]    = st_all["std"]
    feat["sz_all_p25"]    = st_all["p25"]
    feat["sz_all_median"] = st_all["median"]
    feat["sz_all_p75"]    = st_all["p75"]

    # --- Size per-direction cross ---
    feat["sz_mean_max"] = max(st_up["mean"], st_dn["mean"])
    feat["sz_mean_min"] = min(st_up["mean"], st_dn["mean"])
    feat["sz_std_max"]  = max(st_up["std"],  st_dn["std"])
    feat["sz_std_min"]  = min(st_up["std"],  st_dn["std"])

    # --- Size derived ---
    feat["sz_cv"]              = compute_sz_cv(st_all["std"], st_all["mean"], eps)
    feat["sz_coef_variation"]  = feat["sz_cv"]   # legacy alias
    feat["sz_iqr"]             = compute_sz_iqr(st_all["p25"], st_all["p75"])
    feat["sz_qratio"]          = compute_sz_qratio(st_all["p25"], st_all["p75"], eps)
    feat["sz_median_to_mean"]  = compute_sz_median_to_mean(st_all["median"], st_all["mean"], eps)
    feat["sz_p25_median_ratio"]= (st_all["p25"] / st_all["median"]
                                  if st_all["median"] > 0 else 0.0)
    feat["sz_p75_median_ratio"]= (st_all["p75"] / st_all["median"]
                                  if st_all["median"] > 0 else 0.0)
    _iqr_val = feat["sz_iqr"]
    feat["sz_iqr_norm_median"] = float(_iqr_val / (st_all["median"] + eps))

    # --- Timing absolute ---
    feat["iat_all_mean"]   = st_iat_all["mean"]
    feat["iat_all_std"]    = st_iat_all["std"]
    feat["iat_all_p25"]    = st_iat_all["p25"]
    feat["iat_all_median"] = st_iat_all["median"]
    feat["iat_all_p75"]    = st_iat_all["p75"]

    # --- Timing per-direction cross ---
    feat["iat_mean_max"] = max(st_iat_up["mean"], st_iat_dn["mean"])
    feat["iat_mean_min"] = min(st_iat_up["mean"], st_iat_dn["mean"])
    feat["iat_std_max"]  = max(st_iat_up["std"],  st_iat_dn["std"])
    feat["iat_std_min"]  = min(st_iat_up["std"],  st_iat_dn["std"])

    # --- Timing derived ---
    feat["iat_cv"]    = compute_iat_cv(st_iat_all["std"], st_iat_all["mean"], eps)
    feat["iat_iqr"]   = compute_iat_iqr(st_iat_all["p25"], st_iat_all["p75"])
    feat["iat_median"]= st_iat_all["median"]
    feat["iat_p25"]   = st_iat_all["p25"]
    feat["iat_p75"]   = st_iat_all["p75"]

    # --- Directional (unified formulas) ---
    feat["direction_balance_bytes"]   = compute_direction_balance_bytes(bytes_up, bytes_dn, eps)
    feat["direction_balance_packets"] = compute_direction_balance_packets(pkts_up, pkts_dn, eps)
    feat["dispersion_symmetry"]       = compute_dispersion_symmetry(
        st_all["p25"], st_all["median"], st_all["p75"], eps)

    # --- Quality flag ---
    feat["q_packet_count"]   = float(n)
    feat["q_min_packets_ok"] = float(n >= min_packets)

    return feat


# ---------------------------------------------------------------------------
# Batch extraction from flows DataFrame (ISCX / VNAT path)
# ---------------------------------------------------------------------------

def extract_unified_features_from_flows(
    flows_df: pd.DataFrame,
    eps: float = EPS,
    max_n: int = MAX_WINDOW,
    min_packets: int = MIN_PACKETS,
) -> pd.DataFrame:
    """
    Batch extract unified features from a flows DataFrame that contains
    per-packet array columns: ``timestamps``, ``sizes``, ``directions``.

    Input columns required: ``flow_id``, ``capture_id``, ``label``,
    ``timestamps``, ``sizes``, ``directions``.

    Returns a DataFrame with ``flow_id``, ``capture_id``, ``label``,
    all unified features, and quality flags.
    """
    rows = []
    for r in flows_df.itertuples(index=False):
        feat = extract_unified_features_from_arrays(
            timestamps=np.asarray(r.timestamps, dtype=float),
            sizes=np.asarray(r.sizes, dtype=float),
            directions=np.asarray(r.directions, dtype=int),
            eps=eps,
            max_n=max_n,
            min_packets=min_packets,
        )
        feat["flow_id"]    = str(r.flow_id)
        feat["capture_id"] = str(r.capture_id)
        feat["label"]      = int(r.label)
        rows.append(feat)

    out = pd.DataFrame(rows)

    # Ensure numeric types for all feature columns
    id_cols = {"flow_id", "capture_id"}
    for c in out.columns:
        if c not in id_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out


# ---------------------------------------------------------------------------
# Batch extraction from precomputed stats DataFrame (USBVPN path)
# ---------------------------------------------------------------------------

def recompute_unified_features_from_stats(
    stats_df: pd.DataFrame,
    eps: float = EPS,
) -> pd.DataFrame:
    """
    Recompute unified derived features from pre-computed base statistics.
    Used for USBVPN which lacks raw packet arrays.

    Expected input columns:
        sz_all_mean, sz_all_std, sz_all_p25, sz_all_median, sz_all_p75
        sz_mean_max, sz_mean_min, sz_std_max, sz_std_min
        iat_all_mean, iat_all_std, iat_all_p25, iat_all_median, iat_all_p75
        iat_mean_max, iat_mean_min, iat_std_max, iat_std_min
        bytes_up, bytes_down, packets_up, packets_down

    Directional and derived features are RECOMPUTED using unified formulas.
    The base absolute statistics are passed through unchanged.

    Returns a DataFrame with all UNIFIED_FEATURES columns plus
    ``flow_id``, ``capture_id``, ``label``, ``q_packet_count``,
    ``q_min_packets_ok``.
    """
    df = stats_df.copy()

    def _col(name: str, default: float = 0.0) -> pd.Series:
        return df[name].fillna(default).astype(float) if name in df.columns else pd.Series(
            default, index=df.index, dtype=float)

    # --- Pass through absolute stats ---
    for col in ["sz_all_mean", "sz_all_std", "sz_all_p25", "sz_all_median", "sz_all_p75",
                "sz_mean_max", "sz_mean_min", "sz_std_max", "sz_std_min",
                "iat_all_mean", "iat_all_std", "iat_all_p25", "iat_all_median", "iat_all_p75",
                "iat_mean_max", "iat_mean_min", "iat_std_max", "iat_std_min"]:
        df[col] = _col(col)

    # --- Recompute directional features using unified formulas ---
    bu  = _col("bytes_up")
    bd  = _col("bytes_down")
    pu  = _col("packets_up")
    pd_ = _col("packets_down")

    df["direction_balance_bytes"] = (bu - bd) / (bu + bd + eps)
    df["direction_balance_packets"] = (pu - pd_) / (pu + pd_ + eps)

    # --- Recompute dispersion_symmetry using unified formula ---
    p25    = _col("sz_all_p25")
    median = _col("sz_all_median")
    p75    = _col("sz_all_p75")
    num = p75 + p25 - 2.0 * median
    den = (p75 - p25).abs()
    df["dispersion_symmetry"] = np.clip(num / (den + eps), -1.0, 1.0)

    # --- Recompute derived size features ---
    sz_mean = _col("sz_all_mean")
    sz_std  = _col("sz_all_std")
    df["sz_cv"]               = sz_std / (sz_mean + eps)
    df["sz_coef_variation"]   = df["sz_cv"]
    df["sz_iqr"]              = p75 - p25
    df["sz_qratio"]           = p75 / (p25 + eps)
    df["sz_median_to_mean"]   = median / (sz_mean + eps)
    df["sz_p25_median_ratio"] = p25 / (median + eps)
    df["sz_p75_median_ratio"] = p75 / (median + eps)
    df["sz_iqr_norm_median"]  = df["sz_iqr"] / (median + eps)

    # --- Recompute derived timing features ---
    iat_mean = _col("iat_all_mean")
    iat_std  = _col("iat_all_std")
    iat_p25  = _col("iat_all_p25")
    iat_p75  = _col("iat_all_p75")
    df["iat_cv"]    = iat_std / (iat_mean + eps)
    df["iat_iqr"]   = iat_p75 - iat_p25
    df["iat_median"]= _col("iat_all_median")
    df["iat_p25"]   = iat_p25
    df["iat_p75"]   = iat_p75

    # --- Quality flags ---
    if "q_packet_count" not in df.columns and "tot_pkt" in df.columns:
        df["q_packet_count"] = _col("tot_pkt")
    elif "q_packet_count" not in df.columns:
        df["q_packet_count"] = 0.0
    if "q_min_packets_ok" not in df.columns:
        df["q_min_packets_ok"] = (df["q_packet_count"] >= MIN_PACKETS).astype(float)

    return df


# ---------------------------------------------------------------------------
# Feature contract dict (used to create feature_contract.json)
# ---------------------------------------------------------------------------

def get_feature_contract() -> dict:
    """Return the full feature contract as a serialisable dictionary."""
    return {
        "extractor_version": EXTRACTOR_VERSION,
        "created": "2026-05-30",
        "packet_size_mode": PACKET_SIZE_MODE,
        "direction_convention": DIRECTION_CONVENTION,
        "iat_unit": IAT_UNIT,
        "quantile_method": QUANTILE_METHOD,
        "eps": EPS,
        "max_window_packets": MAX_WINDOW,
        "min_packets": MIN_PACKETS,
        "missing_value_policy": MISSING_VALUE_POLICY,
        "ddof_for_std": 0,
        "all_unified_features": UNIFIED_FEATURES,
        "feature_families": {k: v for k, v in ALL_FAMILIES.items()},
        "construction_leak_features_excluded": CONSTRUCTION_LEAK_FEATURES,
        "forbidden_model_input_columns": [
            "dataset", "capture_id", "source_capture_id", "flow_id",
            "source_file", "split", "label", "app", "connection_str",
            "capture_name", "row_id", "flow_key", "file_names",
            "q_packet_count", "q_min_packets_ok",
        ],
        "runtime_compatibility": {
            "iscx": {
                "extraction_path": "raw_arrays",
                "raw_cols_needed": ["timestamps", "sizes", "directions"],
                "all_features_computable": True,
                "notes": "Full re-extraction from raw per-packet arrays.",
            },
            "vnat": {
                "extraction_path": "raw_arrays",
                "raw_cols_needed": ["timestamps", "sizes", "directions"],
                "all_features_computable": True,
                "notes": "Full re-extraction from raw per-packet arrays.",
            },
            "usbvpn": {
                "extraction_path": "precomputed_stats",
                "required_stats": [
                    "sz_all_mean", "sz_all_std", "sz_all_p25", "sz_all_median", "sz_all_p75",
                    "sz_mean_max", "sz_mean_min", "sz_std_max", "sz_std_min",
                    "iat_all_mean", "iat_all_std", "iat_all_p25", "iat_all_median", "iat_all_p75",
                    "iat_mean_max", "iat_mean_min", "iat_std_max", "iat_std_min",
                    "bytes_up", "bytes_down", "packets_up", "packets_down",
                ],
                "all_features_computable": True,
                "notes": (
                    "Raw packet arrays not available. Absolute base-stats are passed through "
                    "from the pre-processed parquet. Directional and derived features are "
                    "RECOMPUTED using unified formulas. Packet-size convention is assumed "
                    "to be IP total length, consistent with ISCX/VNAT, but cannot be "
                    "independently verified."
                ),
            },
            "live_pcap": {
                "extraction_path": "raw_arrays",
                "raw_cols_needed": ["timestamps", "sizes", "directions"],
                "all_features_computable": True,
                "notes": (
                    "Live extractor must use IP total length (ip.len) for sizes. "
                    "Timestamps must be in seconds. "
                    "Direction must be 1=client-to-server, 0=server-to-client."
                ),
            },
        },
    }

