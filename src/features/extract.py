from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import hashlib
import json

import numpy as np
import pandas as pd
import yaml

from src.features.histograms import HistSpec, fixed_hist
from src.features.window_quality import WindowQuality, quality_features



@dataclass(frozen=True)
class FeatureConfig:
    N: int
    min_packets: int
    eps: float

    size_bins: List[float]
    size_max: float

    iat_bins: List[float]
    iat_max: float


def load_feature_config(features_yaml: Path) -> FeatureConfig:
    cfg = yaml.safe_load(features_yaml.read_text(encoding="utf-8")) or {}

    w = cfg.get("window") or {}
    N = int(w.get("N", 100))
    min_packets = int(w.get("min_packets", 10))
    eps = float(w.get("eps", 1e-6))

    h = cfg.get("histograms") or {}
    size = h.get("size") or {}
    iat = h.get("iat") or {}

    size_bins = list(map(float, size.get("bins", [])))
    iat_bins = list(map(float, iat.get("bins", [])))

    if len(size_bins) < 2:
        raise ValueError("features.yaml: histograms.size.bins must have at least 2 edges")
    if len(iat_bins) < 2:
        raise ValueError("features.yaml: histograms.iat.bins must have at least 2 edges")

    size_max = float(size.get("max_size", 2000))
    iat_max = float(iat.get("max_iat", 2.0))

    return FeatureConfig(
        N=N,
        min_packets=min_packets,
        eps=eps,
        size_bins=size_bins,
        size_max=size_max,
        iat_bins=iat_bins,
        iat_max=iat_max,
    )

def feature_config_hash_text(features_yaml_path: str | Path) -> str:
    p = Path(features_yaml_path)
    return hashlib.sha256(p.read_bytes()).hexdigest()

# -----------------------------
# Core feature helpers
# -----------------------------

def _split_by_direction(sizes: np.ndarray, dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # directions are 0/1 in your VNAT preprocessing
    up = sizes[dirs == 1]
    down = sizes[dirs == 0]
    return up, down


def _safe_stats(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
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
        "count": float(x.size),
        "sum": float(x.sum()),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "min": float(x.min()),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "max": float(x.max()),
    }


def _iat(ts: np.ndarray, eps: float) -> np.ndarray:
    if ts.size <= 1:
        return np.asarray([], dtype=float)
    d = np.diff(ts.astype(float))
    # Just in case any tiny negatives ever slip through again
    d = np.maximum(d, eps)
    return d


def _burstiness(iats: np.ndarray) -> float:
    # Simple, explainable: coefficient of variation (std/mean)
    if iats.size == 0:
        return 0.0
    m = float(iats.mean())
    if m <= 0:
        return 0.0
    return float(iats.std(ddof=0) / m)


# -----------------------------
# Public extraction
# -----------------------------

def extract_features_from_flows(
    flows: pd.DataFrame,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    """
    Input `flows` must contain at least:
      - flow_id, capture_id, label
      - timestamps, sizes, directions
      - packet_count, window_complete, min_packets_ok

    Output contains:
      - ids (flow_id, capture_id)
      - label
      - numeric features
    """

    required = {
        "flow_id",
        "capture_id",
        "label",
        "timestamps",
        "sizes",
        "directions",
        "packet_count",
        "window_complete",
        "min_packets_ok",
    }
    missing = required - set(flows.columns)
    if missing:
        raise ValueError(f"Missing required columns in flows dataframe: {sorted(missing)}")

    size_spec = HistSpec(bins=cfg.size_bins, max_value=cfg.size_max, normalize=True)
    iat_spec = HistSpec(bins=cfg.iat_bins, max_value=cfg.iat_max, normalize=True)

    rows: List[Dict[str, float | int | str]] = []

    for r in flows.itertuples(index=False):
        ts = np.asarray(r.timestamps, dtype=float)
        sz = np.asarray(r.sizes, dtype=float)
        dr = np.asarray(r.directions, dtype=int)

        # Defensive: enforce same length
        n = min(ts.size, sz.size, dr.size, cfg.N)
        ts, sz, dr = ts[:n], sz[:n], dr[:n]

        # Duration + rates
        duration = float(ts[-1] - ts[0]) if ts.size >= 2 else 0.0
        duration = max(duration, cfg.eps)

        up_sz, down_sz = _split_by_direction(sz, dr)
        up_bytes = float(up_sz.sum())
        down_bytes = float(down_sz.sum())
        total_bytes = float(sz.sum())

        pkt_count = int(n)
        up_pkts = int(up_sz.size)
        down_pkts = int(down_sz.size)

        # IATs
        iat_all = _iat(ts, cfg.eps)
        # Directional IATs: compute diffs only within that direction stream
        iat_up = _iat(ts[dr == 1], cfg.eps)
        iat_down = _iat(ts[dr == 0], cfg.eps)

        # Stats
        st_sz_all = _safe_stats(sz)
        st_sz_up = _safe_stats(up_sz)
        st_sz_down = _safe_stats(down_sz)

        st_iat_all = _safe_stats(iat_all)
        st_iat_up = _safe_stats(iat_up)
        st_iat_down = _safe_stats(iat_down)

        # Ratios (avoid div by 0)
        eps = cfg.eps
        
        # REMOVED: f_pkt_imbalance and f_byte_imbalance as requested
        # REMOVED: f_iat_burstiness as requested

        # Histograms
        h_size_all = fixed_hist(sz, size_spec)
        h_iat_all = fixed_hist(iat_all, iat_spec)

        window_complete = bool(n >= cfg.N)
        min_packets_ok = bool(n >= cfg.min_packets)

        # Quality features (from your precomputed columns)
        q = WindowQuality(
            packet_count=pkt_count,
            window_complete=window_complete,
            min_packets_ok=min_packets_ok,
        )

        feat: Dict[str, float | int | str] = {
            "flow_id": str(r.flow_id),
            "capture_id": str(r.capture_id),
            "label": int(r.label),
            
            # REMOVED: Imbalance features
            # "f_pkt_imbalance": float(pkt_imbalance),
            # "f_byte_imbalance": float(byte_imbalance),
            # "f_iat_burstiness": _burstiness(iat_all),
        }

        # Flatten stats with prefixes
        # REMOVED: 'count', 'sum', 'min', 'max' to avoid overfitting to specific capture artifacts
        for k, v in st_sz_all.items():
            if k in ("count", "sum", "min", "max"): continue
            feat[f"sz_all_{k}"] = v

        # Coefficient of Variation: unitless burstiness metric
        sz_cv = st_sz_all["std"] / st_sz_all["mean"] if st_sz_all["mean"] > 0 else 0.0
        feat["sz_cv"] = float(sz_cv)
        
        # CHANGED: Direction-invariant stats
        # Instead of up/down, we aggregate them into "primary" (larger volume) and "secondary" (smaller volume)
        # Or just use "all" stats + imbalance ratios.
        # For simplicity and robustness, we will DROP separate up/down stats and rely on "all" + imbalance.
        # This is the strongest way to force direction invariance.
        
        # If we really want distributional info per direction without directionality, we can sort them.
        # e.g. "larger_mean", "smaller_mean".
        # Let's try adding "max_mean" and "min_mean" for size/iat if they differ significantly.
        
        feat["sz_mean_max"] = max(st_sz_up["mean"], st_sz_down["mean"])
        feat["sz_mean_min"] = min(st_sz_up["mean"], st_sz_down["mean"])
        feat["sz_std_max"] = max(st_sz_up["std"], st_sz_down["std"]) # Not necessarily from same direction as mean_max
        feat["sz_std_min"] = min(st_sz_up["std"], st_sz_down["std"])

        for k, v in st_iat_all.items():
            if k in ("count", "sum", "min", "max"): continue
            feat[f"iat_all_{k}"] = v
            
        feat["iat_mean_max"] = max(st_iat_up["mean"], st_iat_down["mean"])
        feat["iat_mean_min"] = min(st_iat_up["mean"], st_iat_down["mean"])
        feat["iat_std_max"] = max(st_iat_up["std"], st_iat_down["std"])
        feat["iat_std_min"] = min(st_iat_up["std"], st_iat_down["std"])

        # Hist features with stable naming
        for i, v in enumerate(h_size_all):
            feat[f"h_size_all_{i:02d}"] = float(v)
        for i, v in enumerate(h_iat_all):
            feat[f"h_iat_all_{i:02d}"] = float(v)

        # Quality features
        feat.update(quality_features(q))

        rows.append(feat)

    out = pd.DataFrame(rows)

    # Make sure every feature column is numeric except ids
    id_cols = {"flow_id", "capture_id"}
    for c in out.columns:
        if c in id_cols:
            continue
        out[c] = pd.to_numeric(out[c], errors="raise")

    return out
