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
        up_pkt_ratio = up_pkts / max(pkt_count, 1)
        up_byte_ratio = up_bytes / max(total_bytes, eps)

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

            "f_duration_s": duration,
            "f_total_pkts": float(pkt_count),
            "f_up_pkts": float(up_pkts),
            "f_down_pkts": float(down_pkts),
            "f_total_bytes": total_bytes,
            "f_up_bytes": up_bytes,
            "f_down_bytes": down_bytes,

            "f_pkts_per_s": float(pkt_count) / duration,
            "f_bytes_per_s": total_bytes / duration,

            "f_up_pkt_ratio": float(up_pkt_ratio),
            "f_up_byte_ratio": float(up_byte_ratio),

            "f_iat_burstiness": _burstiness(iat_all),
        }

        # Flatten stats with prefixes
        for k, v in st_sz_all.items():
            feat[f"sz_all_{k}"] = v
        for k, v in st_sz_up.items():
            feat[f"sz_up_{k}"] = v
        for k, v in st_sz_down.items():
            feat[f"sz_down_{k}"] = v

        for k, v in st_iat_all.items():
            feat[f"iat_all_{k}"] = v
        for k, v in st_iat_up.items():
            feat[f"iat_up_{k}"] = v
        for k, v in st_iat_down.items():
            feat[f"iat_down_{k}"] = v

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
