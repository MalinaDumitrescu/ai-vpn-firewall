from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import hashlib
import json

import numpy as np
import pandas as pd
import yaml

from src.features.window_quality import WindowQuality, quality_features


@dataclass(frozen=True)
class FeatureConfig:
    N: int
    min_packets: int
    eps: float


def load_feature_config(features_yaml: Path) -> FeatureConfig:
    cfg = yaml.safe_load(features_yaml.read_text(encoding="utf-8")) or {}

    w = cfg.get("window") or {}
    N = int(w.get("N", 100))
    min_packets = int(w.get("min_packets", 10))
    eps = float(w.get("eps", 1e-6))

    return FeatureConfig(
        N=N,
        min_packets=min_packets,
        eps=eps,
    )


def feature_config_hash_text(features_yaml_path: str | Path) -> str:
    p = Path(features_yaml_path)
    return hashlib.sha256(p.read_bytes()).hexdigest()


# -----------------------------
# Core feature helpers
# -----------------------------

def _split_by_direction(sizes: np.ndarray, dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    d = np.maximum(d, eps)
    return d


# -----------------------------
# Public extraction
# -----------------------------

def extract_features_from_flows(
        flows: pd.DataFrame,
        cfg: FeatureConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []

    for r in flows.itertuples(index=False):
        ts = np.asarray(r.timestamps, dtype=float)
        sz = np.asarray(r.sizes, dtype=float)
        dr = np.asarray(r.directions, dtype=int)

        n = min(ts.size, sz.size, dr.size, cfg.N)
        ts, sz, dr = ts[:n], sz[:n], dr[:n]

        # Basic flow stats
        up_sz, down_sz = _split_by_direction(sz, dr)
        iat_all = _iat(ts, cfg.eps)
        iat_up = _iat(ts[dr == 1], cfg.eps)
        iat_down = _iat(ts[dr == 0], cfg.eps)

        st_sz_all = _safe_stats(sz)
        st_sz_up = _safe_stats(up_sz)
        st_sz_down = _safe_stats(down_sz)

        st_iat_all = _safe_stats(iat_all)
        st_iat_up = _safe_stats(iat_up)
        st_iat_down = _safe_stats(iat_down)

        # 1. INITIALIZE the dictionary
        feat: Dict[str, float | int | str] = {
            "flow_id": str(r.flow_id),
            "capture_id": str(r.capture_id),
            "label": int(r.label),
        }

        # 2. Add sz_cv (Unitless robustness metric)
        feat["sz_coef_variation"] = float(st_sz_all["std"] / st_sz_all["mean"]) if st_sz_all["mean"] > 0 else 0.0

        # 3. Add Size ratios (replacing raw stats with ratios)
        feat["sz_p25_median_ratio"] = st_sz_all["p25"] / st_sz_all["median"] if st_sz_all["median"] > 0 else 0.0
        feat["sz_p75_median_ratio"] = st_sz_all["p75"] / st_sz_all["median"] if st_sz_all["median"] > 0 else 0.0
        feat["sz_max_min_ratio"] = st_sz_all["max"] / st_sz_all["min"] if st_sz_all["min"] > 0 else 0.0

        # 4. Add Direction-Invariant Size Metrics
        feat["sz_mean_max"] = max(st_sz_up["mean"], st_sz_down["mean"])
        feat["sz_mean_min"] = min(st_sz_up["mean"], st_sz_down["mean"])
        feat["sz_std_max"] = max(st_sz_up["std"], st_sz_down["std"])
        feat["sz_std_min"] = min(st_sz_up["std"], st_sz_down["std"])

        # 5. Add Timing stats
        for k in ("mean", "std", "p25", "median", "p75"):
            feat[f"iat_all_{k}"] = st_iat_all[k]

        feat["iat_mean_max"] = max(st_iat_up["mean"], st_iat_down["mean"])
        feat["iat_mean_min"] = min(st_iat_up["mean"], st_iat_down["mean"])
        feat["iat_std_max"] = max(st_iat_up["std"], st_iat_down["std"])
        feat["iat_std_min"] = min(st_iat_up["std"], st_iat_down["std"])

        # 6. Window Quality
        q = WindowQuality(
            packet_count=int(n),
            min_packets_ok=bool(n >= cfg.min_packets),
        )
        feat.update(quality_features(q))

        rows.append(feat)

    out = pd.DataFrame(rows)

    # Compute session-level invariance features
    session_features = []
    for capture_id, group in out.groupby('capture_id'):
        # Use sz_coef_variation as the "probability" proxy for session features
        probs = group['sz_coef_variation'].values
        
        if len(probs) == 0:
            continue
            
        mean_prob = float(np.mean(probs))
        var_prob = float(np.var(probs))
        
        top_k = min(3, len(probs))
        top_k_mean = float(np.mean(np.sort(probs)[-top_k:]))
        
        threshold = float(np.median(probs))
        fraction = float(np.mean(probs > threshold))
        
        # Consecutive high-prob runs (runs of 2+ consecutive > threshold)
        runs = 0
        current_run = 0
        for p in probs:
            if p > threshold:
                current_run += 1
            else:
                if current_run >= 2:
                    runs += 1
                current_run = 0
        if current_run >= 2:
            runs += 1
        
        session_features.append({
            'capture_id': capture_id,
            'session_mean_prob': mean_prob,
            'session_var_prob': var_prob,
            'session_top_k_mean_prob': top_k_mean,
            'session_consecutive_high_runs': runs,
            'session_fraction_high': fraction
        })
    
    session_df = pd.DataFrame(session_features)
    out = out.merge(session_df, on='capture_id', how='left')

    # Final cleanup
    id_cols = {"flow_id", "capture_id"}
    for c in out.columns:
        if c not in id_cols:
            out[c] = pd.to_numeric(out[c], errors="raise")

    return out
