from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import hashlib

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

        # 4. Add Direction-Invariant Size Metrics (legacy — not used by FeaturePipeline)
        feat["sz_mean_max"] = max(st_sz_up["mean"], st_sz_down["mean"])
        feat["sz_mean_min"] = min(st_sz_up["mean"], st_sz_down["mean"])
        feat["sz_std_max"] = max(st_sz_up["std"], st_sz_down["std"])
        feat["sz_std_min"] = min(st_sz_up["std"], st_sz_down["std"])

        # 5a. Remaining COMPACT_FEATURES (required by FeaturePipeline)
        _eps = cfg.eps
        _iqr = st_sz_all["p75"] - st_sz_all["p25"]
        feat["sz_iqr_norm_median"] = _iqr / (st_sz_all["median"] + _eps)

        _num_sym = st_sz_all["p75"] + st_sz_all["p25"] - (2.0 * st_sz_all["median"])
        _den_sym = abs(st_sz_all["p75"] - st_sz_all["p25"])
        _disp = _num_sym / (_den_sym + _eps)
        feat["dispersion_symmetry"] = float(np.clip(_disp, -1.0, 1.0))

        _bytes_up = float(up_sz.sum()) if up_sz.size > 0 else 0.0
        _bytes_down = float(down_sz.sum()) if down_sz.size > 0 else 0.0
        feat["direction_balance_bytes"] = (_bytes_up - _bytes_down) / (_bytes_up + _bytes_down + _eps)

        _pkts_up = float(up_sz.size)
        _pkts_down = float(down_sz.size)
        feat["direction_balance_packets"] = (_pkts_up - _pkts_down) / (_pkts_up + _pkts_down + _eps)

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

    # Final cleanup
    id_cols = {"flow_id", "capture_id"}
    for c in out.columns:
        if c not in id_cols:
            out[c] = pd.to_numeric(out[c], errors="raise")

    return out
