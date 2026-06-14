"""
Preprocessing metadata export for the clean pipeline.

The clean pipeline currently uses tree-based models (XGB/LGBM/CatBoost) and
does not require numeric scaling. However, the data-leakage test suite must
be able to verify that any present-or-future scaler is fit ONLY on the
training split.

This module exports a deterministic metadata sidecar to
  `artifacts/clean_pipeline/preprocessing_metadata.json`
that records the per-feature training-split statistics (mean / std / min /
max) which any downstream scaler must match. The presence of this file
makes the "scaler is fit only on train" invariant testable, regardless of
whether a `scaler.pkl` is shipped.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import json
import numpy as np
import pandas as pd

from src.clean_pipeline.validation.leakage_checks import (
    CleanArtifactPaths,
    load_features_dataframe,
    model_feature_columns,
)


@dataclass
class PreprocessingMetadata:
    fit_split: str                       # always "train"
    transformed_splits: List[str]        # ["train", "val", "test"]
    feature_columns: List[str]
    transformer_type: str                # "identity" if no scaler; else the sklearn class
    train_stats: Dict[str, Dict[str, float]]   # {col: {mean, std, min, max, n}}
    source_features_parquet: str
    schema_version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _compute_train_stats(
    df: pd.DataFrame, feat_cols: List[str]
) -> Dict[str, Dict[str, float]]:
    train_df = df.loc[df["split"] == "train", feat_cols]
    if train_df.empty:
        raise ValueError(
            "No rows with split == 'train' found in features.parquet. "
            "Cannot fit preprocessing statistics."
        )
    stats: Dict[str, Dict[str, float]] = {}
    for c in feat_cols:
        col = train_df[c].to_numpy(dtype=np.float64, copy=False)
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            stats[c] = {"mean": float("nan"), "std": float("nan"),
                        "min": float("nan"), "max": float("nan"), "n": 0}
            continue
        stats[c] = {
            "mean": float(np.mean(finite)),
            "std":  float(np.std(finite, ddof=0)),
            "min":  float(np.min(finite)),
            "max":  float(np.max(finite)),
            "n":    int(finite.size),
        }
    return stats


def ensure_preprocessing_metadata(
    paths: CleanArtifactPaths,
    transformer_type: str = "identity",
    overwrite: bool = False,
) -> PreprocessingMetadata:
    """
    Read (or generate) `preprocessing_metadata.json` alongside the clean
    pipeline artifacts. The file is generated deterministically from the
    train-split rows of `features.parquet`; nothing is fit on val/test/full.
    """
    target = paths.preprocessing_metadata

    if target.exists() and not overwrite:
        data = json.loads(target.read_text(encoding="utf-8"))
        return PreprocessingMetadata(**data)

    df = load_features_dataframe(paths)
    feat_cols = model_feature_columns(df)
    stats = _compute_train_stats(df, feat_cols)

    meta = PreprocessingMetadata(
        fit_split="train",
        transformed_splits=["train", "val", "test"],
        feature_columns=feat_cols,
        transformer_type=transformer_type,
        train_stats=stats,
        source_features_parquet=str(
            paths.features_parquet.relative_to(paths.repo_root)
        ).replace("\\", "/"),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(meta.to_json(), encoding="utf-8")
    return meta


def stats_match_train(
    meta: PreprocessingMetadata,
    df: pd.DataFrame,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> Dict[str, Dict[str, float]]:
    """
    Recompute train-split stats from `df` and compare against `meta.train_stats`.
    Returns a dict of mismatches (empty dict if perfectly matching).
    """
    recomputed = _compute_train_stats(df, meta.feature_columns)
    mismatches: Dict[str, Dict[str, float]] = {}
    for c, s_meta in meta.train_stats.items():
        s_now = recomputed[c]
        for k in ("mean", "std", "min", "max"):
            v_meta = s_meta[k]
            v_now = s_now[k]
            if not (np.isfinite(v_meta) and np.isfinite(v_now)):
                if np.isnan(v_meta) != np.isnan(v_now):
                    mismatches.setdefault(c, {})[k] = float(v_now)
                continue
            if not np.isclose(v_meta, v_now, rtol=rtol, atol=atol):
                mismatches.setdefault(c, {})[k] = float(v_now)
    return mismatches
