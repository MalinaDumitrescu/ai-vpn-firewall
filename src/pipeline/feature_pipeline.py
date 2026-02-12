# src/pipeline/feature_pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipeline.artifacts import FeatureArtifacts, load_json, load_pickle, save_json, save_pickle, write_text


ID_COLS = ["flow_id", "capture_id"]
LABEL_COL = "label"


def _is_binary_series(s: pd.Series) -> bool:
    """
    Treat a feature as binary if its (cleaned) unique values are subset of {0,1}.
    This keeps quality flags auditable (no scaling).
    """
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return False
    vals = set(x.unique().tolist())
    return vals.issubset({0.0, 1.0, 0, 1})


def _ensure_numeric_finite(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="raise")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    arr = out.to_numpy(dtype=float, copy=False)
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite values found in features after cleanup.")
    return out


@dataclass
class FeaturePipeline:
    """
    Fit on VNAT TRAIN only, apply everywhere.
    - Enforces the same feature columns everywhere
    - Scales continuous numeric features
    - Leaves binary indicators (0/1) unscaled
    """
    feature_cols: Optional[List[str]] = None
    scale_cols: Optional[List[str]] = None
    passthrough_cols: Optional[List[str]] = None
    scaler: Optional[StandardScaler] = None

    def fit(self, df_features: pd.DataFrame) -> "FeaturePipeline":
        missing = [c for c in (ID_COLS + [LABEL_COL]) if c not in df_features.columns]
        if missing:
            raise ValueError(f"Features DF missing required columns: {missing}")

        feat_cols = [c for c in df_features.columns if c not in set(ID_COLS + [LABEL_COL])]
        if not feat_cols:
            raise ValueError("No feature columns detected (everything got filtered out).")

        X = _ensure_numeric_finite(df_features[feat_cols])

        passthrough = [c for c in feat_cols if _is_binary_series(X[c])]
        scale = [c for c in feat_cols if c not in passthrough]

        if not scale:
            raise ValueError("No continuous columns to scale (everything looks binary). Check feature extraction.")

        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X[scale].to_numpy(dtype=float))

        self.feature_cols = feat_cols
        self.scale_cols = scale
        self.passthrough_cols = passthrough
        self.scaler = scaler
        return self

    def transform(self, df_features: pd.DataFrame) -> pd.DataFrame:
        if (
            self.feature_cols is None
            or self.scale_cols is None
            or self.passthrough_cols is None
            or self.scaler is None
        ):
            raise RuntimeError("Pipeline is not fitted. Call fit() first or load() artifacts.")

        for c in ID_COLS + [LABEL_COL]:
            if c not in df_features.columns:
                raise ValueError(f"Features DF missing required column: {c}")

        out = df_features[ID_COLS + [LABEL_COL]].copy()

        X = df_features.copy()
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = 0.0

        X = X[self.feature_cols].copy()
        X = _ensure_numeric_finite(X)

        scaled = self.scaler.transform(X[self.scale_cols].to_numpy(dtype=float))
        Xs = pd.DataFrame(scaled, columns=self.scale_cols, index=df_features.index)

        Xp = X[self.passthrough_cols].copy() if self.passthrough_cols else pd.DataFrame(index=df_features.index)

        return pd.concat([out, Xs, Xp], axis=1)

    def save(self, art: FeatureArtifacts, *, feature_config_hash: str) -> None:
        if (
            self.feature_cols is None
            or self.scale_cols is None
            or self.passthrough_cols is None
            or self.scaler is None
        ):
            raise RuntimeError("Cannot save an unfitted pipeline.")
        
        # Save metadata including scale/passthrough split
        meta = {
            "feature_cols": self.feature_cols,
            "scale_cols": self.scale_cols,
            "passthrough_cols": self.passthrough_cols,
        }
        save_json(art.feature_columns_json, meta)
        save_pickle(art.scaler_pkl, self.scaler)
        write_text(art.feature_config_hash_txt, feature_config_hash.strip() + "\n")

    @staticmethod
    def load(art: FeatureArtifacts) -> "FeaturePipeline":
        meta = load_json(art.feature_columns_json)
        scaler = load_pickle(art.scaler_pkl)
        
        # Handle legacy format (list of strings) vs new format (dict)
        if isinstance(meta, list):
             # Fallback for old artifacts: assume all are scaled
             return FeaturePipeline(
                 feature_cols=meta,
                 scale_cols=meta,
                 passthrough_cols=[],
                 scaler=scaler
             )
        
        return FeaturePipeline(
            feature_cols=meta["feature_cols"],
            scale_cols=meta["scale_cols"],
            passthrough_cols=meta["passthrough_cols"],
            scaler=scaler
        )
