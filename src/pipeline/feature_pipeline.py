# src/pipeline/feature_pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipeline.artifacts import (
    FeatureArtifacts,
    load_json,
    load_pickle,
    save_json,
    save_pickle,
    write_text,
)

ID_COLS = ["flow_id", "capture_id"]
LABEL_COL = "label"


def _ensure_numeric_finite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force numeric dtype, replace inf with NaN, fill NaN with 0.0, and validate finite.
    """
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

    - Enforces the same feature schema everywhere (fills missing with 0.0)
    - Scales continuous numeric features with StandardScaler (linear/affine scaling)
    - Leaves quality indicators (q_*) unscaled

    Notes:
    - XGBoost doesn't require scaling, but calibration/linear meta-models do.
    - Use model_feature_names() to get the deterministic column order for training/inference.
    """
    feature_cols: Optional[List[str]] = None
    scale_cols: Optional[List[str]] = None
    passthrough_cols: Optional[List[str]] = None
    scaler: Optional[StandardScaler] = None

    def fit(self, df_features: pd.DataFrame) -> "FeaturePipeline":
        # Required columns
        missing_req = [c for c in (ID_COLS + [LABEL_COL]) if c not in df_features.columns]
        if missing_req:
            raise ValueError(f"Features DF missing required columns: {missing_req}")

        # Feature columns = everything except IDs + label
        feat_cols = [c for c in df_features.columns if c not in set(ID_COLS + [LABEL_COL])]
        if not feat_cols:
            raise ValueError("No feature columns detected (everything got filtered out).")

        # Enforce numeric + finite on feature columns only
        X = _ensure_numeric_finite(df_features[feat_cols])

        # Explicit passthrough policy: only q_* columns (quality flags)
        passthrough = [c for c in feat_cols if c.startswith("q_")]
        scale = [c for c in feat_cols if c not in passthrough]

        if not scale:
            raise ValueError("No continuous columns to scale. Check feature extraction / feature naming.")
        # It's OK if passthrough is empty.

        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X[scale].to_numpy(dtype=float))

        self.feature_cols = feat_cols
        self.scale_cols = scale
        self.passthrough_cols = passthrough
        self.scaler = scaler
        return self

    def model_feature_names(self) -> List[str]:
        """
        Deterministic feature order for model X matrices.
        Always use this when building X for training/inference.
        """
        if self.scale_cols is None or self.passthrough_cols is None:
            raise RuntimeError("Pipeline is not fitted or loaded.")
        return list(self.scale_cols) + list(self.passthrough_cols)

    def transform(
        self,
        df_features: pd.DataFrame,
        *,
        strict: bool = True,
    ) -> pd.DataFrame:
        """
        Transform a features dataframe into a scaled dataframe.

        strict=True:
          - raises if any expected feature columns are missing (recommended for thesis/eval runs)

        strict=False:
          - fills missing expected feature columns with 0.0 and continues (useful for debugging)
        """
        if self.feature_cols is None or self.scale_cols is None or self.passthrough_cols is None or self.scaler is None:
            raise RuntimeError("Pipeline is not fitted. Call fit() first or load() artifacts.")

        for c in ID_COLS + [LABEL_COL]:
            if c not in df_features.columns:
                raise ValueError(f"Features DF missing required column: {c}")

        out = df_features[ID_COLS + [LABEL_COL]].copy()

        # Check/fill missing expected feature columns
        missing_feats = [c for c in self.feature_cols if c not in df_features.columns]
        if missing_feats and strict:
            raise ValueError(
                f"Missing {len(missing_feats)} expected feature columns at transform. "
                f"Examples: {missing_feats[:10]}"
            )

        X = df_features.copy()
        for c in missing_feats:
            X[c] = 0.0  # only used if strict=False

        # Keep only expected feature cols in the learned order
        X = X[self.feature_cols].copy()
        X = _ensure_numeric_finite(X)

        # Scale continuous
        scaled_arr = self.scaler.transform(X[self.scale_cols].to_numpy(dtype=float))
        Xs = pd.DataFrame(scaled_arr, columns=self.scale_cols, index=df_features.index)

        # Passthrough q_* (unscaled)
        if self.passthrough_cols:
            Xp = X[self.passthrough_cols].copy()
        else:
            Xp = pd.DataFrame(index=df_features.index)

        # Output: IDs + label + model features (scaled first, then passthrough)
        return pd.concat([out, Xs, Xp], axis=1)

    def save(self, art: FeatureArtifacts, *, feature_config_hash: str) -> None:
        if self.feature_cols is None or self.scale_cols is None or self.passthrough_cols is None or self.scaler is None:
            raise RuntimeError("Cannot save an unfitted pipeline.")

        meta = {
            "feature_cols": self.feature_cols,
            "scale_cols": self.scale_cols,
            "passthrough_cols": self.passthrough_cols,
            "model_feature_order": self.model_feature_names(),
            "id_cols": ID_COLS,
            "label_col": LABEL_COL,
        }
        save_json(art.feature_columns_json, meta)
        save_pickle(art.scaler_pkl, self.scaler)
        write_text(art.feature_config_hash_txt, feature_config_hash.strip() + "\n")

    @staticmethod
    def load(art: FeatureArtifacts) -> "FeaturePipeline":
        meta = load_json(art.feature_columns_json)
        scaler = load_pickle(art.scaler_pkl)

        # Legacy format: list[str] = feature columns only
        if isinstance(meta, list):
            return FeaturePipeline(
                feature_cols=meta,
                scale_cols=meta,
                passthrough_cols=[],
                scaler=scaler,
            )

        # New format: dict with explicit scale/passthrough
        feature_cols = meta["feature_cols"]
        scale_cols = meta["scale_cols"]
        passthrough_cols = meta.get("passthrough_cols", [])

        # Safety: ensure no overlap and preserve order as stored
        overlap = set(scale_cols).intersection(passthrough_cols)
        if overlap:
            raise ValueError(f"Invalid feature metadata: scale_cols overlap passthrough_cols: {sorted(overlap)[:10]}")

        return FeaturePipeline(
            feature_cols=feature_cols,
            scale_cols=scale_cols,
            passthrough_cols=passthrough_cols,
            scaler=scaler,
        )
