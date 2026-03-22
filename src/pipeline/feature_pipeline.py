# src/pipeline/feature_pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Any

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
SPLIT_COL = "split"
DATASET_COL = "dataset"

# Explicitly forbidden columns that should never enter the model
FORBIDDEN_COLS = {"app", "file_names", "connection_str", "source_file"}

# Columns to explicitly exclude from model features (e.g. quality flags that are constant or not behavioral)
EXCLUDE_FROM_MODEL = {"q_min_packets_ok", "sample_weight", "q_window_complete", "q_packet_count", "tot_pkt"}


def _ensure_numeric_finite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force numeric dtype, replace inf with NaN, fill NaN with 0.0, and validate finite.
    """
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
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
    - Applies robust numeric transform (clip + log1p) to heavy-tailed features
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
    clip_q: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def fit(self, df_features: pd.DataFrame, fit_provenance: Optional[Dict[str, Any]] = None) -> "FeaturePipeline":
        # Required columns
        missing_req = [c for c in (ID_COLS + [LABEL_COL]) if c not in df_features.columns]
        if missing_req:
            raise ValueError(f"Features DF missing required columns: {missing_req}")

        # Identify all columns that are NOT features
        # Ensure we only try to drop columns that actually exist in df_features
        cols_to_exclude_from_features = set(ID_COLS + [LABEL_COL, SPLIT_COL, DATASET_COL]) | FORBIDDEN_COLS | EXCLUDE_FROM_MODEL
        
        # Create a temporary DataFrame containing only the potential feature columns
        # by dropping the known non-feature columns.
        X_potential_features = df_features.drop(
            columns=[col for col in cols_to_exclude_from_features if col in df_features.columns],
            errors='ignore' # Ignore if a column to drop doesn't exist
        ).copy()

        feat_cols = list(X_potential_features.columns)

        if not feat_cols:
            raise ValueError("No feature columns detected (everything got filtered out).")

        # Enforce numeric + finite on these feature columns
        X_processed = _ensure_numeric_finite(X_potential_features)

        # Explicit passthrough policy: only q_* columns (quality flags)
        passthrough = [c for c in feat_cols if c.startswith("q_")]
        scale = [c for c in feat_cols if c not in passthrough]

        if not scale:
            raise ValueError("No continuous columns to scale. Check feature extraction / feature naming.")
        # It's OK if passthrough is empty.

        # --- Robust Numeric Transform (Fit) ---
        # Identify heavy-tailed features
        heavy_tailed_cols = []
        for c in scale:
            if "iat" in c or c.startswith("sz_") or c.startswith("h_"):
                heavy_tailed_cols.append(c)
        
        clip_q = {}
        for c in heavy_tailed_cols:
            # Compute 99.5th percentile on TRAIN
            q995 = float(X_processed[c].quantile(0.995))
            clip_q[c] = q995
            
            # Apply transform in-place for scaler fitting
            # 1. Clip lower to 0
            X_processed[c] = X_processed[c].clip(lower=0.0)
            # 2. Clip upper to q995
            X_processed[c] = X_processed[c].clip(upper=q995)
            # 3. Log1p
            X_processed[c] = np.log1p(X_processed[c])

        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X_processed[scale].to_numpy(dtype=float)) # Use X_processed here

        self.feature_cols = feat_cols
        self.scale_cols = scale
        self.passthrough_cols = passthrough
        self.scaler = scaler
        self.clip_q = clip_q
        self.metadata = fit_provenance or {}
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
          - raises if any unexpected columns are present (to prevent leakage/schema drift)

        strict=False:
          - fills missing expected feature columns with 0.0 and continues (useful for debugging)
          - ignores unexpected columns
        """
        if self.feature_cols is None or self.scale_cols is None or self.passthrough_cols is None or self.scaler is None:
            raise RuntimeError("Pipeline is not fitted. Call fit() first or load() artifacts.")

        # The split column is not required for transforming, so we don't check for it.
        required_cols = ID_COLS + [LABEL_COL]
        for c in required_cols:
            if c not in df_features.columns:
                raise ValueError(f"Features DF missing required column: {c}")

        out = df_features[required_cols].copy()

        # Check/fill missing expected feature columns
        missing_feats = [c for c in self.feature_cols if c not in df_features.columns]
        if missing_feats and strict:
            raise ValueError(
                f"Missing {len(missing_feats)} expected feature columns at transform. "
                f"Examples: {missing_feats[:10]}"
            )
        
        # Check for unexpected columns (Strict Mode)
        if strict:
            # Allowed: ID cols, Label, Split, Dataset, Forbidden (ignored), Excluded, and Expected Features
            allowed = set(ID_COLS + [LABEL_COL, SPLIT_COL, DATASET_COL]) | FORBIDDEN_COLS | EXCLUDE_FROM_MODEL | set(self.feature_cols)
            current = set(df_features.columns)
            unexpected = current - allowed
            if unexpected:
                raise ValueError(
                    f"Strict mode: Input contains {len(unexpected)} unexpected columns not seen at fit time. "
                    f"Examples: {sorted(list(unexpected))[:10]}\n"
                    "This ensures you are not accidentally passing new features that the model ignores."
                )

        X = df_features.copy()
        for c in missing_feats:
            X[c] = 0.0  # only used if strict=False

        # Keep only expected feature cols in the learned order
        X = X[self.feature_cols].copy()
        X = _ensure_numeric_finite(X)

        # --- Robust Numeric Transform (Apply) ---
        if self.clip_q:
            for c, q995 in self.clip_q.items():
                if c in X.columns:
                    # 1. Clip lower to 0
                    X[c] = X[c].clip(lower=0.0)
                    # 2. Clip upper to q995
                    X[c] = X[c].clip(upper=q995)
                    # 3. Log1p
                    X[c] = np.log1p(X[c])

        # Scale continuous
        scaled_arr = self.scaler.transform(X[self.scale_cols].to_numpy(dtype=float))
        Xs = pd.DataFrame(scaled_arr, columns=self.scale_cols, index=df_features.index)

        # Passthrough q_* (unscaled)
        if self.passthrough_cols:
            Xp = X[self.passthrough_cols].copy()
        else:
            Xp = pd.DataFrame(index=df_features.index)
        
        # CHANGED: Include q_min_packets_ok in output if it exists in input, even if not a model feature
        # This is needed for filtering later in the notebook
        if "q_min_packets_ok" in df_features.columns and "q_min_packets_ok" not in out.columns:
             out["q_min_packets_ok"] = df_features["q_min_packets_ok"]

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
            "metadata": self.metadata,
            "clip_q": self.clip_q, # Save clip values
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
                metadata={},
                clip_q={},
            )

        # New format: dict with explicit scale/passthrough
        feature_cols = meta["feature_cols"]
        scale_cols = meta["scale_cols"]
        passthrough_cols = meta.get("passthrough_cols", [])
        metadata = meta.get("metadata", {})
        clip_q = meta.get("clip_q", {})

        # Safety: ensure no overlap and preserve order as stored
        overlap = set(scale_cols).intersection(passthrough_cols)
        if overlap:
            raise ValueError(f"Invalid feature metadata: scale_cols overlap passthrough_cols: {sorted(overlap)[:10]}")

        return FeaturePipeline(
            feature_cols=feature_cols,
            scale_cols=scale_cols,
            passthrough_cols=passthrough_cols,
            scaler=scaler,
            metadata=metadata,
            clip_q=clip_q,
        )
