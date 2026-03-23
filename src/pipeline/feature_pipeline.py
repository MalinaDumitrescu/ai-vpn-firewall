# src/pipeline/feature_pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

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

# Metadata / forbidden columns: must never enter model features
FORBIDDEN_COLS = {
    "app",
    "file_names",
    "connection_str",
    "source_file",
    "source_capture_id",
}

# Explicit exclusions from model input
EXCLUDE_FROM_MODEL = {
    "q_min_packets_ok",
    "sample_weight",
    "q_window_complete",
    "q_packet_count",
    "tot_pkt",
}


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
    Robust feature pipeline for thesis and evaluation use.

    Behavior:
    - excludes all metadata / forbidden columns
    - keeps deterministic feature ordering
    - applies clip + log1p on heavy-tailed numeric features
    - scales continuous features with StandardScaler
    - leaves q_* passthrough columns unscaled if any survive

    Important:
    - source_capture_id is metadata only and must never become a model feature
    """
    feature_cols: Optional[List[str]] = None
    scale_cols: Optional[List[str]] = None
    passthrough_cols: Optional[List[str]] = None
    scaler: Optional[StandardScaler] = None
    clip_q: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def fit(
        self,
        df_features: pd.DataFrame,
        fit_provenance: Optional[Dict[str, Any]] = None
    ) -> "FeaturePipeline":
        missing_req = [c for c in (ID_COLS + [LABEL_COL]) if c not in df_features.columns]
        if missing_req:
            raise ValueError(f"Features DF missing required columns: {missing_req}")

        cols_to_exclude_from_features = (
            set(ID_COLS + [LABEL_COL, SPLIT_COL, DATASET_COL])
            | FORBIDDEN_COLS
            | EXCLUDE_FROM_MODEL
        )

        X_potential_features = df_features.drop(
            columns=[c for c in cols_to_exclude_from_features if c in df_features.columns],
            errors="ignore",
        ).copy()

        feat_cols = list(X_potential_features.columns)

        if not feat_cols:
            raise ValueError("No feature columns detected after exclusions.")

        # Safety check against metadata leakage by name
        suspicious = [
            c for c in feat_cols
            if any(x in c.lower() for x in ["capture", "source_file", "source_capture", "dataset", "split"])
        ]
        if suspicious:
            raise ValueError(
                "Potential metadata leakage detected in feature candidates: "
                f"{suspicious}"
            )

        X_processed = _ensure_numeric_finite(X_potential_features)

        passthrough = [c for c in feat_cols if c.startswith("q_")]
        scale = [c for c in feat_cols if c not in passthrough]

        if not scale:
            raise ValueError("No continuous columns to scale. Check feature extraction.")

        heavy_tailed_cols = []
        for c in scale:
            if "iat" in c or c.startswith("sz_") or c.startswith("h_"):
                heavy_tailed_cols.append(c)

        clip_q = {}
        for c in heavy_tailed_cols:
            q995 = float(X_processed[c].quantile(0.995))
            clip_q[c] = q995
            X_processed[c] = X_processed[c].clip(lower=0.0, upper=q995)
            X_processed[c] = np.log1p(X_processed[c])

        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X_processed[scale].to_numpy(dtype=float))

        self.feature_cols = feat_cols
        self.scale_cols = scale
        self.passthrough_cols = passthrough
        self.scaler = scaler
        self.clip_q = clip_q
        self.metadata = fit_provenance or {}
        return self

    def model_feature_names(self) -> List[str]:
        if self.scale_cols is None or self.passthrough_cols is None:
            raise RuntimeError("Pipeline is not fitted or loaded.")
        return list(self.scale_cols) + list(self.passthrough_cols)

    def transform(
        self,
        df_features: pd.DataFrame,
        *,
        strict: bool = True,
    ) -> pd.DataFrame:
        if (
            self.feature_cols is None
            or self.scale_cols is None
            or self.passthrough_cols is None
            or self.scaler is None
        ):
            raise RuntimeError("Pipeline is not fitted. Call fit() first or load() artifacts.")

        required_cols = ID_COLS + [LABEL_COL]
        for c in required_cols:
            if c not in df_features.columns:
                raise ValueError(f"Features DF missing required column: {c}")

        out = df_features[required_cols].copy()

        # preserve metadata for downstream analysis only when present
        for meta_col in [SPLIT_COL, DATASET_COL]:
            if meta_col in df_features.columns:
                out[meta_col] = df_features[meta_col].values

        missing_feats = [c for c in self.feature_cols if c not in df_features.columns]
        if missing_feats and strict:
            raise ValueError(
                f"Missing {len(missing_feats)} expected feature columns at transform. "
                f"Examples: {missing_feats[:10]}"
            )

        if strict:
            allowed = (
                set(ID_COLS + [LABEL_COL, SPLIT_COL, DATASET_COL])
                | FORBIDDEN_COLS
                | EXCLUDE_FROM_MODEL
                | set(self.feature_cols)
            )
            current = set(df_features.columns)
            unexpected = current - allowed
            if unexpected:
                raise ValueError(
                    f"Strict mode: Input contains {len(unexpected)} unexpected columns. "
                    f"Examples: {sorted(list(unexpected))[:10]}"
                )

        X = df_features.copy()
        for c in missing_feats:
            X[c] = 0.0

        X = X[self.feature_cols].copy()
        X = _ensure_numeric_finite(X)

        if self.clip_q:
            for c, q995 in self.clip_q.items():
                if c in X.columns:
                    X[c] = X[c].clip(lower=0.0, upper=q995)
                    X[c] = np.log1p(X[c])

        scaled_arr = self.scaler.transform(X[self.scale_cols].to_numpy(dtype=float))
        Xs = pd.DataFrame(scaled_arr, columns=self.scale_cols, index=df_features.index)

        if self.passthrough_cols:
            Xp = X[self.passthrough_cols].copy()
        else:
            Xp = pd.DataFrame(index=df_features.index)

        return pd.concat([out, Xs, Xp], axis=1)

    def save(self, art: FeatureArtifacts, *, feature_config_hash: str) -> None:
        if (
            self.feature_cols is None
            or self.scale_cols is None
            or self.passthrough_cols is None
            or self.scaler is None
        ):
            raise RuntimeError("Cannot save an unfitted pipeline.")

        meta = {
            "feature_cols": self.feature_cols,
            "scale_cols": self.scale_cols,
            "passthrough_cols": self.passthrough_cols,
            "model_feature_order": self.model_feature_names(),
            "id_cols": ID_COLS,
            "label_col": LABEL_COL,
            "metadata": self.metadata,
            "clip_q": self.clip_q,
            "forbidden_cols": sorted(FORBIDDEN_COLS),
            "excluded_from_model": sorted(EXCLUDE_FROM_MODEL),
        }
        save_json(art.feature_columns_json, meta)
        save_pickle(art.scaler_pkl, self.scaler)
        write_text(art.feature_config_hash_txt, feature_config_hash.strip() + "\n")

    @staticmethod
    def load(art: FeatureArtifacts) -> "FeaturePipeline":
        meta = load_json(art.feature_columns_json)
        scaler = load_pickle(art.scaler_pkl)

        if isinstance(meta, list):
            return FeaturePipeline(
                feature_cols=meta,
                scale_cols=meta,
                passthrough_cols=[],
                scaler=scaler,
                metadata={},
                clip_q={},
            )

        feature_cols = meta["feature_cols"]
        scale_cols = meta["scale_cols"]
        passthrough_cols = meta.get("passthrough_cols", [])
        metadata = meta.get("metadata", {})
        clip_q = meta.get("clip_q", {})

        overlap = set(scale_cols).intersection(passthrough_cols)
        if overlap:
            raise ValueError(
                f"Invalid feature metadata: overlap between scale_cols and passthrough_cols: {sorted(overlap)}"
            )

        return FeaturePipeline(
            feature_cols=feature_cols,
            scale_cols=scale_cols,
            passthrough_cols=passthrough_cols,
            scaler=scaler,
            metadata=metadata,
            clip_q=clip_q,
        )