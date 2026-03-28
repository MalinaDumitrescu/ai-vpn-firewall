from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from src.pipeline.artifacts import (
    FeatureArtifacts,
    load_json,
    load_pickle,
    save_json,
    save_pickle,
    write_text,
)

ID_COLS = ["flow_id", "capture_id", "source_file", "source_capture_id"]
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
    "q_window_complete",
    "sample_weight",
    "q_packet_count",  # training-window / availability leak
    "tot_pkt",  # session-length leak
    "source_file",  # ID leak
    "source_capture_id"  # ID leak
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


def _per_capture_normalize(
        X: pd.DataFrame,
        capture_ids: pd.Series,
        scale_cols: List[str],
) -> pd.DataFrame:
    """
    Per-capture normalization:
        z = (x - capture_mean) / capture_std

    Notes:
    - uses capture_id provided separately, so the feature dataframe itself stays clean
    - std==0 or std==NaN (e.g. singleton capture) is replaced with 1.0
    - resulting constant-within-capture features become 0.0 after centering
    """
    out = X.copy()

    if len(capture_ids) != len(out):
        raise ValueError("capture_ids length does not match feature matrix length.")

    for c in scale_cols:
        if c not in out.columns:
            continue

        means = out[c].groupby(capture_ids).transform("mean")
        stds = out[c].groupby(capture_ids).transform("std")
        stds = stds.replace(0, np.nan).fillna(1.0)

        out[c] = (out[c] - means) / stds

    return out


@dataclass
class FeaturePipeline:
    """
    Robust feature pipeline for thesis and evaluation use.

    Behavior:
    - excludes all metadata / forbidden columns
    - excludes histogram features entirely:
        h_size_all_*
        h_iat_all_*
    - excludes known leakage / fingerprint columns
    - keeps deterministic feature ordering
    - applies clip + log1p on heavy-tailed numeric features
    - applies per-capture normalization: (x - capture_mean) / capture_std
    - then applies per-dataset QuantileTransformer (rank normalization)
    - leaves q_* passthrough columns unscaled if any survive

    Important:
    - source_capture_id is metadata only and must never become a model feature
    - per-capture normalization is suitable for offline evaluation, not strict online-first-flow deployment claims
    """
    feature_cols: Optional[List[str]] = None
    scale_cols: Optional[List[str]] = None
    passthrough_cols: Optional[List[str]] = None
    scaler: Optional[Dict[str, QuantileTransformer]] = None
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

        # Remove histogram features entirely
        feat_cols = [
            c for c in feat_cols
            if not (c.startswith("h_size_all_") or c.startswith("h_iat_all_"))
        ]

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

        X_potential_features = X_potential_features[feat_cols].copy()
        X_processed = _ensure_numeric_finite(X_potential_features)

        passthrough = [c for c in feat_cols if c.startswith("q_")]
        scale = [c for c in feat_cols if c not in passthrough]

        if not scale:
            raise ValueError("No continuous columns to scale. Check feature extraction.")

        heavy_tailed_cols = []
        for c in scale:
            if "iat" in c or c.startswith("sz_"):
                heavy_tailed_cols.append(c)

        clip_q: Dict[str, float] = {}
        for c in heavy_tailed_cols:
            q995 = float(X_processed[c].quantile(0.995))
            clip_q[c] = q995
            X_processed[c] = X_processed[c].clip(lower=0.0, upper=q995)
            X_processed[c] = np.log1p(X_processed[c])

        # Apply per-capture normalization before global scaling
        capture_ids = df_features["capture_id"]
        X_processed = _per_capture_normalize(X_processed, capture_ids, scale)

        scalers = {}
        datasets = df_features[DATASET_COL].unique()
        for ds in datasets:
            mask = df_features[DATASET_COL] == ds
            X_ds = X_processed.loc[mask, scale]
            qt = QuantileTransformer(output_distribution="uniform", n_quantiles=min(1000, len(X_ds)), random_state=42)
            qt.fit(X_ds.to_numpy(dtype=float))
            scalers[ds] = qt

        self.feature_cols = feat_cols
        self.scale_cols = scale
        self.passthrough_cols = passthrough
        self.scaler = scalers
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

        # Preserve metadata for downstream analysis only when present
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

        # Apply per-capture normalization before global scaling
        capture_ids = df_features["capture_id"]
        X = _per_capture_normalize(X, capture_ids, self.scale_cols)

        Xs = pd.DataFrame(index=df_features.index, columns=self.scale_cols)
        for ds in self.scaler.keys():
            mask = df_features[DATASET_COL] == ds
            if mask.any():
                X_ds = X.loc[mask, self.scale_cols]
                scaled_ds = self.scaler[ds].transform(X_ds.to_numpy(dtype=float))
                Xs.loc[mask] = scaled_ds

        if self.passthrough_cols:
            Xp = X[self.passthrough_cols].copy()
        else:
            Xp = pd.DataFrame(index=df_features.index)

        # DEBUG: Ensure all feature columns in final output are numeric
        final_df = pd.concat([out, Xs, Xp], axis=1)

        # Check feature columns specifically
        for col in self.scale_cols + self.passthrough_cols:
            if col in final_df.columns and final_df[col].dtype == 'object':
                print(f"WARNING: Feature column '{col}' has object dtype, converting to numeric")
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0.0)

        return final_df

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
            "histograms_removed": True,
            "per_capture_normalization": True,
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