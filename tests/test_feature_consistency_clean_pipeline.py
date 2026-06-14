import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path

import sys
sys.path.append('.')

from src.clean_pipeline.feature_extractor import extract_features_batch
from src.clean_pipeline.feature_families import get_family, family_has_risky_features, validate_family_in_dataframe

# Fixtures
@pytest.fixture(scope="module")
def features_df():
    features_path = Path('artifacts/clean_pipeline/features.parquet')
    if not features_path.exists():
        pytest.skip("features.parquet not found. Run the clean pipeline first.")
    return pd.read_parquet(features_path)

@pytest.fixture(scope="module")
def feature_columns():
    columns_path = Path('artifacts/clean_pipeline/feature_columns.json')
    if not columns_path.exists():
        pytest.skip("feature_columns.json not found. Run the clean pipeline first.")
    with open(columns_path) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def feature_family():
    # Use the family from config or default
    return 'safe_core_plus_temporal'

# 1. Raw flow schema contains required columns
def test_raw_flow_schema_contains_required_columns():
    required = {"flow_id", "capture_id", "dataset", "label", "timestamps", "sizes", "directions"}
    # Minimal valid flow
    flow = {
        "flow_id": "f1",
        "capture_id": "c1",
        "dataset": "test",
        "label": 1,
        "timestamps": [0.0, 0.1, 0.2],
        "sizes": [100, 120, 140],
        "directions": [1, 0, 1],
    }
    # Should not raise
    df = pd.DataFrame([flow])
    extract_features_batch(df, family='safe_core_plus_temporal', progress=False)
    # Remove one required column
    for col in required:
        bad = df.drop(columns=[col])
        with pytest.raises(ValueError, match=f"{col}"):
            extract_features_batch(bad, family='safe_core_plus_temporal', progress=False)

# 2. Extracted features match feature_columns.json
def test_extracted_features_match_feature_columns_json(features_df, feature_columns):
    # Only model features (not metadata)
    model_feats = [c for c in feature_columns]
    df_feats = [c for c in features_df.columns if c in model_feats]
    assert df_feats == model_feats, f"Feature columns mismatch or order changed.\nExpected: {model_feats}\nFound: {df_feats}"
    assert len(df_feats) == len(model_feats)

# 3. Final feature family is direction-invariant
def test_final_feature_family_is_direction_invariant(feature_family):
    risky_patterns = [
        'fwd', 'bwd', 'forward', 'backward', 'src', 'dst', 'client', 'server', 'directional_ratio'
    ]
    family = get_family(feature_family)
    risky = [f for f in family if any(p in f.lower() for p in risky_patterns)]
    assert not risky, f"Direction-dependent risky features found: {risky}"
    # Optionally: check for exact 21-feature set
    if feature_family == 'safe_core_plus_temporal':
        assert len(family) == 21, f"Expected 21 features, got {len(family)}"

# 4. Rate features recomputed match stored values
def test_rate_features_recomputed_match_stored_values(features_df, feature_family):
    # Only rows with positive flow_duration
    df = features_df[features_df['flow_duration'] > 0].copy()
    tol = 1e-6
    # Recompute
    recomputed_packet_rate = df['total_packets'] / df['flow_duration']
    recomputed_byte_rate = df['total_bytes'] / df['flow_duration']
    # Compare
    assert np.allclose(df['packet_rate'], recomputed_packet_rate, atol=tol), (
        f"Packet rate mismatch. Max abs error: {np.max(np.abs(df['packet_rate'] - recomputed_packet_rate))}")
    assert np.allclose(df['byte_rate'], recomputed_byte_rate, atol=tol), (
        f"Byte rate mismatch. Max abs error: {np.max(np.abs(df['byte_rate'] - recomputed_byte_rate))}")
    # No inf/nan
    for col in ['packet_rate', 'byte_rate']:
        assert np.isfinite(df[col]).all(), f"Non-finite values in {col}"
    assert (df['flow_duration'] > 0).all()

# 5. No NaN/Inf in final feature matrix
def test_no_nan_inf_in_final_feature_matrix(features_df, feature_family):
    family = get_family(feature_family)
    for col in family:
        assert col in features_df.columns, f"Missing feature: {col}"
        assert features_df[col].notnull().all(), f"NaN in {col}"
        assert np.isfinite(features_df[col]).all(), f"Inf in {col}"

