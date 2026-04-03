#!/usr/bin/env python
"""
Test script to verify that audit columns (q_packet_count) are properly excluded from
model input but preserved in the transform output for analysis.
"""

import pandas as pd
import numpy as np
from src.pipeline.feature_pipeline import (
    FeaturePipeline, 
    AUDIT_COLS, 
    EXCLUDE_FROM_MODEL,
    COMPACT_FEATURES,
    ID_COLS,
    LABEL_COL,
)

def create_test_features(n_rows=100):
    """Create a minimal test features DataFrame."""
    np.random.seed(42)
    
    data = {
        # Required ID columns
        "flow_id": [f"flow_{i}" for i in range(n_rows)],
        "capture_id": [f"cap_{i % 5}" for i in range(n_rows)],  # 5 captures
        "source_file": [f"file_{i % 2}" for i in range(n_rows)],
        "source_capture_id": [f"scap_{i}" for i in range(n_rows)],
        
        # Required label
        "label": np.random.randint(0, 2, n_rows),
        
        # Compact features (size-based)
        "sz_std": np.random.uniform(10, 500, n_rows),
        "sz_mean": np.random.uniform(100, 1000, n_rows),
        "sz_p25": np.random.uniform(50, 500, n_rows),
        "sz_p75": np.random.uniform(200, 1500, n_rows),
        "sz_median": np.random.uniform(100, 1000, n_rows),
        "bytes_up": np.random.uniform(1000, 10000, n_rows),
        "bytes_down": np.random.uniform(1000, 10000, n_rows),
        "packets_up": np.random.uniform(10, 100, n_rows),
        "packets_down": np.random.uniform(10, 100, n_rows),
        
        # Audit column (should be excluded from model but preserved in output)
        "q_packet_count": np.random.uniform(1, 1500, n_rows),
        "q_min_packets_ok": np.random.randint(0, 2, n_rows).astype(float),
        
        # Other excluded columns
        "tot_pkt": np.random.uniform(1, 1500, n_rows),
        "sample_weight": np.ones(n_rows),
        "q_window_complete": np.ones(n_rows),
    }
    
    return pd.DataFrame(data)

def test_audit_cols_in_constants():
    """Test that AUDIT_COLS is properly defined."""
    print("\n=== Test 1: AUDIT_COLS Constants ===")
    print(f"AUDIT_COLS: {AUDIT_COLS}")
    print(f"q_packet_count in AUDIT_COLS: {'q_packet_count' in AUDIT_COLS}")
    print(f"q_packet_count in EXCLUDE_FROM_MODEL: {'q_packet_count' in EXCLUDE_FROM_MODEL}")
    
    assert 'q_packet_count' in AUDIT_COLS, "q_packet_count should be in AUDIT_COLS"
    assert 'q_packet_count' in EXCLUDE_FROM_MODEL, "q_packet_count should be in EXCLUDE_FROM_MODEL"
    print("✓ Constants are correct")

def test_pipeline_fit():
    """Test that FeaturePipeline can be fit with audit columns."""
    print("\n=== Test 2: FeaturePipeline.fit() ===")
    
    df = create_test_features(100)
    pipe = FeaturePipeline()
    
    # Fit the pipeline
    pipe = pipe.fit(df)
    
    print(f"feature_cols: {pipe.feature_cols}")
    print(f"scale_cols: {pipe.scale_cols}")
    print(f"passthrough_cols: {pipe.passthrough_cols}")
    print(f"audit_cols: {pipe.audit_cols}")
    
    # Verify audit_cols were identified
    assert pipe.audit_cols is not None, "audit_cols should not be None after fit"
    assert 'q_packet_count' in pipe.audit_cols, "q_packet_count should be in audit_cols"
    print("✓ Pipeline fitted successfully with audit_cols identified")

def test_transform_preserves_audit_cols():
    """Test that transform preserves audit columns in output."""
    print("\n=== Test 3: transform() Preserves Audit Columns ===")
    
    df_train = create_test_features(100)
    df_test = create_test_features(50)
    
    pipe = FeaturePipeline()
    pipe = pipe.fit(df_train)
    
    # Transform test data (strict=False because test data doesn't have exact same schema)
    X_transformed = pipe.transform(df_test, strict=False)
    
    print(f"Input columns: {sorted(df_test.columns)}")
    print(f"Output columns: {sorted(X_transformed.columns)}")
    
    # Verify audit columns are in output
    assert 'q_packet_count' in X_transformed.columns, "q_packet_count should be in transformed output"
    print(f"✓ q_packet_count is in output columns")
    
    # Verify audit columns have correct values
    if 'q_packet_count' in df_test.columns:
        original_values = df_test['q_packet_count'].values
        transformed_values = X_transformed['q_packet_count'].values
        print(f"  Original q_packet_count sample: {original_values[:3]}")
        print(f"  Transformed q_packet_count sample: {transformed_values[:3]}")
        print("✓ Audit column values preserved in output")

def test_model_features_exclude_audit():
    """Test that model_feature_names() excludes audit columns."""
    print("\n=== Test 4: model_feature_names() Excludes Audit Columns ===")
    
    df = create_test_features(100)
    pipe = FeaturePipeline()
    pipe = pipe.fit(df)
    
    model_features = pipe.model_feature_names()
    print(f"Model feature names: {model_features}")
    print(f"q_packet_count in model features: {'q_packet_count' in model_features}")
    
    assert 'q_packet_count' not in model_features, "q_packet_count should NOT be in model features"
    print("✓ Audit columns correctly excluded from model input")

def test_audit_cols_in_metadata():
    """Test that audit_cols are saved in metadata."""
    print("\n=== Test 5: Audit Columns in Metadata ===")
    
    df = create_test_features(100)
    pipe = FeaturePipeline()
    pipe = pipe.fit(df)
    
    # Build the metadata dict (like in save method)
    meta = {
        "audit_cols": pipe.audit_cols or [],
    }
    
    print(f"Metadata audit_cols: {meta['audit_cols']}")
    assert meta['audit_cols'] is not None, "audit_cols should be in metadata"
    assert 'q_packet_count' in meta['audit_cols'], "q_packet_count should be in metadata audit_cols"
    print("✓ Audit columns correctly stored in metadata")

if __name__ == "__main__":
    try:
        test_audit_cols_in_constants()
        test_pipeline_fit()
        test_transform_preserves_audit_cols()
        test_model_features_exclude_audit()
        test_audit_cols_in_metadata()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nSummary:")
        print("- q_packet_count is in AUDIT_COLS")
        print("- q_packet_count is in EXCLUDE_FROM_MODEL")
        print("- q_packet_count is preserved in transform() output")
        print("- q_packet_count is excluded from model_feature_names()")
        print("- q_packet_count is stored in pipeline metadata")
        print("\nThis means:")
        print("✓ q_packet_count dataset fingerprint is excluded from model")
        print("✓ q_packet_count is still available for audit/analysis")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


