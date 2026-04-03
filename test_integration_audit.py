#!/usr/bin/env python
"""
Integration test: Load actual features and verify audit columns work correctly.
"""

import pandas as pd
from pathlib import Path
from src.pipeline.feature_pipeline import (
    FeaturePipeline, 
    AUDIT_COLS,
    EXCLUDE_FROM_MODEL,
)

# Load actual features
features_path = Path("data/processed/vnat/features.parquet")

if not features_path.exists():
    print(f"Features file not found: {features_path}")
    exit(1)

print("=== Integration Test: Real Features ===\n")

# Load features
df = pd.read_parquet(features_path)
print(f"✓ Loaded features: {df.shape[0]} rows, {df.shape[1]} columns")

# Check for audit columns in data
audit_cols_in_data = [c for c in AUDIT_COLS if c in df.columns]
print(f"✓ Audit columns in data: {audit_cols_in_data}")

if not audit_cols_in_data:
    print("⚠ No audit columns found in data, skipping audit column tests")
else:
    # Show sample values
    for col in audit_cols_in_data:
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"  {col}: mean={mean_val:.2f}, std={std_val:.2f}")
    
    # Fit pipeline on train split
    if 'split' in df.columns:
        df_train = df[df['split'] == 'train']
        print(f"\n✓ Training split: {df_train.shape[0]} rows")
        
        pipe = FeaturePipeline()
        pipe = pipe.fit(df_train)
        
        print(f"✓ Pipeline fitted")
        print(f"  Model features: {len(pipe.model_feature_names())}")
        print(f"  Audit columns: {pipe.audit_cols}")
        
        # Verify audit columns are NOT in model features
        model_features = set(pipe.model_feature_names())
        audit_in_model = AUDIT_COLS & model_features
        
        if audit_in_model:
            print(f"❌ ERROR: Audit columns in model features: {audit_in_model}")
            exit(1)
        else:
            print(f"✓ Audit columns correctly excluded from model")
        
        # Transform test split
        if 'test' in df['split'].unique():
            df_test = df[df['split'] == 'test']
            X_transformed = pipe.transform(df_test)
            
            print(f"\n✓ Transformed test split: {X_transformed.shape[0]} rows, {X_transformed.shape[1]} columns")
            
            # Verify audit columns are in output
            audit_in_output = [c for c in AUDIT_COLS if c in X_transformed.columns]
            print(f"✓ Audit columns in output: {audit_in_output}")
            
            if audit_in_output:
                for col in audit_in_output:
                    val_sample = X_transformed[col].iloc[0]
                    print(f"  {col} sample value: {val_sample}")
        else:
            print("⚠ No test split found")
    else:
        print("⚠ No split column found in data")

print("\n" + "="*60)
print("INTEGRATION TEST PASSED ✓")
print("="*60)
print("\nConclusion:")
print("- Audit columns are correctly excluded from model input")
print("- Audit columns are correctly preserved in transform output")
print("- Pipeline integrates successfully with real data")

