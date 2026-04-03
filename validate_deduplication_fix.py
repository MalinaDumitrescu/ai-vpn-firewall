#!/usr/bin/env python
"""
Cross-Split Deduplication Validation Test

This script validates that the duplicate removal fix works correctly
and eliminates cross-split leakage while preserving data integrity.
"""

import pandas as pd
from src.pipeline.data_preparation import load_and_prepare_data
from src.pipeline.feature_pipeline import FeaturePipeline
from src.utils.logging import setup_logger

logger = setup_logger()

def main():
    print("=" * 80)
    print("CROSS-SPLIT DEDUPLICATION VALIDATION TEST")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading data...")
    df_all = load_and_prepare_data()
    df_train = df_all[df_all['split'] == 'train'].copy()
    df_test = df_all[df_all['split'] == 'test'].copy()
    print(f"✓ Train: {len(df_train)} flows, Test: {len(df_test)} flows")
    
    # Fit pipeline
    print("\n[2/5] Fitting feature pipeline...")
    pipe = FeaturePipeline().fit(df_train)
    model_features = pipe.model_feature_names()
    print(f"✓ Model features: {len(model_features)}")
    
    # Transform data
    print("\n[3/5] Transforming data...")
    X_train_t = pipe.transform(df_train)
    X_test_t = pipe.transform(df_test)
    print(f"✓ Transformed train: {X_train_t.shape}, test: {X_test_t.shape}")
    
    # Check for duplicates BEFORE deduplication
    print("\n[4/5] Detecting cross-split duplicates (BEFORE fix)...")
    train_tuples = set(map(tuple, X_train_t[model_features].values))
    test_values = X_test_t[model_features].values
    test_duplicates_before = [tuple(row) in train_tuples for row in test_values]
    duplicates_before = sum(test_duplicates_before)
    leak_pct_before = (duplicates_before / len(X_test_t)) * 100
    
    print(f"  Identical flows: {duplicates_before}")
    print(f"  Leakage percentage: {leak_pct_before:.2f}%")
    
    if duplicates_before > 0:
        print("  ⚠ Cross-split duplicates detected!")
        
        # Apply fix
        print("\n[5/5] Applying cross-split deduplication fix...")
        df_all_combined = pd.concat([X_train_t, X_test_t], ignore_index=False)
        df_deduped, removed = FeaturePipeline.remove_cross_split_duplicates_in_transformed_space(
            df_all_combined,
            model_features
        )
        
        # Verify fix
        X_test_deduped = df_deduped[df_deduped['split'] == 'test']
        X_train_deduped = df_deduped[df_deduped['split'] == 'train']
        
        train_tuples_after = set(map(tuple, X_train_deduped[model_features].values))
        test_values_after = X_test_deduped[model_features].values
        test_duplicates_after = [tuple(row) in train_tuples_after for row in test_values_after]
        duplicates_after = sum(test_duplicates_after)
        leak_pct_after = (duplicates_after / len(X_test_deduped)) * 100 if len(X_test_deduped) > 0 else 0
        
        print(f"  Removed: {removed} flows")
        print(f"  New test set size: {len(X_test_deduped)}")
        print(f"  Remaining duplicates: {duplicates_after}")
        print(f"  Leakage percentage after: {leak_pct_after:.2f}%")
        
        print("\n" + "=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        print(f"Before: {duplicates_before} duplicates ({leak_pct_before:.2f}% leakage)")
        print(f"After:  {duplicates_after} duplicates ({leak_pct_after:.2f}% leakage)")
        print(f"Data loss: {removed} flows ({(removed / len(df_test)) * 100:.2f}%)")
        
        if duplicates_after == 0:
            print("\n✓ SUCCESS: Cross-split deduplication eliminated all duplicates!")
        else:
            print(f"\n⚠ WARNING: {duplicates_after} duplicates remain")
    else:
        print("  ✓ No duplicates found!")
    
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

