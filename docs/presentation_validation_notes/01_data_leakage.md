# Data Leakage Validation

## 1. Presentation purpose

This validation category ensures that the clean VPN detection pipeline is protected against hidden information leakage between train, validation, and test splits. In encrypted traffic classification, even a small amount of leakage—such as flows from the same capture appearing in multiple splits, or preprocessing fitted on test or full data—can artificially inflate model performance. These tests guarantee that the evaluation is not contaminated by such errors, supporting the credibility of the thesis results. The suite also checks for row-level feature duplication across splits, threshold provenance, and metadata leakage in model features.

## 2. Slide placement

- Suggested slide title: Data Leakage Validation
- Suggested timing: 1–1.5 minutes
- Previous slide connection: After the clean pipeline overview
- Next slide connection: Split integrity validation

## 3. Tests implemented

| Test name | File path | Purpose | CI-ready? | Status |
|---|---|---|---|---|
| test_no_capture_overlap_between_splits | tests/test_data_leakage_clean_pipeline.py | Asserts no capture_id appears in more than one split; checks manifest and features.parquet agree | Yes | PASS |
| test_no_exact_duplicate_feature_rows_across_splits | tests/test_data_leakage_clean_pipeline.py | Detects row-level feature duplication across splits (row hash) | Yes | FAIL (9 hashes, 21 rows) |
| test_thresholds_are_validation_derived_only | tests/test_data_leakage_clean_pipeline.py | Verifies all deployment thresholds are derived from validation split only | Yes | PASS |
| test_scaler_is_fit_only_on_training_data | tests/test_data_leakage_clean_pipeline.py | Checks that any preprocessing/scaler is fit only on train split | Yes | PASS |
| test_no_metadata_columns_in_model_features | tests/test_data_leakage_clean_pipeline.py | Ensures no metadata/forbidden columns in model features | Yes | PASS |

## 4. What the tests prove

- No capture appears in more than one split (capture-level integrity).
- No exact feature row appears in more than one split (row-level leakage).
- All deployment thresholds are derived from validation data only.
- Preprocessing/scaling is fit only on the training split.
- Model input features are free of metadata columns that could leak information.

## 5. What the tests do NOT prove

- These tests do not prove cross-dataset generalization or domain robustness.
- They do not guarantee that all forms of subtle leakage (e.g., semantic or timing-based) are absent.
- They do not check for false positives on realistic benign traffic scenarios.
- They only verify artifacts produced by the current clean pipeline (not legacy pipeline).

## 6. Figures / graphics for PowerPoint

| Figure | Path | Type | Slide use | Status |
|---|---|---|---|---|
| Capture overlap matrix | figures/validation/capture_overlap_matrix.png | Heatmap | Visual proof of split disjointness | GENERATED |

### Figure interpretation

The capture overlap matrix shows the number of unique captures in train, validation, and test splits. All off-diagonal values are zero, confirming that no capture appears in more than one split. This supports the claim that the evaluation is free from capture-level leakage.

## 7. Tables for PowerPoint

| Check | Status | Details |
|---|---|---|
| No capture overlap | PASS | All pairwise intersections empty; manifests and features.parquet agree |
| No duplicate feature rows across splits | FAIL | 9 hashes, 21 rows span >1 split (short, low-information flows) |
| Thresholds from validation only | PASS | All policies: fit_split = val |
| Scaler fit only on train | PASS | fit_split = train, stats match train only |
| No metadata columns in features | PASS | 0 forbidden substrings, 0 metadata columns |

## 8. Code slices for PowerPoint

File: tests/test_data_leakage_clean_pipeline.py

```python
def test_no_capture_overlap_between_splits(paths, split_captures, features_df, summary_sink):
    train = split_captures["train"]
    val = split_captures["val"]
    test = split_captures["test"]
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
```

File: tests/test_data_leakage_clean_pipeline.py

```python
def test_no_exact_duplicate_feature_rows_across_splits(features_df, summary_sink):
    feat_cols = model_feature_columns(features_df)
    row_hash = hash_feature_rows(features_df, feat_cols)
    work = features_df[["split"]].copy()
    work["row_hash"] = row_hash.values
    splits_per_hash = work.groupby("row_hash")["split"].nunique()
    cross_hashes = splits_per_hash[splits_per_hash > 1].index
    assert len(cross_hashes) == 0
```

File: scripts/run_data_leakage_audit.py

```python
mat = capture_overlap_matrix(splits)
fig, ax = plt.subplots(...)
plt.savefig(out_png, bbox_inches="tight")
```

