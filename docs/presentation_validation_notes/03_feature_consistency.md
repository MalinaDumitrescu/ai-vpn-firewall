# Feature Consistency Validation

## 1. Presentation purpose

This validation category ensures that the features used by the clean VPN detection pipeline are consistent, direction-invariant, and formula-correct. Feature consistency is essential to prevent hidden leakage, guarantee reproducibility, and ensure that the model receives only safe, well-defined descriptors. By locking the feature schema, verifying direction-invariance, and checking formula correctness, these validations protect against accidental inclusion of risky or misaligned features and support the scientific rigor of the thesis.

## 2. Slide placement

- Suggested slide title: Feature Consistency Validation
- Suggested timing: 1–1.5 minutes
- Previous slide connection: After split integrity validation
- Next slide connection: Preprocessing/scaling validation

## 3. Tests implemented

| Test name | File path | Purpose | CI-ready? | Status |
|---|---|---|---|---|
| test_raw_flow_schema_contains_required_columns | tests/test_feature_consistency_clean_pipeline.py | Checks that all required raw flow columns are present before feature extraction | Yes | NOT RUN |
| test_extracted_features_match_feature_columns_json | tests/test_feature_consistency_clean_pipeline.py | Ensures extracted features exactly match the locked model schema (order, names) | Yes | NOT RUN |
| test_final_feature_family_is_direction_invariant | tests/test_feature_consistency_clean_pipeline.py | Verifies that the final model feature set is direction-invariant and free of risky descriptors | Yes | NOT RUN |
| test_rate_features_recomputed_match_stored_values | tests/test_feature_consistency_clean_pipeline.py | Checks that stored rate features match recomputed values from base columns | Yes | NOT RUN |
| test_no_nan_inf_in_final_feature_matrix | tests/test_feature_consistency_clean_pipeline.py | Ensures no NaN or Inf values in the final model feature matrix | Yes | NOT RUN |

## 4. What the tests prove

- The raw flow schema is sufficient and enforced for feature extraction.
- The final model feature columns are locked, ordered, and match the exported schema.
- No direction-dependent or risky features are present in the final model input.
- Rate-derived features are formula-consistent and numerically correct.
- The model never receives NaN or infinite feature values.

## 5. What the tests do NOT prove

- These tests do not check for cross-dataset generalization or domain adaptation.
- They do not validate model performance or calibration.
- They do not guarantee that all features are optimal, only that they are safe and consistent.
- They only verify clean pipeline artifacts, not legacy or out-of-scope pipelines.

## 6. Figures / graphics for PowerPoint

| Figure | Path | Type | Slide use | Status |
|---|---|---|---|---|
| Rate feature formula consistency | figures/validation/rate_feature_formula_consistency.png | Scatter plot (2 panels, log scale) | Show formula correctness for packet/byte rate | GENERATED |

### Figure interpretation

The rate feature formula consistency figure shows scatter plots of stored versus recomputed packet and byte rates for all flows. Points should lie on the diagonal, confirming that the stored values are mathematically correct. This supports the claim that the feature extraction logic is formula-consistent and free from silent errors.

## 7. Tables for PowerPoint

No presentation table generated yet.

## 8. Code slices for PowerPoint

File: tests/test_feature_consistency_clean_pipeline.py

```python
def test_extracted_features_match_feature_columns_json(features_df, feature_columns):
    model_feats = [c for c in feature_columns]
    df_feats = [c for c in features_df.columns if c in model_feats]
    assert df_feats == model_feats, f"Feature columns mismatch or order changed."
```

File: tests/test_feature_consistency_clean_pipeline.py

```python
def test_final_feature_family_is_direction_invariant(feature_family):
    risky_patterns = ['fwd', 'bwd', 'src', 'dst', 'client', 'server']
    family = get_family(feature_family)
    risky = [f for f in family if any(p in f.lower() for p in risky_patterns)]
    assert not risky, f"Direction-dependent risky features found: {risky}"
```

File: notebooks/validation_feature_consistency_audit.ipynb

```python
df = features[features['flow_duration'] > 0].copy()
recomputed_packet_rate = df['total_packets'] / df['flow_duration']
plt.scatter(df['packet_rate'], recomputed_packet_rate, alpha=0.3, s=5)
plt.plot([min, max], [min, max], 'r--')
plt.xscale('log'); plt.yscale('log')
plt.savefig('figures/validation/rate_feature_formula_consistency.png')
```

