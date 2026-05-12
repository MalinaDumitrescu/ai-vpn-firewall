# Preprocessing & Scaling Validation

## 1. Presentation purpose

This validation category ensures that all preprocessing and scaling steps in the clean VPN detection pipeline are leakage-safe and reproducible. It proves that all fitted transformers (such as quantile scaling) are trained exclusively on the training split and then applied unchanged to validation and test data. This prevents information from leaking from validation or test sets into the model, which could otherwise lead to over-optimistic results. The tests and audit artifacts guarantee that the main protocol never uses dataset-wise normalization and that the feature order and fit statistics are locked and auditable.

## 2. Slide placement

- Suggested slide title: Preprocessing & Scaling Validation
- Suggested timing: 1–1.5 minutes
- Previous slide connection: After feature consistency validation
- Next slide connection: Model stability validation

## 3. Tests implemented

| Test name | File path | Purpose | CI-ready? | Status |
|---|---|---|---|---|
| test_preprocessing_transformers_fit_on_train_only | tests/test_preprocessing_scaling_clean_pipeline.py | Verifies all transformers are fit on train only and applied to val/test | Yes | NOT RUN |
| test_no_datasetwise_normalization_in_main_pipeline | tests/test_preprocessing_scaling_clean_pipeline.py | Ensures dataset-wise normalization is not used in the main protocol | Yes | NOT RUN |
| test_transformer_feature_order_matches_model_schema | tests/test_preprocessing_scaling_clean_pipeline.py | Checks transformer feature order matches model schema | Yes | NOT RUN |
| test_val_test_scaling_does_not_change_fit_statistics | tests/test_preprocessing_scaling_clean_pipeline.py | Ensures val/test scaling does not update fit statistics | Yes | NOT RUN |
| test_monotonic_scaling_does_not_change_single_feature_auc_direction | tests/test_preprocessing_scaling_clean_pipeline.py | Checks monotonic scaling does not flip feature AUC direction | Yes | NOT RUN |

## 4. What the tests prove

- All preprocessing/scaling transformers are fit on train only and never refit on val/test.
- No dataset-wise normalization is used in the main evaluation protocol.
- The feature order for scaling matches the model schema exactly.
- Transforming val/test does not alter the fitted transformer statistics.
- Monotonic scaling preserves the direction of feature discrimination.

## 5. What the tests do NOT prove

- These tests do not check for model performance or calibration.
- They do not guarantee optimal scaling, only leakage safety and reproducibility.
- They do not validate legacy or out-of-scope pipelines.
- They do not prove cross-dataset generalization.

## 6. Figures / graphics for PowerPoint

| Figure | Path | Type | Slide use | Status |
|---|---|---|---|---|
| Preprocessing fit/transform protocol | figures/validation/preprocessing_fit_transform_protocol.png | Pipeline diagram | Show train-only fitting and transform protocol | GENERATED |
| Scaled feature distribution examples | figures/validation/scaled_feature_distribution_examples.png | KDE plots | Show scaled feature distributions for train/val/test | GENERATED |

### Figure interpretation

- The protocol diagram visually confirms that the transformer is fit on train only and then applied to all splits, supporting the claim of leakage-free preprocessing.
- The scaled feature distribution plots show that train, val, and test are all scaled using the same statistics, with no evidence of dataset-wise normalization.

## 7. Tables for PowerPoint

No presentation table generated yet.

## 8. Code slices for PowerPoint

File: tests/test_preprocessing_scaling_clean_pipeline.py

```python
def test_preprocessing_transformers_fit_on_train_only(preprocessing_meta, features_df, feature_columns):
    assert preprocessing_meta['fit_split'] == 'train'
    assert set(preprocessing_meta['transformed_splits']) == {'train', 'val', 'test'}
    assert preprocessing_meta['feature_list'] == feature_columns
```

File: tests/test_preprocessing_scaling_clean_pipeline.py

```python
def test_no_datasetwise_normalization_in_main_pipeline(preprocessing_meta):
    assert not preprocessing_meta.get('datasetwise_normalization', False)
```

File: notebooks/validation_preprocessing_scaling_audit.ipynb

```python
fig, ax = plt.subplots(figsize=(7, 2))
ax.axis('off')
ax.text(0.1, 0.5, 'Train', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightblue'))
ax.text(0.4, 0.5, 'Fit transformer', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax.text(0.7, 0.5, 'Transform train', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat'))
ax.text(0.7, 0.2, 'Transform val', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat'))
ax.text(0.7, 0.8, 'Transform test', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat'))
plt.savefig('figures/validation/preprocessing_fit_transform_protocol.png')
```

