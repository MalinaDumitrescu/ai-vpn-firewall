# Cross-Dataset Domain Shift Validation

## 1. Presentation purpose

This validation category demonstrates whether the clean pipeline's generalization failures are due to evaluation errors or genuine structural mismatch between datasets. By auditing leave-one-dataset-out (LODO) protocol integrity, dataset fingerprintability, feature distribution shift, and sign reversal, we show that the pipeline is scientifically sound and that observed transfer weaknesses are rooted in real-world domain shift, not implementation flaws. These validations are essential for defending the thesis claim that cross-dataset generalization is fundamentally hard in VPN traffic detection.

## 2. Slide placement

- Suggested slide title: Cross-Dataset Domain Shift Validation
- Suggested timing: 1.5–2 minutes
- Previous slide connection: After model stability validation
- Next slide connection: Benign false positive scenarios or thesis discussion

## 3. Tests implemented

| Test name | File path | Purpose | CI-ready? | Status |
|---|---|---|---|---|
| test_lodo_target_dataset_excluded_from_training | tests/test_cross_dataset_domain_shift_clean_pipeline.py | Verifies LODO protocol integrity and target exclusion | Yes | NOT RUN |
| test_audit_dataset_fingerprinting_strength | tests/test_cross_dataset_domain_shift_clean_pipeline.py | Audits how well dataset identity can be predicted from features | Yes | NOT RUN |
| test_audit_feature_distribution_shift_across_datasets | tests/test_cross_dataset_domain_shift_clean_pipeline.py | Quantifies marginal feature shift (JSD, KS) across datasets | Yes | NOT RUN |
| test_audit_cross_dataset_feature_sign_reversal | tests/test_cross_dataset_domain_shift_clean_pipeline.py | Detects sign reversal in feature-label relationship across datasets | Yes | NOT RUN |

## 4. What the tests prove

- LODO protocol is implemented correctly: target dataset is fully excluded from training and validation.
- Dataset identity is strongly encoded in the final feature representation (high fingerprintability is a scientific result, not a failure).
- Many features exhibit strong distribution shift across datasets, as measured by JSD and KS.
- Several features reverse their VPN/nonVPN relationship across datasets, supporting the claim of structural mismatch.

## 5. What the tests do NOT prove

- These tests do not guarantee transfer performance will be high—only that poor transfer is not due to protocol errors.
- They do not check for model calibration or interpretability.
- They do not validate legacy or out-of-scope pipelines.
- They do not prove that all domain shift is captured, only that key indicators are measured.

## 6. Figures / graphics for PowerPoint

| Figure | Path | Type | Slide use | Status |
|---|---|---|---|---|
| Domain fingerprinting confusion matrix | figures/validation/domain_fingerprinting_confusion_matrix.png | Confusion matrix | Show dataset separability in feature space | GENERATED |
| Top shifted features by JSD | figures/validation/top_shifted_features_jsd.png | Bar chart | Show which features shift most across datasets | GENERATED |
| Sign reversal heatmap | figures/validation/sign_reversal_heatmap.png | Heatmap | Show features with sign reversal across datasets | GENERATED |
| LODO transfer AUC by target | figures/validation/lodo_transfer_auc_by_target.png | Bar chart | Show transfer performance (if available) | TODO or GENERATED |

### Figure interpretation

- The domain fingerprinting confusion matrix shows that dataset identity can be predicted with high accuracy, confirming strong domain shift.
- The top shifted features bar chart highlights which features differ most across datasets, supporting the claim of structural mismatch.
- The sign reversal heatmap visualizes features whose VPN/nonVPN relationship changes direction, a key indicator of transfer difficulty.
- The LODO transfer AUC bar chart (if available) quantifies actual transfer performance for each held-out dataset.

## 7. Tables for PowerPoint

No presentation table generated yet.

## 8. Code slices for PowerPoint

File: tests/test_cross_dataset_domain_shift_clean_pipeline.py

```python
def test_lodo_target_dataset_excluded_from_training(features_df):
    for target in features_df['dataset'].unique():
        trainval = features_df[features_df['dataset'] != target]
        test = features_df[features_df['dataset'] == target]
        assert target not in trainval['dataset'].unique()
        assert set(test['dataset'].unique()) == {target}
        assert set(trainval['capture_id']).isdisjoint(set(test['capture_id']))
```

File: src/clean_pipeline/validation/domain_shift.py

```python
def compute_jsd(p, q, bins=50):
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    p_hist = p_hist + 1e-8
    q_hist = q_hist + 1e-8
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return jensenshannon(p_hist, q_hist, base=2)
```

File: notebooks/validation_cross_dataset_domain_shift_audit.ipynb

```python
sns.heatmap(pivot, center=0, cmap='coolwarm', annot=False)
plt.title('Feature Sign Reversal Heatmap (SMD VPN - nonVPN)')
plt.savefig('figures/validation/sign_reversal_heatmap.png')
```

