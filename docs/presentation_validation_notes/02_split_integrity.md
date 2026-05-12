# Split Integrity Validation

## 1. Presentation purpose

This validation category ensures that the data splits used for training, validation, and testing in the clean VPN detection pipeline are robust and free from structural flaws. Split integrity is critical because improper splits can lead to over-optimistic results, such as information leakage or class imbalance, which undermine the credibility of the evaluation. By systematically checking split assignments, class balance, capture dominance, and split stability, we guarantee that the model's reported performance reflects real-world deployment conditions. These validations are essential for reproducibility and for defending the scientific rigor of the thesis.

## 2. Slide placement

- Suggested slide title: Split Integrity Validation
- Suggested timing: 1.5–2 minutes
- Previous slide connection: After data leakage validation
- Next slide connection: Feature consistency validation

## 3. Tests implemented

| Test name | File path | Purpose | CI-ready? | Status |
|---|---|---|---|---|
| test_every_flow_assigned_to_exactly_one_split | tests/test_split_integrity_clean_pipeline.py | Checks that every flow is assigned to exactly one split and no duplicates exist | Yes | NOT RUN |
| test_each_dataset_split_contains_both_classes | tests/test_split_integrity_clean_pipeline.py | Ensures both VPN and nonVPN classes are present in each dataset/split | Yes | NOT RUN |
| test_no_single_capture_dominates_split | tests/test_split_integrity_clean_pipeline.py | Verifies that no single capture dominates any split beyond the allowed threshold | Yes | NOT RUN |
| test_splitter_stability_across_random_seeds | tests/test_split_integrity_clean_pipeline.py | Checks split stability and constraint adherence across random seeds | Yes (slow) | NOT RUN |
| test_fail_on_constraint_violation | tests/test_split_integrity_clean_pipeline.py | Ensures pipeline fails if constraint violations are present and config is set | Yes | NOT RUN |

## 4. What the tests prove

- Every flow is assigned to exactly one split (train, val, or test).
- Each dataset/split contains both VPN and nonVPN classes (unless documented).
- No single capture dominates any split beyond the configured threshold.
- The splitting process is stable and constraint-respecting across random seeds.
- The pipeline enforces failure if split constraints are violated (when configured).

## 5. What the tests do NOT prove

- These tests do not guarantee cross-dataset generalization or domain adaptation.
- They do not check for feature leakage or preprocessing contamination.
- They do not validate model performance or calibration.
- They only verify the integrity of splits in the clean pipeline artifacts.

## 6. Figures / graphics for PowerPoint

| Figure | Path | Type | Slide use | Status |
|---|---|---|---|---|
| Split composition per dataset | figures/validation/split_composition_per_dataset.png | Stacked bar chart | Show class balance and split composition | GENERATED |
| Capture size distribution | figures/validation/capture_size_distribution.png | Boxplot (log scale) | Illustrate capture size variability across splits | GENERATED |
| Split constraint violations by seed | figures/validation/split_constraint_violations_by_seed.png | Bar chart | Demonstrate split stability and constraint adherence | GENERATED |

### Figure interpretation

- The split composition bar chart shows the number of VPN and nonVPN flows in each split for every dataset, confirming class balance and split assignments.
- The capture size distribution boxplot visualizes the distribution of flows per capture (log scale) across splits and datasets, highlighting whether any capture dominates a split.
- The constraint violations by seed bar chart displays the number of split constraint violations for each random seed, supporting claims of splitter stability and robustness.

## 7. Tables for PowerPoint

No presentation table generated yet.

## 8. Code slices for PowerPoint

File: tests/test_split_integrity_clean_pipeline.py

```python
def test_every_flow_assigned_to_exactly_one_split(features_df):
    assert features_df['split'].notnull().all()
    assert set(features_df['split'].unique()) <= {'train', 'val', 'test'}
    if 'flow_id' in features_df.columns:
        dup = features_df.groupby('flow_id')['split'].nunique()
        assert (dup <= 1).all()
```

File: tests/test_split_integrity_clean_pipeline.py

```python
def test_no_single_capture_dominates_split(features_df):
    for ds in features_df['dataset'].unique():
        for split in ['train', 'val', 'test']:
            sub = features_df[(features_df['dataset'] == ds) & (features_df['split'] == split)]
            if len(sub) == 0:
                continue
            cap_counts = sub['capture_id'].value_counts()
            share = cap_counts.max() / len(sub)
            assert share <= 0.40
```

File: notebooks/validation_split_integrity_audit.ipynb

```python
split_counts = features.groupby(['dataset', 'split', 'label']).size().unstack(fill_value=0)
split_counts[['nonVPN', 'VPN']].plot(kind='bar', stacked=True)
plt.savefig('figures/validation/split_composition_per_dataset.png')
```

