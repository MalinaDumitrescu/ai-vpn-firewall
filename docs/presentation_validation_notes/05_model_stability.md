# Model Stability Validation

## 1. Presentation purpose

This validation category ensures that the clean pipeline's reported results are robust to random seed, threshold selection, and session composition. Model stability validation is essential to prove that high performance is not an artifact of a lucky seed, unstable threshold, or small-sample randomness. By systematically checking metric variability across seeds, threshold selection protocol, and bootstrap confidence intervals, these validations support the scientific credibility and reproducibility of the thesis.

## 2. Slide placement

- Suggested slide title: Model Stability Validation
- Suggested timing: 1–1.5 minutes
- Previous slide connection: After preprocessing & scaling validation
- Next slide connection: Cross-dataset/domain shift validation

## 3. Tests implemented

| Test name | File path | Purpose | CI-ready? | Status |
|---|---|---|---|---|
| test_model_metric_stability_across_seeds | tests/test_model_stability_clean_pipeline.py | Checks metric variability (AUC, recall, FPR) across seeds | Slow | NOT RUN |
| test_validation_selected_threshold_stability | tests/test_model_stability_clean_pipeline.py | Verifies threshold is always selected from validation, not test | Yes | NOT RUN |
| test_session_level_bootstrap_confidence_intervals | tests/test_model_stability_clean_pipeline.py | Computes session-level bootstrap CIs for metrics | Yes | NOT RUN |
| test_model_performance_does_not_regress_below_baseline | tests/test_model_stability_clean_pipeline.py | Prevents regression below baseline metrics | Yes | NOT RUN |

## 4. What the tests prove

- Model performance metrics (AUC, recall, FPR) are stable across random seeds.
- Thresholds are always selected from validation data, never test.
- Session-level bootstrap confidence intervals are finite and auditable.
- The clean pipeline does not silently regress below a documented baseline.

## 5. What the tests do NOT prove

- These tests do not guarantee generalization to new datasets or domains.
- They do not check for model calibration or interpretability.
- They do not validate legacy or out-of-scope pipelines.
- They do not guarantee that all sources of instability are eliminated, only that key metrics are stable within project-defined tolerances.

## 6. Figures / graphics for PowerPoint

| Figure | Path | Type | Slide use | Status |
|---|---|---|---|---|
| Metric variability across seeds | figures/validation/metric_variability_across_seeds.png | Boxplot/line plot | Show AUC, recall, FPR, threshold variability | GENERATED |
| Threshold distribution across seeds | figures/validation/threshold_distribution_across_seeds.png | Histogram | Show threshold selection stability | GENERATED |
| Bootstrap confidence intervals | figures/validation/bootstrap_confidence_intervals.png | Error bar chart | Show session-level metric uncertainty | GENERATED |

### Figure interpretation

- The metric variability figure shows that AUC, recall, and FPR remain stable across seeds, supporting the claim that results are not due to a lucky random seed.
- The threshold distribution plot demonstrates that the decision threshold is always selected from validation and does not vary excessively across seeds.
- The bootstrap confidence interval chart quantifies the uncertainty in AUC, recall, and FPR, using session-level resampling to avoid over-optimistic estimates.

## 7. Tables for PowerPoint

No presentation table generated yet.

## 8. Code slices for PowerPoint

File: tests/test_model_stability_clean_pipeline.py

```python
def test_model_metric_stability_across_seeds(seed_metrics):
    std_auc = seed_metrics['test_auc'].std()
    assert std_auc <= 0.01, f"Test AUC std too high: {std_auc}"
```

File: tests/test_model_stability_clean_pipeline.py

```python
def test_validation_selected_threshold_stability(seed_metrics):
    assert (seed_metrics['threshold_source'] == 'val').all()
    std_thr = seed_metrics['threshold'].std()
    assert std_thr <= 0.05
```

File: notebooks/validation_model_stability_audit.ipynb

```python
plt.errorbar(metrics, means, yerr=[np.array(means)-np.array(ci_lowers), np.array(ci_uppers)-np.array(means)], fmt='o', capsize=5)
plt.title('Bootstrap Confidence Intervals')
plt.savefig('figures/validation/bootstrap_confidence_intervals.png')
```

