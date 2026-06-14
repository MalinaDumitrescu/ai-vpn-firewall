import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path

SEEDS = [42, 123, 456, 789, 2024]

@pytest.fixture(scope="module")
def seed_metrics():
    path = Path('artifacts/validation/model_seed_stability.csv')
    if not path.exists():
        pytest.skip("model_seed_stability.csv not found. Run the seed stability experiment or audit notebook.")
    return pd.read_csv(path)

@pytest.fixture(scope="module")
def baseline_metrics():
    path = Path('artifacts/validation/baseline_metrics.json')
    if not path.exists():
        pytest.skip("baseline_metrics.json not found. Create from known-good results.")
    with open(path) as f:
        return json.load(f)

# 1. Model metric stability across seeds
@pytest.mark.slow
def test_model_metric_stability_across_seeds(seed_metrics):
    std_auc = seed_metrics['test_auc'].std()
    assert std_auc <= 0.01, f"Test AUC std too high: {std_auc}"
    # Optionally: warn if recall/FPR are unstable
    if seed_metrics['test_recall'].std() > 0.05:
        pytest.xfail("Test recall varies >0.05 across seeds (threshold sensitive)")
    if seed_metrics['test_fpr'].std() > 0.05:
        pytest.xfail("Test FPR varies >0.05 across seeds (threshold sensitive)")

# 2. Validation-selected threshold stability
def test_validation_selected_threshold_stability(seed_metrics):
    assert (seed_metrics['threshold_source'] == 'val').all(), "Threshold must be selected from validation only."
    std_thr = seed_metrics['threshold'].std()
    assert std_thr <= 0.05, f"Threshold std too high: {std_thr}"

# 3. Session-level bootstrap confidence intervals
def test_session_level_bootstrap_confidence_intervals():
    path = Path('artifacts/validation/model_stability_summary.csv')
    if not path.exists():
        pytest.skip("model_stability_summary.csv not found. Run the audit notebook.")
    df = pd.read_csv(path)
    for col in ['auc_ci_lower', 'auc_ci_upper', 'recall_ci_lower', 'recall_ci_upper', 'fpr_ci_lower', 'fpr_ci_upper']:
        assert np.isfinite(df[col]).all(), f"Non-finite CI for {col}"

# 4. Model performance does not regress below baseline
def test_model_performance_does_not_regress_below_baseline(seed_metrics, baseline_metrics):
    # Use mean over seeds for current
    cur = seed_metrics.mean(numeric_only=True)
    assert cur['test_auc'] >= baseline_metrics['test_auc'] - 0.02, f"Test AUC regressed: {cur['test_auc']} < {baseline_metrics['test_auc']}"
    assert cur['test_recall'] >= baseline_metrics['test_recall'] - 0.05, f"Recall regressed: {cur['test_recall']} < {baseline_metrics['test_recall']}"
    assert cur['test_fpr'] <= baseline_metrics['test_fpr'] + 0.02, f"FPR increased: {cur['test_fpr']} > {baseline_metrics['test_fpr']}"

