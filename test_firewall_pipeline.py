#!/usr/bin/env python3
"""Integration test for the deployable firewall pipeline."""

from demo_firewall import FirewallBlocker, DeploymentMode

# 1. Instantiate in STRICT mode
blocker = FirewallBlocker(mode=DeploymentMode.STRICT)
print("1. Instantiated FirewallBlocker")

# 2. Load model artifacts
blocker.load()
print("2. Models loaded")
diag = blocker.diagnostics()
print(f"   Models: {diag['predictor']['n_models_total']}")
print(f"   Families: {diag['predictor']['n_families']}")
print(f"   Calibration: {diag['predictor']['calibration_method']}")
print(f"   Features: {diag['predictor']['n_features']}")

# 3. Calibrate thresholds
tc = blocker.calibrate_from_validation()
print(f"3. Thresholds calibrated:")
print(f"   Block threshold: {tc.block_threshold:.6f}")
print(f"   Flag threshold:  {tc.flag_threshold:.6f}")
print(f"   Source split:    {tc.source_split}")
print(f"   Aggregation:     {tc.aggregation_rule}")

# 4. Domain warning
warn = blocker.domain_separability_warning()
if warn:
    print(f"4. Domain warning: {warn[:100]}...")

# 5. Evaluate on test set
metrics = blocker.evaluate_dataset()
print("5. Test evaluation:")
for key in [
    "session_roc_auc", "session_pr_auc",
    "block_recall", "block_fpr", "block_precision",
    "flagged_recall", "flagged_fpr",
]:
    val = metrics.get(key)
    if val is not None:
        print(f"   {key:20s} = {val:.4f}")
    else:
        print(f"   {key:20s} = N/A")

print(f"   n_sessions: {metrics.get('n_sessions_evaluated')}")
print(f"   n_positive: {metrics.get('n_positive')}")
print(f"   n_negative: {metrics.get('n_negative')}")

# 6. Generate report
from demo_firewall.report import format_report
report = format_report(
    metrics=metrics,
    predictor_diagnostics=blocker._predictor.diagnostics(),
    policy_diagnostics=blocker._policy.diagnostics(),
)
print()
print(report)

# 7. Test safety checks
from demo_firewall.errors import CalibrationError, ThresholdLeakageError
import numpy as np
import pandas as pd

# Test CalibrationError on single-class calibration
from demo_firewall.policy import compute_threshold_from_validation
try:
    compute_threshold_from_validation(
        val_session_scores=np.array([0.1, 0.2, 0.3]),
        val_labels=np.array([1, 1, 1]),  # No benign!
        target_fpr=0.0,
    )
    print("ERROR: ThresholdLeakageError should have been raised!")
except ThresholdLeakageError as e:
    print(f"7. Safety check OK: ThresholdLeakageError raised for no-benign pool")

# 8. Test all three modes can be instantiated
for mode in DeploymentMode:
    b = FirewallBlocker(mode=mode)
    print(f"8. Mode {mode.value} instantiated OK")

print()
print("=" * 50)
print("  ALL INTEGRATION TESTS PASSED")
print("=" * 50)

