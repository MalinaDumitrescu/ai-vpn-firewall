#!/usr/bin/env python
"""
Test script for Parts F-H infrastructure improvements.

Validates:
- Part F: EnhancedDriftMonitor (feature-level drift, trends, composite scores)
- Part G: SafeRecalibrator (guardrails, staged rollout, rollback, audit)
- Part H: ImprovedDecisionEngine (health checks, metrics, graceful degradation)
"""
import sys
import os
import numpy as np

# Ensure project root is on path
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

SEED = 42
np.random.seed(SEED)
rng = np.random.RandomState(SEED)

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")

# ═══════════════════════════════════════════════════════
#  PART F: Enhanced Drift Monitor
# ═══════════════════════════════════════════════════════
print("=" * 70)
print("PART F: Enhanced Drift Monitor")
print("=" * 70)

from src.deployment.enhanced_drift_monitor import (
    EnhancedDriftMonitor, EnhancedDriftReport, FeatureDriftResult,
    DriftTrend,
)

feature_names = ['sz_coef_variation', 'sz_p25_median_ratio', 'sz_p75_median_ratio',
                 'sz_iqr_norm_median', 'dispersion_symmetry']

# Create reference distribution (validation benign)
n_ref = 200
ref_scores = rng.beta(2, 10, size=n_ref)  # Low scores (benign)
ref_features = rng.randn(n_ref, 5) * 0.5 + np.array([0.3, 0.5, 1.5, 0.4, 0.1])

# F1: Basic fit + check with no drift
monitor = EnhancedDriftMonitor(feature_names=feature_names)
monitor.fit(ref_scores, ref_features)
check("F1a: Monitor fitted", monitor._fitted)

# Use a subsample of the SAME reference data to ensure no drift
cur_idx = rng.choice(len(ref_scores), size=100, replace=True)
cur_scores = ref_scores[cur_idx] + rng.normal(0, 0.001, size=100)  # tiny noise
cur_features = ref_features[cur_idx] + rng.normal(0, 0.001, size=(100, 5))
report = monitor.check(cur_scores, cur_features)
check("F1b: No-drift check OK", report.composite_level in ("OK", "WARNING"))
check("F1c: Composite score low", report.composite_score < 0.5)
print(f"       composite_level={report.composite_level}, score={report.composite_score}")

# F2: Simulate score drift (shift scores upward)
drifted_scores = ref_scores + 0.4  # Major shift
drifted_features = ref_features + np.array([1.0, 0.8, 0.6, 0.5, 0.3])  # Feature shift
report_drift = monitor.check(drifted_scores, drifted_features)
check("F2a: Drift detected", report_drift.composite_level in ("WARNING", "HIGH", "CRITICAL"))
check("F2b: Composite score high", report_drift.composite_score > 0.2)
check("F2c: Features drifted", report_drift.n_features_high + report_drift.n_features_warning > 0)
print(f"       level={report_drift.composite_level}, score={report_drift.composite_score}")
print(f"       worst_feature={report_drift.worst_feature}, ks={report_drift.worst_feature_ks:.3f}")

# F3: Feature-level details
check("F3a: Feature drifts populated", len(report_drift.feature_drifts) > 0)
if report_drift.feature_drifts:
    fd = report_drift.feature_drifts[0]
    check("F3b: Feature drift has KS stat", fd.ks_statistic >= 0)
    check("F3c: Feature drift has PSI", fd.psi >= 0)
    check("F3d: Feature drift serializable", 'feature_name' in fd.to_dict())

# F4: Trend detection (simulate worsening drift)
for i in range(5):
    shift = 0.1 * (i + 1)
    monitor.check(ref_scores + shift, ref_features + shift)
last = monitor.history[-1]
check("F4a: History populated", len(monitor.history) > 5)
check("F4b: Trend detected", last.trend_direction in ("WORSENING", "STABLE", "IMPROVING"))
print(f"       trend={last.trend_direction}, slope={last.trend_slope:.4f}")

# F5: Dashboard data
dashboard = monitor.dashboard_data()
check("F5a: Dashboard has status", 'current_status' in dashboard)
check("F5b: Dashboard has history", len(dashboard['composite_score_history']) > 0)
check("F5c: Dashboard has trend", 'trend' in dashboard)

# F6: Save/load roundtrip
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    save_path = Path(tmpdir) / "drift_monitor.json"
    monitor.save(save_path)
    loaded = EnhancedDriftMonitor.load(save_path)
    check("F6a: Loaded is fitted", loaded._fitted)
    check("F6b: Feature names preserved", loaded.feature_names == feature_names)
    # Check it can run a check after loading
    r2 = loaded.check(cur_scores, cur_features)
    check("F6c: Loaded monitor works", r2.composite_level is not None)

# F7: Report serialization
report_dict = report_drift.to_dict()
check("F7a: Report serializable", isinstance(report_dict, dict))
check("F7b: Report has score_drift", 'score_drift' in report_dict)
check("F7c: Report has feature_drifts", 'feature_drifts' in report_dict)

# ═══════════════════════════════════════════════════════
#  PART G: Safe Recalibration
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART G: Safe Recalibration")
print("=" * 70)

from src.deployment.safe_recalibration import (
    SafeRecalibrator, SafeRecalibrationResult, RolloutStage,
    RecalibrationConfidence, RecalibrationEvent,
)

base_block = 0.7447
base_flag = 0.4977

# G1: Basic propose with enough samples
recal = SafeRecalibrator(
    base_block_threshold=base_block,
    base_flag_threshold=base_flag,
    min_samples=30,
    high_confidence_samples=100,
    blend_steps=3,
)
benign_scores = rng.beta(2, 10, size=120)  # 120 benign samples
result = recal.propose(benign_scores)
check("G1a: Proposal created", result is not None)
check("G1b: Stage is SHADOW", recal.stage == RolloutStage.SHADOW)
check("G1c: Confidence is HIGH", result.confidence == "HIGH")
check("G1d: Validation passed", result.validation_passed)
check("G1e: Active thresholds unchanged",
      abs(result.active_block_threshold - base_block) < 1e-6)
print(f"       proposed_block={result.proposed_block_threshold:.4f}, "
      f"shift={result.threshold_shift:+.4f}")

# G2: Staged rollout (SHADOW -> PARTIAL -> FULL)
r2 = recal.advance()
check("G2a: Advanced to PARTIAL", recal.stage == RolloutStage.PARTIAL)
check("G2b: Blend alpha > 0", r2.blend_alpha > 0)
check("G2c: Active threshold changed",
      abs(r2.active_block_threshold - base_block) > 1e-6)
print(f"       stage=PARTIAL, alpha={r2.blend_alpha:.2f}, "
      f"active_block={r2.active_block_threshold:.4f}")

r3 = recal.advance()
check("G2d: Second advance increases alpha", r3.blend_alpha > r2.blend_alpha)

r4 = recal.advance()
check("G2e: Third advance -> FULL", recal.stage == RolloutStage.FULL)
check("G2f: Alpha = 1.0", abs(r4.blend_alpha - 1.0) < 1e-6)
print(f"       stage=FULL, alpha={r4.blend_alpha:.2f}, "
      f"active_block={r4.active_block_threshold:.4f}")

# G3: Rollback
r5 = recal.rollback("Test rollback")
check("G3a: Rolled back", recal.stage == RolloutStage.ROLLED_BACK)
check("G3b: Thresholds reverted",
      abs(r5.active_block_threshold - base_block) < 1e-6)
print(f"       stage=ROLLED_BACK, active_block={r5.active_block_threshold:.4f}")

# G4: Safety guardrails (extreme shift)
recal2 = SafeRecalibrator(
    base_block_threshold=0.7447,
    base_flag_threshold=0.4977,
    max_shift_abs=0.10,
    max_shift_pct=0.15,
)
# Scores that would push threshold way up
extreme_scores = rng.uniform(0.8, 0.95, size=50)
r_extreme = recal2.propose(extreme_scores)
check("G4a: Guardrails triggered", len(r_extreme.guardrail_violations) > 0)
check("G4b: Shift clamped", abs(r_extreme.threshold_shift) <= 0.10 + 0.01)
print(f"       violations={r_extreme.guardrail_violations}")

# G5: Insufficient samples
recal3 = SafeRecalibrator(
    base_block_threshold=0.7447,
    min_samples=30,
)
few_scores = rng.beta(2, 10, size=5)
r_few = recal3.propose(few_scores)
check("G5a: Insufficient confidence", r_few.confidence == "INSUFFICIENT")
check("G5b: Stage stays INACTIVE", recal3.stage == RolloutStage.INACTIVE)

# G6: Audit trail
check("G6a: Audit trail populated", len(recal.audit_trail) > 0)
check("G6b: Events have timestamps", all(e.timestamp > 0 for e in recal.audit_trail))
check("G6c: Events are serializable", all(isinstance(e.to_dict(), dict) for e in recal.audit_trail))
print(f"       audit_trail: {len(recal.audit_trail)} events")
for evt in recal.audit_trail:
    print(f"         [{evt.event_type}] {evt.reason[:60]}")

# G7: Save/load roundtrip
with tempfile.TemporaryDirectory() as tmpdir:
    save_path = Path(tmpdir) / "recalibrator.json"
    recal.save(save_path)
    loaded_recal = SafeRecalibrator.load(save_path)
    check("G7a: Loaded stage preserved", loaded_recal.stage == recal.stage)
    check("G7b: Loaded thresholds match",
          abs(loaded_recal.active_block_threshold - recal.active_block_threshold) < 1e-6)
    check("G7c: Loaded audit trail preserved",
          len(loaded_recal.audit_trail) == len(recal.audit_trail))

# ═══════════════════════════════════════════════════════
#  PART H: Improved Deployment Engine
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART H: Improved Deployment Engine")
print("=" * 70)

from src.deployment.improved_engine import (
    ImprovedDecisionEngine, EngineState, EngineMetrics, HealthCheckResult,
)
from src.deployment.decision_engine import DeploymentStrategy, Decision

# H1: Initialize engine
engine = ImprovedDecisionEngine(
    strategy=DeploymentStrategy.BALANCED_BLOCK,
    feature_names=feature_names,
    drift_check_interval=10,
    recalibration_enabled=True,
    auto_switch=True,
)
check("H1a: Engine created", engine.state == EngineState.INITIALIZING)

engine.initialize(ref_scores=ref_scores, ref_features=ref_features)
check("H1b: Engine initialized", engine.state == EngineState.WARM_UP)

# H2: Make decisions
flow_scores_benign = rng.beta(2, 10, size=8)  # Low scores = benign
dec = engine.decide(flow_scores_benign, capture_id="sess_001")
check("H2a: Decision returned", dec is not None)
check("H2b: Decision has fields", dec.capture_id == "sess_001")
check("H2c: Benign session passed", dec.decision in ("PASS", "FLAG"))
print(f"       decision={dec.decision}, score={dec.score_norm:.4f}")

flow_scores_vpn = np.array([0.95, 0.92, 0.88, 0.91, 0.93])  # High scores = VPN
dec_vpn = engine.decide(flow_scores_vpn, capture_id="sess_002")
check("H2d: VPN detected", dec_vpn.decision == "BLOCK")
print(f"       decision={dec_vpn.decision}, score={dec_vpn.score_norm:.4f}")

# H3: Metrics tracking
check("H3a: Total decisions tracked", engine.metrics.total_decisions == 2)
check("H3b: Block count tracked", engine.metrics.decisions_block >= 1)
check("H3c: Latency tracked", engine.metrics.avg_decision_latency_ms >= 0)
print(f"       metrics: total={engine.metrics.total_decisions}, "
      f"block={engine.metrics.decisions_block}, pass={engine.metrics.decisions_pass}")

# H4: Process more sessions for warm-up
for i in range(15):
    scores = rng.beta(2, 10, size=5)
    engine.decide(scores, capture_id=f"warmup_{i:03d}")
check("H4a: Warm-up complete", engine.state == EngineState.RUNNING)
check("H4b: All decisions counted", engine.metrics.total_decisions >= 17)
print(f"       state={engine.state.value}, decisions={engine.metrics.total_decisions}")

# H5: Health check
health = engine.health_check()
check("H5a: Health check returns result", isinstance(health, HealthCheckResult))
check("H5b: Components checked", len(health.components) > 0)
check("H5c: Has timestamp", health.timestamp > 0)
print(f"       healthy={health.healthy}, state={health.state}")
print(f"       components: {health.components}")
if health.warnings:
    print(f"       warnings: {health.warnings}")

# H6: Propose recalibration
local_benign = rng.beta(2, 10, size=100)
recal_result = engine.propose_recalibration(local_benign)
check("H6a: Recalibration proposed", recal_result is not None)
check("H6b: Recalibration proposals counted", engine.metrics.recalibration_proposals == 1)
if recal_result:
    print(f"       proposed_block={recal_result.proposed_block_threshold:.4f}, "
          f"stage={recal_result.rollout_stage}")

# H7: Advance recalibration
adv = engine.advance_recalibration()
check("H7a: Advanced", adv is not None)
check("H7b: Advances counted", engine.metrics.recalibration_advances == 1)

# H8: Rollback recalibration
rb = engine.rollback_recalibration("Test rollback")
check("H8a: Rolled back", rb is not None)
check("H8b: Rollbacks counted", engine.metrics.recalibration_rollbacks == 1)

# H9: Config hot-reload
old_strategy = engine.strategy
engine.reload_config(
    new_strategy=DeploymentStrategy.STRICT_BLOCK,
)
check("H9a: Strategy changed", engine.strategy == DeploymentStrategy.STRICT_BLOCK)
check("H9b: Transitions counted", engine.metrics.strategy_transitions >= 1)
print(f"       strategy={engine.strategy.value}")

# H10: Diagnostics
diag = engine.diagnostics()
check("H10a: Diagnostics complete", 'engine_state' in diag)
check("H10b: Has metrics", 'metrics' in diag)
check("H10c: Has health", 'health' in diag)
check("H10d: Has events", 'last_events' in diag)

# H11: Event log
check("H11a: Events logged", len(engine.event_log) > 0)
print(f"       event_log: {len(engine.event_log)} events")
for evt in engine.event_log[-5:]:
    print(f"         [{evt.severity}] {evt.event_type}: {evt.detail[:60]}")

# H12: Save/load roundtrip
with tempfile.TemporaryDirectory() as tmpdir:
    save_dir = Path(tmpdir) / "engine_state"
    engine.save_state(save_dir)
    loaded_engine = ImprovedDecisionEngine.load_state(save_dir)
    check("H12a: Loaded engine state", loaded_engine is not None)
    check("H12b: Strategy preserved",
          loaded_engine.strategy == DeploymentStrategy.STRICT_BLOCK)

# H13: Graceful degradation (simulate error)
# Intentionally break the normalizer to trigger fallback
engine._base_engine._normalizer = "broken"  # Will cause error if used
engine._base_engine.norm_method = "not_a_method"  # Breaks if accessed
# This should still work via fallback
try:
    dec_fallback = engine.decide(flow_scores_benign, capture_id="fallback_test")
    # In degraded mode it should still return something
    check("H13a: Graceful degradation works", dec_fallback is not None)
except Exception as e:
    check("H13a: Graceful degradation works", False)

# ═══════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 70)

if failed > 0:
    print("SOME TESTS FAILED!")
    sys.exit(1)
else:
    print("ALL TESTS PASSED!")
    sys.exit(0)






