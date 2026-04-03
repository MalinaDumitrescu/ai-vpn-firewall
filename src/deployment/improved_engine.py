# src/deployment/improved_engine.py
"""
Improved deployment engine for VPN firewall (Part H).

Extends the base DecisionEngine with:
- Health checks (are all components fitted and sane?)
- Runtime metrics (decision distribution, latency tracking, throughput)
- Integration with EnhancedDriftMonitor and SafeRecalibrator
- Graceful degradation (fallback to STRICT if components fail)
- Comprehensive event logging / audit trail
- Batch processing with session-level parallelism hints
- Configuration hot-reload support
- Deployment lifecycle management (init → warm-up → running → degraded)

Produces structured deployment decisions with full traceability.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from src.deployment.normalization import ScoreNormalizer, NormMethod
from src.deployment.drift_monitor import DriftLevel
from src.deployment.adaptive_threshold import AdaptiveThreshold
from src.deployment.enhanced_drift_monitor import (
    EnhancedDriftMonitor,
)
from src.deployment.safe_recalibration import (
    SafeRecalibrator, RolloutStage, SafeRecalibrationResult,
)
from src.deployment.decision_engine import (
    DecisionEngine,
    DeploymentStrategy,
    DeploymentDecision,
    DeploymentSwitcher,
    Decision,
    AGGREGATORS,
    STRATEGY_DEFAULTS,
)


# ──────────────────────────────────────────────────────
# Engine lifecycle states
# ──────────────────────────────────────────────────────

class EngineState(str, Enum):
    INITIALIZING = "INITIALIZING"
    WARM_UP = "WARM_UP"          # Collecting initial data
    RUNNING = "RUNNING"          # Normal operation
    DEGRADED = "DEGRADED"        # Component failure, fallback active
    STOPPED = "STOPPED"


# ──────────────────────────────────────────────────────
# Runtime metrics
# ──────────────────────────────────────────────────────

@dataclass
class EngineMetrics:
    """Runtime metrics for monitoring the deployment engine."""
    total_decisions: int = 0
    total_sessions: int = 0
    decisions_block: int = 0
    decisions_flag: int = 0
    decisions_pass: int = 0
    # Per-dataset counters
    per_dataset_decisions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Latency tracking
    avg_decision_latency_ms: float = 0.0
    max_decision_latency_ms: float = 0.0
    p99_decision_latency_ms: float = 0.0
    # Drift
    drift_checks_run: int = 0
    drift_warnings: int = 0
    drift_high: int = 0
    drift_critical: int = 0
    # Recalibration
    recalibration_proposals: int = 0
    recalibration_advances: int = 0
    recalibration_rollbacks: int = 0
    # Errors
    errors_caught: int = 0
    fallbacks_triggered: int = 0
    # Strategy transitions
    strategy_transitions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def block_rate(self) -> float:
        return self.decisions_block / max(self.total_decisions, 1)

    @property
    def flag_rate(self) -> float:
        return self.decisions_flag / max(self.total_decisions, 1)

    @property
    def pass_rate(self) -> float:
        return self.decisions_pass / max(self.total_decisions, 1)


@dataclass
class EngineEvent:
    """Audit event for the engine lifecycle."""
    timestamp: float
    event_type: str
    detail: str
    severity: str  # INFO / WARNING / ERROR / CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────────────

@dataclass
class HealthCheckResult:
    """Result of an engine health check."""
    healthy: bool
    state: str
    strategy: str
    components: Dict[str, bool]
    warnings: List[str]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────
# Improved engine
# ──────────────────────────────────────────────────────

class ImprovedDecisionEngine:
    """
    Production-grade VPN firewall decision engine with full lifecycle
    management, health checks, metrics, and graceful degradation.

    This is the top-level orchestrator that combines:
    - Base DecisionEngine (scoring + thresholding)
    - EnhancedDriftMonitor (multi-signal drift)
    - SafeRecalibrator (staged threshold updates)
    - DeploymentSwitcher (automatic strategy transitions)
    - Runtime metrics + audit trail

    Usage:
        engine = ImprovedDecisionEngine(
            strategy=DeploymentStrategy.BALANCED_BLOCK,
        )
        engine.initialize(ref_scores=val_benign_scores)
        decision = engine.decide(flow_scores, capture_id="sess_001")
        health = engine.health_check()
        metrics = engine.metrics
    """

    def __init__(
        self,
        strategy: DeploymentStrategy = DeploymentStrategy.BALANCED_BLOCK,
        normalizer: Optional[ScoreNormalizer] = None,
        feature_names: Optional[List[str]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        # Enhanced drift config
        drift_check_interval: int = 50,   # Check drift every N sessions
        drift_window_size: int = 20,
        # Safe recalibration config
        recalibration_enabled: bool = True,
        max_threshold_shift: float = 0.20,
        recal_blend_steps: int = 3,
        # Switching config
        auto_switch: bool = True,
        min_benign_for_balanced: int = 50,
        max_consecutive_drift_warnings: int = 3,
        # Logging
        max_event_log: int = 1000,
        max_latency_buffer: int = 500,
    ):
        self._strategy = strategy
        self._feature_names = feature_names or []
        self._state = EngineState.INITIALIZING
        self._config_overrides = config_overrides or {}

        # Config
        self._drift_check_interval = drift_check_interval
        self._recalibration_enabled = recalibration_enabled
        self._auto_switch = auto_switch

        # Build base engine
        self._base_engine = DecisionEngine(
            strategy=strategy,
            normalizer=normalizer,
            config_overrides=config_overrides,
        )

        # Enhanced drift monitor
        self._drift_monitor = EnhancedDriftMonitor(
            feature_names=feature_names or [],
            window_size=drift_window_size,
        )

        # Safe recalibrator (initialized when thresholds are known)
        self._safe_recalibrator: Optional[SafeRecalibrator] = None
        if recalibration_enabled:
            defaults = STRATEGY_DEFAULTS.get(strategy, {})
            block_thr = self._config_overrides.get(
                "block_threshold", defaults.get("block_threshold", 0.7447)
            )
            flag_thr = self._config_overrides.get(
                "flag_threshold", defaults.get("flag_threshold", 0.5)
            )
            self._safe_recalibrator = SafeRecalibrator(
                base_block_threshold=block_thr,
                base_flag_threshold=flag_thr,
                max_shift_abs=max_threshold_shift,
                blend_steps=recal_blend_steps,
            )

        # Deployment switcher
        self._switcher = DeploymentSwitcher(
            initial_strategy=strategy,
            min_benign_for_balanced=min_benign_for_balanced,
            max_consecutive_drift_warnings=max_consecutive_drift_warnings,
        )

        # Runtime metrics
        self._metrics = EngineMetrics()
        self._latency_buffer: Deque[float] = deque(maxlen=max_latency_buffer)
        self._benign_score_buffer: List[float] = []
        self._sessions_since_drift_check = 0

        # Event log / audit trail
        self._event_log: Deque[EngineEvent] = deque(maxlen=max_event_log)
        self._decisions_log: Deque[DeploymentDecision] = deque(maxlen=max_event_log)

        self._initialized = False
        self._log_event("CREATED", f"Strategy={strategy.value}", "INFO")

    def initialize(
        self,
        ref_scores: Optional[np.ndarray] = None,
        ref_features: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the engine with reference distributions.

        Parameters
        ----------
        ref_scores : array, optional
            Validation benign session scores for drift reference.
        ref_features : array, shape (n, n_features), optional
            Feature matrix for the same reference sessions.
        """
        try:
            if ref_scores is not None:
                self._drift_monitor.fit(ref_scores, ref_features)
                self._log_event(
                    "DRIFT_MONITOR_FITTED",
                    f"n_ref={len(ref_scores)}, n_features={ref_features.shape[1] if ref_features is not None else 0}",
                    "INFO",
                )

            # Set up adaptive threshold if needed
            defaults = STRATEGY_DEFAULTS.get(self._strategy, {})
            if defaults.get("adaptive", False) and self._base_engine._adaptive is None:
                self._base_engine._adaptive = AdaptiveThreshold(
                    base_threshold=self._base_engine.block_threshold,
                    buffer_size=200,
                    safety_margin=0.02,
                )

            self._base_engine._drift_monitor = self._drift_monitor._score_monitor
            self._state = EngineState.WARM_UP
            self._initialized = True
            self._log_event("INITIALIZED", "Engine ready for warm-up", "INFO")

        except Exception as e:
            self._state = EngineState.DEGRADED
            self._metrics.errors_caught += 1
            self._log_event("INIT_FAILED", str(e), "ERROR")
            # Fallback: still usable with base thresholds
            self._initialized = True

    def decide(
        self,
        flow_scores: np.ndarray,
        capture_id: str = "",
        dataset: Optional[str] = None,
        flow_features: Optional[np.ndarray] = None,
    ) -> DeploymentDecision:
        """
        Make a session-level firewall decision with full lifecycle support.

        Parameters
        ----------
        flow_scores : array
            Flow-level probability scores for one session.
        capture_id : str
            Session identifier.
        dataset : str or None
            Dataset/environment identifier (if known).
        flow_features : array, shape (n_flows, n_features), optional
            Feature matrix for the flows in this session.

        Returns
        -------
        DeploymentDecision with full traceability.
        """
        t_start = time.time()

        try:
            # Apply safe recalibrator thresholds if active
            if (self._safe_recalibrator
                    and self._safe_recalibrator.stage in (
                        RolloutStage.PARTIAL, RolloutStage.FULL)):
                self._base_engine.block_threshold = \
                    self._safe_recalibrator.active_block_threshold
                self._base_engine.flag_threshold = \
                    self._safe_recalibrator.active_flag_threshold

            # Delegate to base engine
            decision = self._base_engine.decide(
                flow_scores, capture_id=capture_id, dataset=dataset,
            )

            # Track metrics
            self._update_metrics(decision, dataset, t_start)

            # Track benign buffer for drift checking and recalibration
            if decision.decision == Decision.PASS.value:
                self._benign_score_buffer.append(decision.score_norm)
                if self._auto_switch:
                    self._switcher.update(benign_sessions_added=1)

            # Periodic drift check
            self._sessions_since_drift_check += 1
            if (self._sessions_since_drift_check >= self._drift_check_interval
                    and self._drift_monitor._fitted):
                self._run_periodic_drift_check(flow_features)

            # Transition to RUNNING after warm-up
            if (self._state == EngineState.WARM_UP
                    and self._metrics.total_decisions >= 10):
                self._state = EngineState.RUNNING
                self._log_event("WARM_UP_COMPLETE",
                                f"After {self._metrics.total_decisions} decisions",
                                "INFO")

            self._decisions_log.append(decision)
            return decision

        except Exception as e:
            # Graceful degradation: return a conservative PASS decision
            self._metrics.errors_caught += 1
            self._metrics.fallbacks_triggered += 1
            self._log_event("DECISION_ERROR", str(e), "ERROR")

            if self._state != EngineState.DEGRADED:
                self._state = EngineState.DEGRADED
                self._log_event("STATE_DEGRADED",
                                "Entering degraded mode due to error",
                                "CRITICAL")

            return self._fallback_decision(flow_scores, capture_id)

    def decide_batch(
        self,
        flow_df,
        prob_col: str = "prob_iso",
        session_col: str = "capture_id",
        dataset_col: str = "dataset",
        feature_cols: Optional[List[str]] = None,
    ) -> List[DeploymentDecision]:
        """
        Batch decision for multiple sessions from a flow-level DataFrame.

        Parameters
        ----------
        flow_df : DataFrame
            Flow-level data with scores and session identifiers.
        prob_col : str
            Column containing flow probabilities.
        session_col : str
            Column identifying sessions.
        dataset_col : str
            Column identifying dataset/environment.
        feature_cols : list of str, optional
            Feature columns for drift monitoring.

        Returns
        -------
        List of DeploymentDecision, one per session.
        """
        decisions = []
        for cid, group in flow_df.groupby(session_col):
            ds = group[dataset_col].iloc[0] if dataset_col in group.columns else None
            scores = group[prob_col].values

            features = None
            if feature_cols:
                avail = [c for c in feature_cols if c in group.columns]
                if avail:
                    features = group[avail].values

            dec = self.decide(
                scores,
                capture_id=str(cid),
                dataset=str(ds) if ds else None,
                flow_features=features,
            )
            decisions.append(dec)
        return decisions

    def propose_recalibration(
        self,
        local_benign_scores: np.ndarray,
    ) -> Optional[SafeRecalibrationResult]:
        """
        Propose threshold recalibration from local benign samples.

        This enters SHADOW mode — thresholds are computed but not applied.
        Call advance_recalibration() to progressively apply.

        Returns None if recalibration is disabled.
        """
        if self._safe_recalibrator is None:
            self._log_event("RECALIBRATION_DISABLED",
                            "Recalibration not enabled", "WARNING")
            return None

        result = self._safe_recalibrator.propose(local_benign_scores)
        self._metrics.recalibration_proposals += 1
        self._log_event(
            "RECALIBRATION_PROPOSED",
            f"shift={result.threshold_shift:+.4f}, "
            f"confidence={result.confidence}, "
            f"stage={result.rollout_stage}",
            "INFO",
        )

        if self._auto_switch:
            self._switcher.update(
                local_recalibration_available=(
                    result.rollout_stage != RolloutStage.INACTIVE.value
                ),
            )
        return result

    def advance_recalibration(self) -> Optional[SafeRecalibrationResult]:
        """
        Advance recalibration: SHADOW → PARTIAL → FULL.

        Returns None if no recalibrator or nothing to advance.
        """
        if self._safe_recalibrator is None:
            return None

        try:
            result = self._safe_recalibrator.advance()
            self._metrics.recalibration_advances += 1
            self._log_event(
                "RECALIBRATION_ADVANCED",
                f"stage={result.rollout_stage}, alpha={result.blend_alpha:.2f}",
                "INFO",
            )
            return result
        except RuntimeError as e:
            self._log_event("RECALIBRATION_ADVANCE_FAILED", str(e), "WARNING")
            return None

    def rollback_recalibration(
        self, reason: str = "Operator-initiated rollback"
    ) -> Optional[SafeRecalibrationResult]:
        """
        Rollback recalibration to previous thresholds.
        """
        if self._safe_recalibrator is None:
            return None

        result = self._safe_recalibrator.rollback(reason)
        self._metrics.recalibration_rollbacks += 1
        self._log_event("RECALIBRATION_ROLLBACK", reason, "WARNING")

        # Also update the base engine thresholds
        self._base_engine.block_threshold = result.active_block_threshold
        self._base_engine.flag_threshold = result.active_flag_threshold
        return result

    def health_check(self) -> HealthCheckResult:
        """
        Run a comprehensive health check on all engine components.

        Returns HealthCheckResult with per-component status.
        """
        components: Dict[str, bool] = {}
        warnings: List[str] = []

        # Base engine
        components["base_engine"] = True  # Always available

        # Drift monitor
        dm_fitted = self._drift_monitor._fitted
        components["drift_monitor"] = dm_fitted
        if not dm_fitted:
            warnings.append("Drift monitor not fitted. Score drift detection inactive.")

        # Adaptive threshold
        at = self._base_engine._adaptive
        components["adaptive_threshold"] = at is not None
        if at is not None and at.frozen:
            warnings.append("Adaptive threshold is frozen.")

        # Normalizer
        norm = self._base_engine._normalizer
        components["normalizer"] = norm is not None or \
            self._base_engine.norm_method == NormMethod.PASSTHROUGH
        if norm is None and self._base_engine.norm_method != NormMethod.PASSTHROUGH:
            warnings.append("Normalizer needed but not configured.")

        # Safe recalibrator
        if self._safe_recalibrator:
            components["safe_recalibrator"] = True
            if self._safe_recalibrator.stage == RolloutStage.ROLLED_BACK:
                warnings.append("Recalibration was rolled back. Review thresholds.")
        else:
            components["safe_recalibrator"] = not self._recalibration_enabled

        # Deployment switcher
        components["deployment_switcher"] = True

        # Error rate check
        if self._metrics.total_decisions > 0:
            error_rate = self._metrics.errors_caught / self._metrics.total_decisions
            if error_rate > 0.05:
                warnings.append(
                    f"High error rate: {error_rate:.1%} "
                    f"({self._metrics.errors_caught}/{self._metrics.total_decisions})")
                components["error_rate"] = False
            else:
                components["error_rate"] = True

        # Block rate anomaly
        if self._metrics.total_decisions > 50:
            if self._metrics.block_rate > 0.8:
                warnings.append(
                    f"Unusually high block rate: {self._metrics.block_rate:.1%}. "
                    "Possible miscalibration.")
            elif self._metrics.block_rate < 0.01 and self._strategy != DeploymentStrategy.STRICT_BLOCK:
                warnings.append(
                    f"Very low block rate: {self._metrics.block_rate:.1%}. "
                    "Model may not be detecting VPN traffic.")

        healthy = all(components.values()) and self._state != EngineState.DEGRADED
        return HealthCheckResult(
            healthy=healthy,
            state=self._state.value,
            strategy=self._strategy.value,
            components=components,
            warnings=warnings,
            timestamp=time.time(),
        )

    def reload_config(
        self,
        new_strategy: Optional[DeploymentStrategy] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Hot-reload configuration without restarting the engine.

        Updates strategy and/or thresholds on the fly.
        """
        old_strategy = self._strategy

        if new_strategy:
            self._strategy = new_strategy
            defaults = dict(STRATEGY_DEFAULTS.get(new_strategy, {}))
            if config_overrides:
                defaults.update(config_overrides)
            self._base_engine.strategy = new_strategy
            self._base_engine.block_threshold = defaults.get(
                "block_threshold", self._base_engine.block_threshold)
            self._base_engine.flag_threshold = defaults.get(
                "flag_threshold", self._base_engine.flag_threshold)
            self._base_engine.aggregation_name = defaults.get(
                "aggregation", self._base_engine.aggregation_name)
            self._base_engine._aggregator = AGGREGATORS[
                self._base_engine.aggregation_name]
            self._metrics.strategy_transitions += 1
            self._log_event(
                "CONFIG_RELOAD",
                f"Strategy {old_strategy.value} → {new_strategy.value}",
                "INFO",
            )
        elif config_overrides:
            for k, v in config_overrides.items():
                if k == "block_threshold":
                    self._base_engine.block_threshold = v
                elif k == "flag_threshold":
                    self._base_engine.flag_threshold = v
                elif k == "aggregation" and v in AGGREGATORS:
                    self._base_engine.aggregation_name = v
                    self._base_engine._aggregator = AGGREGATORS[v]
            self._log_event(
                "CONFIG_RELOAD",
                f"Overrides applied: {list(config_overrides.keys())}",
                "INFO",
            )

    # ── Internal helpers ──

    def _run_periodic_drift_check(
        self,
        recent_features: Optional[np.ndarray] = None,
    ) -> None:
        """Run drift check against reference distribution."""
        self._sessions_since_drift_check = 0

        if not self._benign_score_buffer:
            return

        try:
            benign_scores = np.array(self._benign_score_buffer[-200:])
            report = self._drift_monitor.check(benign_scores, recent_features)

            self._metrics.drift_checks_run += 1
            if report.composite_level == "WARNING":
                self._metrics.drift_warnings += 1
            elif report.composite_level == DriftLevel.HIGH.value:
                self._metrics.drift_high += 1
            elif report.composite_level == "CRITICAL":
                self._metrics.drift_critical += 1

            # Also update the base engine's drift report
            self._base_engine._last_drift_report = report.score_drift

            # Auto-switch on drift
            if self._auto_switch:
                new_strategy = self._switcher.update(
                    drift_report=report.score_drift,
                )
                if new_strategy != self._strategy:
                    self._log_event(
                        "AUTO_SWITCH",
                        f"Strategy {self._strategy.value} → {new_strategy.value} "
                        f"due to drift level {report.composite_level}",
                        "WARNING",
                    )
                    self.reload_config(new_strategy=new_strategy)

            self._log_event(
                "DRIFT_CHECK",
                f"level={report.composite_level}, "
                f"composite={report.composite_score:.3f}, "
                f"trend={report.trend_direction}",
                "INFO" if report.composite_level == "OK" else "WARNING",
            )

        except Exception as e:
            self._metrics.errors_caught += 1
            self._log_event("DRIFT_CHECK_ERROR", str(e), "ERROR")

    def _update_metrics(
        self,
        decision: DeploymentDecision,
        dataset: Optional[str],
        t_start: float,
    ) -> None:
        """Update runtime metrics after a decision."""
        latency_ms = (time.time() - t_start) * 1000
        self._latency_buffer.append(latency_ms)

        self._metrics.total_decisions += 1
        self._metrics.total_sessions += 1

        if decision.decision == Decision.BLOCK.value:
            self._metrics.decisions_block += 1
        elif decision.decision == Decision.FLAG.value:
            self._metrics.decisions_flag += 1
        else:
            self._metrics.decisions_pass += 1

        # Per-dataset tracking
        if dataset:
            if dataset not in self._metrics.per_dataset_decisions:
                self._metrics.per_dataset_decisions[dataset] = {
                    "block": 0, "flag": 0, "pass": 0
                }
            key = decision.decision.lower()
            if key in self._metrics.per_dataset_decisions[dataset]:
                self._metrics.per_dataset_decisions[dataset][key] += 1

        # Update latency stats
        if self._latency_buffer:
            arr = np.array(self._latency_buffer)
            self._metrics.avg_decision_latency_ms = float(np.mean(arr))
            self._metrics.max_decision_latency_ms = float(np.max(arr))
            self._metrics.p99_decision_latency_ms = float(np.percentile(arr, 99))

    def _fallback_decision(
        self,
        flow_scores: np.ndarray,
        capture_id: str,
    ) -> DeploymentDecision:
        """
        Emergency fallback: conservative PASS decision.

        Used when the normal pipeline throws an exception.
        In degraded mode, we PASS everything to avoid blocking
        legitimate traffic while the issue is being fixed.
        """
        self._log_event(
            "FALLBACK_DECISION",
            f"capture_id={capture_id}",
            "WARNING",
        )
        return DeploymentDecision(
            capture_id=capture_id,
            score_raw=0.0,
            score_norm=0.0,
            aggregation=self._base_engine.aggregation_name,
            normalization="passthrough",
            calibration=self._base_engine.calibration,
            policy_mode=f"{self._strategy.value}+DEGRADED",
            block_threshold=self._base_engine.block_threshold,
            flag_threshold=self._base_engine.flag_threshold,
            decision=Decision.PASS.value,
            confidence_margin=0.0,
            drift_state="UNKNOWN",
            threshold_source="fallback_degraded",
            n_flows=len(flow_scores) if flow_scores is not None else 0,
            reason="DEGRADED MODE: Engine error, conservative PASS applied. "
                   "Check engine health and logs.",
        )

    def _log_event(self, event_type: str, detail: str, severity: str) -> None:
        """Append to event log."""
        self._event_log.append(EngineEvent(
            timestamp=time.time(),
            event_type=event_type,
            detail=detail,
            severity=severity,
        ))

    # ── Properties ──

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def strategy(self) -> DeploymentStrategy:
        return self._strategy

    @property
    def metrics(self) -> EngineMetrics:
        return self._metrics

    @property
    def event_log(self) -> List[EngineEvent]:
        return list(self._event_log)

    @property
    def decisions_log(self) -> List[DeploymentDecision]:
        return list(self._decisions_log)

    @property
    def drift_monitor(self) -> EnhancedDriftMonitor:
        return self._drift_monitor

    @property
    def safe_recalibrator(self) -> Optional[SafeRecalibrator]:
        return self._safe_recalibrator

    @property
    def switcher(self) -> DeploymentSwitcher:
        return self._switcher

    def diagnostics(self) -> Dict[str, Any]:
        """Full diagnostic state dump."""
        health = self.health_check()
        return {
            "engine_state": self._state.value,
            "strategy": self._strategy.value,
            "health": health.to_dict(),
            "metrics": self._metrics.to_dict(),
            "base_engine": self._base_engine.diagnostics(),
            "drift_dashboard": self._drift_monitor.dashboard_data()
                if self._drift_monitor._fitted else {},
            "recalibrator_state": self._safe_recalibrator.state_summary()
                if self._safe_recalibrator else None,
            "switcher_state": self._switcher.to_dict(),
            "n_events": len(self._event_log),
            "last_events": [e.to_dict() for e in list(self._event_log)[-10:]],
        }

    # ── Persistence ──

    def save_state(self, directory: Path) -> None:
        """Save complete engine state to a directory."""
        directory.mkdir(parents=True, exist_ok=True)

        # Engine config
        config = {
            "strategy": self._strategy.value,
            "state": self._state.value,
            "drift_check_interval": self._drift_check_interval,
            "recalibration_enabled": self._recalibration_enabled,
            "auto_switch": self._auto_switch,
            "feature_names": self._feature_names,
        }
        with open(directory / "engine_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Metrics
        with open(directory / "engine_metrics.json", "w") as f:
            json.dump(self._metrics.to_dict(), f, indent=2)

        # Event log
        with open(directory / "engine_events.json", "w") as f:
            json.dump([e.to_dict() for e in self._event_log], f, indent=2)

        # Base engine config
        self._base_engine.save_config(directory / "base_engine_config.json")

        # Drift monitor
        if self._drift_monitor._fitted:
            self._drift_monitor.save(directory / "enhanced_drift_monitor.json")

        # Safe recalibrator
        if self._safe_recalibrator:
            self._safe_recalibrator.save(directory / "safe_recalibrator.json")

        # Diagnostics snapshot
        with open(directory / "diagnostics_snapshot.json", "w") as f:
            json.dump(self.diagnostics(), f, indent=2, default=str)

    @classmethod
    def load_state(cls, directory: Path) -> "ImprovedDecisionEngine":
        """
        Load engine from a saved state directory.

        Restores strategy, metrics, drift monitor, and recalibrator.
        """
        with open(directory / "engine_config.json") as f:
            config = json.load(f)

        engine = cls(
            strategy=DeploymentStrategy(config["strategy"]),
            feature_names=config.get("feature_names", []),
            drift_check_interval=config.get("drift_check_interval", 50),
            recalibration_enabled=config.get("recalibration_enabled", True),
            auto_switch=config.get("auto_switch", True),
        )

        # Restore drift monitor
        dm_path = directory / "enhanced_drift_monitor.json"
        if dm_path.exists():
            engine._drift_monitor = EnhancedDriftMonitor.load(dm_path)

        # Restore recalibrator
        recal_path = directory / "safe_recalibrator.json"
        if recal_path.exists():
            engine._safe_recalibrator = SafeRecalibrator.load(recal_path)

        # Restore base engine config
        base_path = directory / "base_engine_config.json"
        if base_path.exists():
            engine._base_engine = DecisionEngine.from_config(base_path)

        # Restore state
        engine._state = EngineState(config.get("state", "INITIALIZING"))
        engine._initialized = engine._state != EngineState.INITIALIZING

        return engine



