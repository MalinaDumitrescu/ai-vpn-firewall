# src/deployment/__init__.py
"""
Production deployment policy layer for VPN firewall detection.

Provides score normalization, drift monitoring, adaptive thresholding,
local recalibration, and a unified decision engine that turns raw
detector scores into structured BLOCK / FLAG / PASS decisions.

Parts F-H infrastructure improvements add:
- Enhanced drift monitoring with feature-level + trend detection (Part F)
- Safe recalibration with guardrails, staged rollout, and rollback (Part G)
- Improved deployment engine with health checks, metrics, graceful degradation (Part H)
"""
from src.deployment.normalization import ScoreNormalizer, NormMethod
from src.deployment.drift_monitor import DriftMonitor, DriftReport, DriftLevel
from src.deployment.adaptive_threshold import AdaptiveThreshold
from src.deployment.recalibration import LocalRecalibrator
from src.deployment.decision_engine import (
    DecisionEngine,
    DeploymentStrategy,
    DeploymentDecision,
    DeploymentSwitcher,
    Decision,
    AGGREGATORS,
)
# Part F: Enhanced drift monitoring
from src.deployment.enhanced_drift_monitor import (
    EnhancedDriftMonitor,
    EnhancedDriftReport,
    FeatureDriftResult,
)
# Part G: Safe recalibration
from src.deployment.safe_recalibration import (
    SafeRecalibrator,
    SafeRecalibrationResult,
    RolloutStage,
)
# Part H: Improved deployment engine
from src.deployment.improved_engine import (
    ImprovedDecisionEngine,
    EngineState,
    EngineMetrics,
    HealthCheckResult,
)

__all__ = [
    # Base components
    "ScoreNormalizer",
    "NormMethod",
    "DriftMonitor",
    "DriftReport",
    "DriftLevel",
    "AdaptiveThreshold",
    "LocalRecalibrator",
    "DecisionEngine",
    "DeploymentStrategy",
    "DeploymentDecision",
    "DeploymentSwitcher",
    "Decision",
    "AGGREGATORS",
    # Part F: Enhanced drift
    "EnhancedDriftMonitor",
    "EnhancedDriftReport",
    "FeatureDriftResult",
    # Part G: Safe recalibration
    "SafeRecalibrator",
    "SafeRecalibrationResult",
    "RolloutStage",
    # Part H: Improved engine
    "ImprovedDecisionEngine",
    "EngineState",
    "EngineMetrics",
    "HealthCheckResult",
]

