# demo_firewall/__init__.py
"""
VPN Detection Firewall — Deployable Inference Pipeline.

A zero-block-FPR VPN detection system using ensemble inference
with calibrated probabilities and session-level aggregation.

Architecture
------------
pcap/packets → FlowTracker → EnsemblePredictor → FirewallPolicy → Decision
                 (Stage 1-2)    (Stage 3)          (Stage 4-5)

Quick Start
-----------
>>> from demo_firewall import FirewallBlocker, DeploymentMode
>>>
>>> blocker = FirewallBlocker(mode=DeploymentMode.STRICT)
>>> blocker.load()
>>> blocker.calibrate_from_validation()
>>> decision = blocker.predict_pcap("capture.pcap")
>>> print(decision.decision)  # BLOCK / FLAG / ALLOW

Deployment Modes
----------------
- STRICT:   Zero block-FPR. p90 aggregation. Production default.
- BALANCED: ≤0.1% FPR. Weighted top-5 mean. Monitored deployment.
- RESEARCH: Raw probabilities. No thresholding. Offline analysis.

Safety Guarantees
-----------------
- CalibrationError: raised if calibration split has only one class.
- ThresholdLeakageError: raised if threshold computed on contaminated data.
- Domain separability warning if direction features included.
"""

from demo_firewall.blocker import FirewallBlocker
from demo_firewall.config import DeploymentMode, ArtifactPaths, default_artifact_paths
from demo_firewall.errors import (
    FirewallPipelineError,
    CalibrationError,
    ThresholdLeakageError,
    ModelLoadError,
    FeatureExtractionError,
    InsufficientDataError,
)
from demo_firewall.flow_tracker import FlowTracker
from demo_firewall.predictor import EnsemblePredictor
from demo_firewall.policy import (
    FirewallPolicy,
    Decision,
    SessionDecision,
    FlowDecision,
)

__all__ = [
    # Main entry point
    "FirewallBlocker",
    # Configuration
    "DeploymentMode",
    "ArtifactPaths",
    "default_artifact_paths",
    # Components
    "FlowTracker",
    "EnsemblePredictor",
    "FirewallPolicy",
    # Decision types
    "Decision",
    "SessionDecision",
    "FlowDecision",
    # Errors
    "FirewallPipelineError",
    "CalibrationError",
    "ThresholdLeakageError",
    "ModelLoadError",
    "FeatureExtractionError",
    "InsufficientDataError",
]

__version__ = "1.0.0"

