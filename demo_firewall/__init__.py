# demo_firewall/__init__.py
"""
VPN Detection Firewall — Deployable Inference Pipeline.

A zero-block-FPR VPN detection system using ensemble inference
with calibrated probabilities and session-level aggregation.

Architecture
------------
pcap/packets → FlowTracker → EnsemblePredictor → FirewallPolicy → Decision
                 (Stage 1-2)    (Stage 3)          (Stage 4-5)

Open-Set Policy (new — full_canonical__lgbm)
--------------------------------------------
An uncertainty-aware three-tier policy that reduces risky binary decisions
under domain shift.  Uses validation-derived thresholds:

  PASS             score < 0.027090  (p95 benign)
  FLAG_REVIEW      0.027090 <= score < 0.165365  (uncertainty band)
  SIMULATED_BLOCK  score >= 0.165365  [SIMULATION ONLY]

Quick Start
-----------
>>> from demo_firewall import FirewallBlocker, DeploymentMode
>>>
>>> blocker = FirewallBlocker(mode=DeploymentMode.STRICT)
>>> blocker.load()
>>> blocker.calibrate_from_validation()
>>> decision = blocker.predict_pcap("capture.pcap")
>>> print(decision.decision)  # BLOCK / FLAG / ALLOW

Open-Set Quick Start
--------------------
>>> from demo_firewall.open_set_policy import load_policy
>>> import pandas as pd
>>>
>>> policy = load_policy(repo_root=Path("."))
>>> preds = pd.read_csv("val_predictions.csv")
>>> decisions = policy.evaluate_dataframe(preds)
>>> report = policy.dashboard_report(decisions)
>>> print(policy.render_dashboard(report))

Deployment Modes
----------------
- STRICT:    Zero block-FPR. p90 aggregation. Production default.
- BALANCED:  ≤0.1% FPR. Weighted top-5 mean. Monitored deployment.
- RESEARCH:  Raw probabilities. No thresholding. Offline analysis.
- OPEN_SET:  Three-tier uncertainty-aware. PASS/FLAG_REVIEW/SIMULATED_BLOCK.

Safety Guarantees
-----------------
- CalibrationError: raised if calibration split has only one class.
- ThresholdLeakageError: raised if threshold computed on contaminated data.
- Domain separability warning if direction features included.
- All SIMULATED_BLOCK decisions are simulation only — no real packet blocking.
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
from demo_firewall.open_set_policy import (
    OpenSetFirewallPolicy,
    OpenSetThresholds,
    FirewallAction,
    SessionDecisionV2,
    FlowRecord,
    load_policy,
)
from demo_firewall.report import (
    render_status_cards,
    render_events_table,
    render_open_set_dashboard,
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
    # Legacy decision types
    "Decision",
    "SessionDecision",
    "FlowDecision",
    # Open-set three-tier policy
    "OpenSetFirewallPolicy",
    "OpenSetThresholds",
    "FirewallAction",
    "SessionDecisionV2",
    "FlowRecord",
    "load_policy",
    # Dashboard rendering
    "render_status_cards",
    "render_events_table",
    "render_open_set_dashboard",
    # Errors
    "FirewallPipelineError",
    "CalibrationError",
    "ThresholdLeakageError",
    "ModelLoadError",
    "FeatureExtractionError",
    "InsufficientDataError",
]

__version__ = "1.1.0"
