# src/deployment/decision_engine.py
"""
Unified firewall decision engine.

Combines:
- Score normalization (passthrough / rank-norm / z-norm)
- Session aggregation (p90 / wt5 / wt7 / p80 / p85 / median / trimmed_mean)
- Drift monitoring (KS / PSI)
- Adaptive thresholding (benign buffer)
- Local recalibration support
- Drift-reactive escalation
- Deployment switching logic

Produces structured deployment decisions with full traceability.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.deployment.normalization import ScoreNormalizer, NormMethod
from src.deployment.drift_monitor import DriftMonitor, DriftReport, DriftLevel
from src.deployment.adaptive_threshold import AdaptiveThreshold
from src.deployment.recalibration import LocalRecalibrator, RecalibrationResult


# ──────────────────────────────────────────────────────
# Strategy and decision types
# ──────────────────────────────────────────────────────

class DeploymentStrategy(str, Enum):
    """Available deployment strategies."""
    STRICT_BLOCK = "strict_block"                   # Zero/near-zero FPR, auto-block
    BALANCED_BLOCK = "balanced_block"                # High recall with low FPR
    FLAG_REVIEW = "flag_review"                      # Two-tier block + flag
    UNKNOWN_ENV_ADAPTIVE = "unknown_env_adaptive"    # Context-aware adaptive
    CONSERVATIVE_RAW = "conservative_raw"            # Raw scores, strict threshold
    LOCAL_RECALIBRATION = "local_recalibration"      # Locally recalibrated thresholds

    # Backward compat alias
    ADAPTIVE_ENV = "unknown_env_adaptive"


class Decision(str, Enum):
    BLOCK = "BLOCK"
    FLAG = "FLAG"
    PASS = "PASS"


@dataclass
class DeploymentDecision:
    """Full structured decision output for one session."""
    capture_id: str
    score_raw: float           # Pre-normalization aggregated score
    score_norm: float          # Post-normalization score
    aggregation: str           # Aggregation rule used
    normalization: str         # Normalization method used
    calibration: str           # Calibration method (prob_iso / prob_raw / etc.)
    policy_mode: str           # DeploymentStrategy name
    block_threshold: float
    flag_threshold: float
    decision: str              # BLOCK / FLAG / PASS
    confidence_margin: float   # Distance from nearest threshold
    drift_state: str           # OK / WARNING / HIGH
    threshold_source: str      # Where the threshold came from
    n_flows: int
    reason: str                # Human-readable explanation

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────
# Aggregation functions
# ──────────────────────────────────────────────────────

def _p80_agg(x: np.ndarray) -> float:
    return float(np.percentile(x, 80)) if len(x) > 0 else 0.0

def _p85_agg(x: np.ndarray) -> float:
    return float(np.percentile(x, 85)) if len(x) > 0 else 0.0

def _p90_agg(x: np.ndarray) -> float:
    return float(np.percentile(x, 90)) if len(x) > 0 else 0.0

def _wt5_agg(x: np.ndarray) -> float:
    vals = np.sort(x)[::-1][:5]
    if len(vals) == 0:
        return 0.0
    w = np.array([0.40, 0.25, 0.15, 0.10, 0.10])[:len(vals)]
    w = w / w.sum()
    return float(np.sum(vals * w))

def _wt7_agg(x: np.ndarray) -> float:
    """Weighted top-7 aggregation — slightly more robust than wt5."""
    vals = np.sort(x)[::-1][:7]
    if len(vals) == 0:
        return 0.0
    w = np.array([0.30, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05])[:len(vals)]
    w = w / w.sum()
    return float(np.sum(vals * w))

def _median_agg(x: np.ndarray) -> float:
    return float(np.median(x)) if len(x) > 0 else 0.0

def _trimmed_mean_agg(x: np.ndarray) -> float:
    """Trimmed mean: remove bottom 10% and top 10%, average the rest."""
    if len(x) < 5:
        return float(np.mean(x)) if len(x) > 0 else 0.0
    s = np.sort(x)
    n = len(s)
    lo = max(1, int(n * 0.1))
    hi = n - lo
    return float(np.mean(s[lo:hi]))


AGGREGATORS = {
    "p80": _p80_agg,
    "p85": _p85_agg,
    "p90": _p90_agg,
    "wt5": _wt5_agg,
    "wt7": _wt7_agg,
    "median": _median_agg,
    "trimmed_mean": _trimmed_mean_agg,
}

# ──────────────────────────────────────────────────────
# Strategy configurations
# ──────────────────────────────────────────────────────

# Default configurations for each strategy
STRATEGY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    DeploymentStrategy.STRICT_BLOCK: {
        "aggregation": "p90",
        "calibration": "prob_raw",
        "normalization": NormMethod.PASSTHROUGH,
        "block_threshold": 0.9697,
        "flag_threshold": 0.90,
        "adaptive": False,
        "recalibrate": False,
        "drift_escalation": True,
        "drift_tighten_factor": 0.8,
        "description": "Zero-FPR automatic blocking. Raw scores, strict threshold.",
    },
    DeploymentStrategy.BALANCED_BLOCK: {
        "aggregation": "wt5",
        "calibration": "prob_iso",
        "normalization": NormMethod.PASSTHROUGH,
        "block_threshold": 0.7447,
        "flag_threshold": 0.4977,
        "adaptive": False,
        "recalibrate": False,
        "drift_escalation": True,
        "drift_tighten_factor": 0.85,
        "description": "High recall with low FPR. Recommended production mode.",
    },
    DeploymentStrategy.FLAG_REVIEW: {
        "aggregation": "wt5",
        "calibration": "prob_iso",
        "normalization": NormMethod.PASSTHROUGH,
        "block_threshold": 0.7447,
        "flag_threshold": 0.4977,
        "adaptive": False,
        "recalibrate": False,
        "drift_escalation": False,
        "description": "Two-tier: confident BLOCK + borderline FLAG for review.",
    },
    DeploymentStrategy.UNKNOWN_ENV_ADAPTIVE: {
        "aggregation": "wt5",
        "calibration": "prob_iso",
        "normalization": NormMethod.PASSTHROUGH,
        "block_threshold": 0.7447,
        "flag_threshold": 0.4977,
        "adaptive": True,
        "recalibrate": False,
        "drift_escalation": True,
        "drift_tighten_factor": 0.8,
        "description": "Starts strict, adapts thresholds from benign traffic buffer.",
    },
    DeploymentStrategy.CONSERVATIVE_RAW: {
        "aggregation": "p80",
        "calibration": "prob_raw",
        "normalization": NormMethod.RANK_NORM,
        "block_threshold": 0.9675,
        "flag_threshold": 0.90,
        "adaptive": False,
        "recalibrate": False,
        "drift_escalation": False,
        "description": "Rank-normalized raw scores. Maximum FPR safety.",
    },
    DeploymentStrategy.LOCAL_RECALIBRATION: {
        "aggregation": "wt5",
        "calibration": "prob_iso",
        "normalization": NormMethod.PASSTHROUGH,
        "block_threshold": 0.7447,
        "flag_threshold": 0.4977,
        "adaptive": False,
        "recalibrate": True,
        "min_recal_samples": 30,
        "drift_escalation": True,
        "drift_tighten_factor": 0.85,
        "description": "Uses local benign samples for threshold recalibration.",
    },
}


# ──────────────────────────────────────────────────────
# Deployment switching logic
# ──────────────────────────────────────────────────────

class DeploymentSwitcher:
    """
    Implements recommended deployment switching logic:

    - unknown environment → STRICT_BLOCK
    - low drift + enough benign local samples → BALANCED_BLOCK
    - persistent drift → STRICT_BLOCK + request local recalibration
    - human-supervised environment → FLAG_REVIEW mode
    - local recalibration artifact present → LOCAL_RECALIBRATION

    All transitions are logged and auditable.
    """

    def __init__(
        self,
        initial_strategy: DeploymentStrategy = DeploymentStrategy.STRICT_BLOCK,
        min_benign_for_balanced: int = 50,
        min_benign_for_recalibration: int = 30,
        max_consecutive_drift_warnings: int = 3,
    ):
        self.current_strategy = initial_strategy
        self.min_benign_for_balanced = min_benign_for_balanced
        self.min_benign_for_recalibration = min_benign_for_recalibration
        self.max_consecutive_drift_warnings = max_consecutive_drift_warnings

        self._benign_count = 0
        self._consecutive_drift_warnings = 0
        self._transitions: List[Dict[str, Any]] = []
        self._has_local_recal = False
        self._is_supervised = False

    def update(
        self,
        drift_report: Optional[DriftReport] = None,
        benign_sessions_added: int = 0,
        local_recalibration_available: bool = False,
        supervised_mode: bool = False,
    ) -> DeploymentStrategy:
        """
        Evaluate current state and potentially switch deployment strategy.
        Returns the (possibly updated) strategy.
        """
        old_strategy = self.current_strategy
        self._benign_count += benign_sessions_added
        self._has_local_recal = local_recalibration_available
        self._is_supervised = supervised_mode

        # Track drift
        if drift_report is not None:
            if drift_report.level in (DriftLevel.WARNING, DriftLevel.HIGH):
                self._consecutive_drift_warnings += 1
            else:
                self._consecutive_drift_warnings = 0

        # Decision logic
        if self._is_supervised:
            new_strategy = DeploymentStrategy.FLAG_REVIEW
        elif self._consecutive_drift_warnings >= self.max_consecutive_drift_warnings:
            new_strategy = DeploymentStrategy.STRICT_BLOCK
        elif drift_report and drift_report.level == DriftLevel.HIGH:
            new_strategy = DeploymentStrategy.STRICT_BLOCK
        elif self._has_local_recal and self._benign_count >= self.min_benign_for_recalibration:
            new_strategy = DeploymentStrategy.LOCAL_RECALIBRATION
        elif self._benign_count >= self.min_benign_for_balanced and self._consecutive_drift_warnings == 0:
            new_strategy = DeploymentStrategy.BALANCED_BLOCK
        else:
            new_strategy = DeploymentStrategy.STRICT_BLOCK

        if new_strategy != old_strategy:
            self._transitions.append({
                "from": old_strategy.value,
                "to": new_strategy.value,
                "benign_count": self._benign_count,
                "consecutive_drift_warnings": self._consecutive_drift_warnings,
                "has_local_recal": self._has_local_recal,
                "is_supervised": self._is_supervised,
            })

        self.current_strategy = new_strategy
        return new_strategy

    @property
    def transitions(self) -> List[Dict[str, Any]]:
        return list(self._transitions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_strategy": self.current_strategy.value,
            "benign_count": self._benign_count,
            "consecutive_drift_warnings": self._consecutive_drift_warnings,
            "has_local_recal": self._has_local_recal,
            "is_supervised": self._is_supervised,
            "transitions": self._transitions,
        }


# ──────────────────────────────────────────────────────
# Decision engine
# ──────────────────────────────────────────────────────

class DecisionEngine:
    """
    Production VPN firewall decision engine.

    Turns flow-level detector scores into structured session-level
    BLOCK / FLAG / PASS decisions with full provenance.

    Usage:
        engine = DecisionEngine(strategy=DeploymentStrategy.BALANCED_BLOCK)
        decision = engine.decide(flow_scores, capture_id="session_001")
    """

    def __init__(
        self,
        strategy: DeploymentStrategy = DeploymentStrategy.BALANCED_BLOCK,
        normalizer: Optional[ScoreNormalizer] = None,
        drift_monitor: Optional[DriftMonitor] = None,
        adaptive_threshold: Optional[AdaptiveThreshold] = None,
        recalibrator: Optional[LocalRecalibrator] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.strategy = strategy
        defaults = dict(STRATEGY_DEFAULTS.get(strategy, {}))
        if config_overrides:
            defaults.update(config_overrides)

        self.aggregation_name: str = defaults["aggregation"]
        self.calibration: str = defaults["calibration"]
        self.norm_method = defaults.get("normalization", NormMethod.PASSTHROUGH)
        self.block_threshold: float = defaults["block_threshold"]
        self.flag_threshold: float = defaults["flag_threshold"]
        self.use_adaptive: bool = defaults.get("adaptive", False)
        self.use_recalibrate: bool = defaults.get("recalibrate", False)
        self.drift_escalation: bool = defaults.get("drift_escalation", True)
        self.drift_tighten_factor: float = defaults.get("drift_tighten_factor", 0.8)
        self.description: str = defaults.get("description", "")

        self._aggregator = AGGREGATORS[self.aggregation_name]
        self._normalizer = normalizer
        self._drift_monitor = drift_monitor
        self._adaptive = adaptive_threshold
        self._recalibrator = recalibrator
        self._recalibration_result: Optional[RecalibrationResult] = None
        self._last_drift_report: Optional[DriftReport] = None
        self._decisions_log: List[DeploymentDecision] = []

        # If adaptive but no AdaptiveThreshold provided, create one
        if self.use_adaptive and self._adaptive is None:
            self._adaptive = AdaptiveThreshold(
                base_threshold=self.block_threshold,
                buffer_size=200,
                safety_margin=0.02,
            )

    def apply_local_recalibration(self, local_benign_scores: np.ndarray) -> RecalibrationResult:
        """
        Apply local recalibration from benign traffic samples.
        Updates block/flag thresholds if recalibration confidence is acceptable.
        """
        if self._recalibrator is None:
            self._recalibrator = LocalRecalibrator(
                base_block_threshold=self.block_threshold,
                base_flag_threshold=self.flag_threshold,
            )

        result = self._recalibrator.recalibrate(local_benign_scores)
        self._recalibration_result = result

        if result.confidence in ("high", "moderate"):
            self.block_threshold = result.local_block_threshold
            self.flag_threshold = result.local_flag_threshold
            return result
        else:
            # Low confidence: keep base thresholds
            return result

    def decide(
        self,
        flow_scores: np.ndarray,
        capture_id: str = "",
        dataset: Optional[str] = None,
    ) -> DeploymentDecision:
        """
        Make a session-level firewall decision.

        Parameters
        ----------
        flow_scores : array
            Flow-level probability scores for one session.
        capture_id : str
            Session identifier.
        dataset : str or None
            Dataset/environment identifier (if known).

        Returns
        -------
        DeploymentDecision with full decision traceability.
        """
        flow_scores = np.asarray(flow_scores, dtype=float)
        flow_scores = flow_scores[np.isfinite(flow_scores)]

        if len(flow_scores) == 0:
            return DeploymentDecision(
                capture_id=capture_id, score_raw=0.0, score_norm=0.0,
                aggregation=self.aggregation_name, normalization=self.norm_method.value
                if isinstance(self.norm_method, NormMethod) else str(self.norm_method),
                calibration=self.calibration,
                policy_mode=self.strategy.value,
                block_threshold=self.block_threshold,
                flag_threshold=self.flag_threshold,
                decision=Decision.PASS.value,
                confidence_margin=self.flag_threshold,
                drift_state=DriftLevel.OK.value,
                threshold_source="base",
                n_flows=0,
                reason="No valid flow scores.",
            )

        # 1. Aggregate
        score_raw = self._aggregator(flow_scores)

        # 2. Normalize
        if self._normalizer and self.norm_method != NormMethod.PASSTHROUGH:
            score_norm = float(self._normalizer.transform(
                np.array([score_raw]), dataset=dataset
            )[0])
        else:
            score_norm = score_raw

        # 3. Get effective thresholds
        threshold_source = "base_val"
        if self.use_adaptive and self._adaptive:
            block_thr = self._adaptive.current_threshold
            threshold_source = "adaptive"
        elif self._recalibration_result and self._recalibration_result.confidence in ("high", "moderate"):
            block_thr = self._recalibration_result.local_block_threshold
            threshold_source = f"local_recal_{self._recalibration_result.confidence}"
        else:
            block_thr = self.block_threshold
        flag_thr = self.flag_threshold

        # 4. Drift check (if monitor available)
        drift_state = DriftLevel.OK
        if self._drift_monitor and self._last_drift_report:
            drift_state = self._last_drift_report.level

        # 4a. Drift-reactive escalation
        if self.drift_escalation and drift_state == DriftLevel.HIGH:
            flag_thr = flag_thr * self.drift_tighten_factor
            threshold_source += "+drift_escalated"

        # 5. Decision
        if score_norm >= block_thr:
            decision = Decision.BLOCK
            margin = score_norm - block_thr
            reason = (f"Session score {score_norm:.4f} >= block threshold {block_thr:.4f}. "
                      f"Automatic VPN block.")
        elif score_norm >= flag_thr:
            if self.strategy == DeploymentStrategy.FLAG_REVIEW:
                decision = Decision.FLAG
                margin = min(score_norm - flag_thr, block_thr - score_norm)
                reason = (f"Session score {score_norm:.4f} in FLAG zone "
                          f"[{flag_thr:.4f}, {block_thr:.4f}). Manual review recommended.")
            elif self.strategy == DeploymentStrategy.STRICT_BLOCK:
                decision = Decision.PASS
                margin = block_thr - score_norm  # FIX: was flag_thr - score_norm (wrong)
                reason = (f"Session score {score_norm:.4f} below strict block threshold "
                          f"{block_thr:.4f}. Passed.")
            else:
                decision = Decision.FLAG
                margin = min(score_norm - flag_thr, block_thr - score_norm)
                reason = (f"Session score {score_norm:.4f} in borderline zone. "
                          f"Flagged for review.")
        else:
            decision = Decision.PASS
            margin = flag_thr - score_norm
            reason = (f"Session score {score_norm:.4f} < flag threshold {flag_thr:.4f}. "
                      f"Traffic allowed.")

        # Append drift state to reason
        if drift_state != DriftLevel.OK:
            reason = f"[{drift_state.value}] " + reason

        # 6. Update adaptive buffer if PASS (presumed benign)
        if self.use_adaptive and self._adaptive and decision == Decision.PASS:
            self._adaptive.update(score_norm)

        norm_label = (self.norm_method.value
                      if isinstance(self.norm_method, NormMethod)
                      else str(self.norm_method))

        dec = DeploymentDecision(
            capture_id=capture_id,
            score_raw=round(score_raw, 6),
            score_norm=round(score_norm, 6),
            aggregation=self.aggregation_name,
            normalization=norm_label,
            calibration=self.calibration,
            policy_mode=self.strategy.value,
            block_threshold=round(block_thr, 6),
            flag_threshold=round(flag_thr, 6),
            decision=decision.value,
            confidence_margin=round(margin, 6),
            drift_state=drift_state.value,
            threshold_source=threshold_source,
            n_flows=len(flow_scores),
            reason=reason,
        )
        self._decisions_log.append(dec)
        return dec

    def decide_batch(
        self,
        flow_df,
        prob_col: str = "prob_iso",
        session_col: str = "capture_id",
        dataset_col: str = "dataset",
    ) -> List[DeploymentDecision]:
        """
        Batch decision for multiple sessions from a flow-level DataFrame.
        """
        decisions = []
        for cid, group in flow_df.groupby(session_col):
            ds = group[dataset_col].iloc[0] if dataset_col in group.columns else None
            scores = group[prob_col].values
            dec = self.decide(scores, capture_id=str(cid), dataset=str(ds) if ds else None)
            decisions.append(dec)
        return decisions

    def run_drift_check(self, session_scores: np.ndarray) -> DriftReport:
        """Run drift check against reference distribution."""
        if self._drift_monitor is None:
            raise RuntimeError("No drift monitor configured.")
        report = self._drift_monitor.check(session_scores)
        self._last_drift_report = report
        return report

    @property
    def decisions_log(self) -> List[DeploymentDecision]:
        return list(self._decisions_log)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "aggregation": self.aggregation_name,
            "calibration": self.calibration,
            "normalization": self.norm_method.value if isinstance(self.norm_method, NormMethod)
                             else str(self.norm_method),
            "block_threshold": self.block_threshold,
            "flag_threshold": self.flag_threshold,
            "adaptive": self.use_adaptive,
            "adaptive_state": self._adaptive.state.to_dict() if self._adaptive else None,
            "recalibrate": self.use_recalibrate,
            "recalibration_state": self._recalibration_result.to_dict() if self._recalibration_result else None,
            "drift_escalation": self.drift_escalation,
            "drift_tighten_factor": self.drift_tighten_factor,
            "last_drift": self._last_drift_report.to_dict() if self._last_drift_report else None,
            "n_decisions": len(self._decisions_log),
        }

    def save_config(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.diagnostics(), f, indent=2, default=str)

    @classmethod
    def from_config(cls, config_path: Path) -> "DecisionEngine":
        """Load engine from a saved config JSON."""
        with open(config_path) as f:
            cfg = json.load(f)
        strategy = DeploymentStrategy(cfg["strategy"])
        return cls(
            strategy=strategy,
            config_overrides={
                "aggregation": cfg.get("aggregation"),
                "calibration": cfg.get("calibration"),
                "normalization": NormMethod(cfg.get("normalization", "passthrough")),
                "block_threshold": cfg.get("block_threshold"),
                "flag_threshold": cfg.get("flag_threshold"),
                "adaptive": cfg.get("adaptive", False),
                "recalibrate": cfg.get("recalibrate", False),
                "drift_escalation": cfg.get("drift_escalation", True),
                "drift_tighten_factor": cfg.get("drift_tighten_factor", 0.8),
            },
        )

