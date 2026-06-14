# demo_firewall/policy.py
"""
Stage 4 & 5 — Session aggregation, threshold selection, and decision logic.

Aggregates flow-level predictions into session-level scores,
applies deployment-mode-specific thresholds, and returns
structured firewall decisions with confidence metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging import setup_logger

from demo_firewall.config import (
    DeploymentMode,
    MODE_CONFIGS,
    ThresholdConfig,
)
from demo_firewall.errors import (
    ThresholdLeakageError,
    InsufficientDataError,
)

logger = setup_logger(name="firewall.policy")


# ──────────────────────────────────────────────────────
# Decision types
# ──────────────────────────────────────────────────────

class Decision(str, Enum):
    """
    Firewall action decision.

    Legacy two-tier values (BLOCK / FLAG / ALLOW) are preserved for
    backward compatibility with existing callers.

    The open-set three-tier policy uses FirewallAction from
    demo_firewall.open_set_policy instead.  Mapping:
        ALLOW  → PASS
        FLAG   → FLAG_REVIEW
        BLOCK  → SIMULATED_BLOCK  (simulation only — no real packet blocking)
    """
    BLOCK = "BLOCK"                    # Legacy: high-confidence VPN
    FLAG = "FLAG"                      # Legacy: elevated score, flag for review
    ALLOW = "ALLOW"                    # Legacy: below threshold, allow

    # Open-set three-tier aliases (preferred for new callers)
    SIMULATED_BLOCK = "SIMULATED_BLOCK"   # simulation only
    FLAG_REVIEW = "FLAG_REVIEW"
    PASS = "PASS"


@dataclass
class FlowDecision:
    """Decision result for a single flow."""
    flow_id: str
    capture_id: str
    probability: float
    decision: Decision
    threshold_used: float
    calibration_method: str
    confidence_margin: float


@dataclass
class SessionDecision:
    """Decision result for a session (capture)."""
    capture_id: str
    session_score: float
    decision: Decision
    block_threshold: float
    flag_threshold: float
    aggregation_rule: str
    n_flows: int
    n_flows_above_block: int
    n_flows_above_flag: int
    confidence_margin: float
    deployment_mode: str
    flow_decisions: List[FlowDecision]


# ──────────────────────────────────────────────────────
# Session aggregation functions
# ──────────────────────────────────────────────────────

def _p90_aggregation(probs: np.ndarray) -> float:
    """90th percentile aggregation — conservative, used for STRICT mode."""
    if len(probs) == 0:
        return 0.0
    return float(np.percentile(probs, 90))


def _weighted_top5_mean(probs: np.ndarray, k: int = 5) -> float:
    """Weighted top-k mean — balanced, used for BALANCED mode."""
    vals = np.sort(probs)[::-1][:k]
    if len(vals) == 0:
        return 0.0
    weights = np.array([0.40, 0.25, 0.15, 0.10, 0.10])[:len(vals)]
    weights = weights / weights.sum()
    return float(np.sum(vals * weights))


def _mean_aggregation(probs: np.ndarray) -> float:
    """Simple mean — used for RESEARCH mode."""
    if len(probs) == 0:
        return 0.0
    return float(np.mean(probs))


_AGGREGATORS = {
    "p90": _p90_aggregation,
    "weighted_top5_mean": _weighted_top5_mean,
    "mean": _mean_aggregation,
}


# ──────────────────────────────────────────────────────
# Threshold computation
# ──────────────────────────────────────────────────────

def compute_threshold_from_validation(
    val_session_scores: np.ndarray,
    val_labels: np.ndarray,
    target_fpr: float = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the blocking threshold from validation data.

    For STRICT mode (target_fpr=0.0):
        threshold = max(score among benign sessions)
        Guarantees block_FPR = 0 on validation data.

    Parameters
    ----------
    val_session_scores : array
        Session-level scores for validation sessions.
    val_labels : array
        Binary labels (0=benign, 1=VPN) for validation sessions.
    target_fpr : float
        Target false positive rate.

    Returns
    -------
    threshold : float
    metadata : dict
        Provenance information about the threshold computation.
    """
    val_session_scores = np.asarray(val_session_scores, dtype=float)
    val_labels = np.asarray(val_labels, dtype=int)

    benign_scores = val_session_scores[val_labels == 0]
    vpn_scores = val_session_scores[val_labels == 1]

    if len(benign_scores) == 0:
        raise ThresholdLeakageError(
            "No benign sessions in validation set. "
            "Cannot compute a safe threshold."
        )

    if target_fpr == 0.0:
        # STRICT: threshold = max(benign score)
        # Any session scoring above this is declared VPN
        threshold = float(np.max(benign_scores))
    else:
        # Use quantile method
        threshold = float(np.quantile(benign_scores, 1.0 - target_fpr))

    metadata = {
        "threshold": threshold,
        "target_fpr": target_fpr,
        "n_benign_sessions": int(len(benign_scores)),
        "n_vpn_sessions": int(len(vpn_scores)),
        "max_benign_score": float(np.max(benign_scores)),
        "min_vpn_score": float(np.min(vpn_scores)) if len(vpn_scores) > 0 else None,
        "source_split": "val",
        "computed_on_benign_only": True,
    }

    return threshold, metadata


# ──────────────────────────────────────────────────────
# Firewall policy engine
# ──────────────────────────────────────────────────────

class FirewallPolicy:
    """
    Applies deployment policy to flow/session predictions.

    Manages:
    - Session-level aggregation (p90 / weighted_top5_mean / mean)
    - Threshold application (block / flag / allow)
    - Safety validation
    """

    def __init__(
        self,
        mode: DeploymentMode = DeploymentMode.STRICT,
        block_threshold: Optional[float] = None,
        flag_threshold: Optional[float] = None,
        threshold_config: Optional[ThresholdConfig] = None,
    ):
        """
        Parameters
        ----------
        mode : DeploymentMode
            Operating mode.
        block_threshold : float or None
            Override block threshold. If None, must be set via
            calibrate_thresholds() before making decisions.
        flag_threshold : float or None
            Override flag threshold (lower than block).
            If None, defaults to block_threshold * 0.7.
        threshold_config : ThresholdConfig or None
            Full threshold provenance record.
        """
        self.mode = mode
        self.mode_config = MODE_CONFIGS[mode]

        self._block_threshold = block_threshold
        self._flag_threshold = flag_threshold
        self._threshold_config = threshold_config
        self._thresholds_calibrated = block_threshold is not None

        # Get aggregator
        rule = self.mode_config.aggregation_rule
        if rule not in _AGGREGATORS:
            raise ValueError(f"Unknown aggregation rule: {rule}")
        self._aggregator = _AGGREGATORS[rule]
        self._aggregation_rule = rule

    def calibrate_thresholds(
        self,
        val_preds: pd.DataFrame,
        prob_col: str = "prob_cal",
        label_col: str = "label",
        session_col: str = "capture_id",
    ) -> ThresholdConfig:
        """
        Compute thresholds from validation predictions.

        Parameters
        ----------
        val_preds : pd.DataFrame
            Flow-level validation predictions with prob_col and label_col.
        prob_col : str
            Probability column name.
        label_col : str
            Label column name.
        session_col : str
            Session identifier column name.

        Returns
        -------
        ThresholdConfig
            Computed threshold with provenance metadata.
        """
        # Aggregate to session level
        session_scores = (
            val_preds.groupby(session_col)[prob_col]
            .apply(lambda x: self._aggregator(x.values))
            .reset_index()
        )
        session_labels = val_preds.groupby(session_col)[label_col].max().reset_index()
        session_df = session_scores.merge(session_labels, on=session_col)

        # Validate no label contamination in benign pool
        if self.mode_config.enforce_zero_block_fpr:
            benign_mask = session_df[label_col] == 0
            if not benign_mask.any():
                raise ThresholdLeakageError(
                    "Validation set contains no benign sessions. "
                    "Cannot compute zero-FPR threshold."
                )

        # Compute block threshold
        block_threshold, meta = compute_threshold_from_validation(
            val_session_scores=session_df[prob_col].values,
            val_labels=session_df[label_col].values,
            target_fpr=self.mode_config.target_fpr,
        )

        # Flag threshold: more lenient (0.1% FPR or 70% of block threshold)
        if self.mode == DeploymentMode.STRICT:
            flag_threshold, _ = compute_threshold_from_validation(
                val_session_scores=session_df[prob_col].values,
                val_labels=session_df[label_col].values,
                target_fpr=0.001,
            )
            flag_threshold = min(flag_threshold, block_threshold * 0.7)
        elif self.mode == DeploymentMode.BALANCED:
            flag_threshold = block_threshold * 0.5
        else:
            flag_threshold = 0.5  # RESEARCH mode

        self._block_threshold = block_threshold
        self._flag_threshold = flag_threshold
        self._thresholds_calibrated = True

        self._threshold_config = ThresholdConfig(
            block_threshold=block_threshold,
            flag_threshold=flag_threshold,
            source_split="val",
            aggregation_rule=self._aggregation_rule,
            calibration_method="isotonic",
            computed_on_benign_only=True,
        )

        logger.info(
            f"Thresholds calibrated ({self.mode.value}): "
            f"block={block_threshold:.6f}, flag={flag_threshold:.6f}"
        )

        return self._threshold_config

    def predict_session(
        self,
        flow_preds: pd.DataFrame,
        prob_col: str = "prob_cal",
        label_col: str = "label",
        session_col: str = "capture_id",
    ) -> SessionDecision:
        """
        Make a session-level firewall decision.

        Parameters
        ----------
        flow_preds : pd.DataFrame
            Flow-level predictions for ONE session/capture.

        Returns
        -------
        SessionDecision
            Structured decision with full metadata.
        """
        if not self._thresholds_calibrated:
            raise RuntimeError(
                "Thresholds not calibrated. Call calibrate_thresholds() first "
                "or provide explicit thresholds at construction."
            )

        if len(flow_preds) == 0:
            raise InsufficientDataError("No flow predictions to aggregate.")

        capture_id = str(flow_preds[session_col].iloc[0])
        probs = flow_preds[prob_col].values.astype(float)

        # Aggregate
        session_score = self._aggregator(probs)

        # Decide
        if self.mode == DeploymentMode.RESEARCH:
            decision = Decision.ALLOW  # No thresholding in research mode
        elif session_score > self._block_threshold:
            decision = Decision.BLOCK
        elif session_score > self._flag_threshold:
            decision = Decision.FLAG
        else:
            decision = Decision.ALLOW

        # Confidence margin: distance from nearest threshold
        if decision == Decision.BLOCK:
            confidence_margin = session_score - self._block_threshold
        elif decision == Decision.FLAG:
            confidence_margin = min(
                session_score - self._flag_threshold,
                self._block_threshold - session_score,
            )
        else:
            confidence_margin = self._flag_threshold - session_score

        # Per-flow decisions
        flow_decisions = []
        for _, row in flow_preds.iterrows():
            p = float(row[prob_col])
            if p > self._block_threshold:
                fd = Decision.BLOCK
            elif p > self._flag_threshold:
                fd = Decision.FLAG
            else:
                fd = Decision.ALLOW

            flow_decisions.append(FlowDecision(
                flow_id=str(row.get("flow_id", "")),
                capture_id=capture_id,
                probability=p,
                decision=fd,
                threshold_used=self._block_threshold,
                calibration_method=str(row.get("calibration_method", "unknown")),
                confidence_margin=abs(p - 0.5) * 2.0,
            ))

        return SessionDecision(
            capture_id=capture_id,
            session_score=session_score,
            decision=decision,
            block_threshold=self._block_threshold,
            flag_threshold=self._flag_threshold,
            aggregation_rule=self._aggregation_rule,
            n_flows=len(probs),
            n_flows_above_block=int(np.sum(probs > self._block_threshold)),
            n_flows_above_flag=int(np.sum(probs > self._flag_threshold)),
            confidence_margin=confidence_margin,
            deployment_mode=str(self.mode.value),
            flow_decisions=flow_decisions,
        )

    def predict_sessions_batch(
        self,
        flow_preds: pd.DataFrame,
        prob_col: str = "prob_cal",
        label_col: str = "label",
        session_col: str = "capture_id",
    ) -> List[SessionDecision]:
        """
        Make decisions for multiple sessions at once.

        Parameters
        ----------
        flow_preds : pd.DataFrame
            Flow-level predictions for multiple sessions.

        Returns
        -------
        List[SessionDecision]
        """
        decisions = []
        for cid, group in flow_preds.groupby(session_col):
            try:
                dec = self.predict_session(
                    group,
                    prob_col=prob_col,
                    label_col=label_col,
                    session_col=session_col,
                )
                decisions.append(dec)
            except InsufficientDataError:
                logger.warning(f"Skipping session '{cid}': no valid flows")
                continue

        return decisions

    @property
    def block_threshold(self) -> float:
        if self._block_threshold is None:
            raise RuntimeError("Threshold not set.")
        return self._block_threshold

    @property
    def flag_threshold(self) -> float:
        if self._flag_threshold is None:
            raise RuntimeError("Threshold not set.")
        return self._flag_threshold

    @property
    def threshold_config(self) -> Optional[ThresholdConfig]:
        return self._threshold_config

    def diagnostics(self) -> Dict[str, Any]:
        """Return policy diagnostics."""
        return {
            "mode": self.mode.value,
            "aggregation_rule": self._aggregation_rule,
            "block_threshold": self._block_threshold,
            "flag_threshold": self._flag_threshold,
            "thresholds_calibrated": self._thresholds_calibrated,
            "enforce_zero_block_fpr": self.mode_config.enforce_zero_block_fpr,
            "target_fpr": self.mode_config.target_fpr,
        }



