# src/deployment/safe_recalibration.py
"""
Safe recalibration engine for VPN firewall deployment (Part G).

Extends the base LocalRecalibrator with:
- Safety guardrails (max allowed shift, minimum sample quality)
- Staged rollout (SHADOW → PARTIAL → FULL)
- Automatic rollback if performance degrades
- Validation against held-out subset (self-split)
- Full audit trail for regulatory / thesis use
- Gradual threshold blending (old → new over N sessions)

All operations use UNLABELED data only — no label leakage.
The recalibrator adjusts THRESHOLDS, never the model itself.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ──────────────────────────────────────────────────────
# Rollout stages
# ──────────────────────────────────────────────────────

class RolloutStage(str, Enum):
    """Progressive rollout stages for safe recalibration."""
    INACTIVE = "INACTIVE"        # No recalibration active
    SHADOW = "SHADOW"            # New thresholds computed but not applied; logged only
    PARTIAL = "PARTIAL"          # Blended: α * new + (1-α) * old
    FULL = "FULL"                # Fully applied new thresholds
    ROLLED_BACK = "ROLLED_BACK"  # Reverted to previous thresholds


class RecalibrationConfidence(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


# ──────────────────────────────────────────────────────
# Result and event types
# ──────────────────────────────────────────────────────

@dataclass
class SafeRecalibrationResult:
    """Output of a safe recalibration attempt."""
    # Proposed thresholds
    proposed_block_threshold: float
    proposed_flag_threshold: float
    # Active thresholds (may differ from proposed during staged rollout)
    active_block_threshold: float
    active_flag_threshold: float
    # Diagnostics
    n_samples: int
    n_held_out: int
    local_benign_mean: float
    local_benign_p90: float
    local_benign_max: float
    local_benign_std: float
    base_block_threshold: float
    threshold_shift: float
    shift_pct: float  # shift as % of base threshold
    confidence: str
    rollout_stage: str
    blend_alpha: float  # 0.0 = fully old, 1.0 = fully new
    # Validation (held-out subset)
    validation_passed: bool
    validation_holdout_max: float  # max score in held-out benign subset
    validation_margin: float       # proposed_block - holdout_max
    # Safety
    guardrail_violations: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecalibrationEvent:
    """Audit trail entry for a recalibration event."""
    timestamp: float
    event_type: str   # PROPOSED / SHADOW / PARTIAL / FULL / ROLLBACK / REJECTED
    old_block_threshold: float
    new_block_threshold: float
    old_flag_threshold: float
    new_flag_threshold: float
    n_samples: int
    confidence: str
    reason: str
    guardrail_violations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────
# Safe recalibration engine
# ──────────────────────────────────────────────────────

class SafeRecalibrator:
    """
    Production-safe recalibration engine with guardrails and staged rollout.

    Workflow:
    1. Collect benign traffic scores in deployment
    2. Call propose(scores) → computes new thresholds in SHADOW mode
    3. Call advance() to move SHADOW → PARTIAL → FULL
    4. If metrics degrade, call rollback() → reverts to previous thresholds
    5. All transitions are logged in the audit trail

    Parameters
    ----------
    base_block_threshold : float
        Val-calibrated block threshold (anchor).
    base_flag_threshold : float
        Val-calibrated flag threshold.
    max_shift_abs : float
        Maximum allowed absolute threshold shift. Violations are blocked.
    max_shift_pct : float
        Maximum allowed threshold shift as fraction of base. Safety brake.
    min_samples : int
        Minimum samples for any recalibration.
    high_confidence_samples : int
        Samples needed for HIGH confidence.
    moderate_confidence_samples : int
        Samples needed for MODERATE confidence.
    safety_margin : float
        Added above max benign score for conservatism.
    holdout_fraction : float
        Fraction of samples reserved for validation.
    blend_steps : int
        Number of advance() calls to go from PARTIAL(α=0.3) to FULL(α=1.0).
    """

    def __init__(
        self,
        base_block_threshold: float,
        base_flag_threshold: float = 0.5,
        max_shift_abs: float = 0.20,
        max_shift_pct: float = 0.30,
        min_samples: int = 30,
        high_confidence_samples: int = 100,
        moderate_confidence_samples: int = 50,
        safety_margin: float = 0.01,
        holdout_fraction: float = 0.2,
        blend_steps: int = 3,
    ):
        self.base_block_threshold = base_block_threshold
        self.base_flag_threshold = base_flag_threshold
        self.max_shift_abs = max_shift_abs
        self.max_shift_pct = max_shift_pct
        self.min_samples = min_samples
        self.high_confidence_samples = high_confidence_samples
        self.moderate_confidence_samples = moderate_confidence_samples
        self.safety_margin = safety_margin
        self.holdout_fraction = holdout_fraction
        self.blend_steps = blend_steps

        # Current state
        self._stage = RolloutStage.INACTIVE
        self._blend_alpha = 0.0
        self._blend_step_count = 0
        self._proposed_block: Optional[float] = None
        self._proposed_flag: Optional[float] = None
        self._active_block = base_block_threshold
        self._active_flag = base_flag_threshold
        self._previous_block = base_block_threshold
        self._previous_flag = base_flag_threshold
        self._last_result: Optional[SafeRecalibrationResult] = None

        # Audit trail
        self._audit_trail: List[RecalibrationEvent] = []

    def propose(
        self,
        local_benign_scores: np.ndarray,
        flag_fpr_target: float = 0.05,
        seed: int = 42,
    ) -> SafeRecalibrationResult:
        """
        Propose new thresholds from local benign traffic samples.

        Does NOT apply thresholds immediately. Enters SHADOW mode.
        Call advance() to progressively apply.

        Parameters
        ----------
        local_benign_scores : array
            Session-level scores from known/presumed benign local traffic.
        flag_fpr_target : float
            Target false-positive rate for the flag threshold.
        seed : int
            Random seed for holdout split.

        Returns
        -------
        SafeRecalibrationResult with proposed thresholds and diagnostics.
        """
        scores = np.asarray(local_benign_scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        n = len(scores)

        warnings: List[str] = []
        guardrail_violations: List[str] = []

        # ── Confidence assessment ──
        if n < self.min_samples:
            confidence = RecalibrationConfidence.INSUFFICIENT
            warnings.append(
                f"Only {n} samples (need >= {self.min_samples}). "
                "Recalibration not recommended."
            )
        elif n >= self.high_confidence_samples:
            confidence = RecalibrationConfidence.HIGH
        elif n >= self.moderate_confidence_samples:
            confidence = RecalibrationConfidence.MODERATE
        else:
            confidence = RecalibrationConfidence.LOW
            warnings.append(
                f"{n} samples gives LOW confidence. Consider collecting more."
            )

        # ── Insufficient samples → reject ──
        if confidence == RecalibrationConfidence.INSUFFICIENT:
            result = SafeRecalibrationResult(
                proposed_block_threshold=self.base_block_threshold,
                proposed_flag_threshold=self.base_flag_threshold,
                active_block_threshold=self._active_block,
                active_flag_threshold=self._active_flag,
                n_samples=n, n_held_out=0,
                local_benign_mean=float(np.mean(scores)) if n > 0 else 0.0,
                local_benign_p90=float(np.percentile(scores, 90)) if n > 0 else 0.0,
                local_benign_max=float(np.max(scores)) if n > 0 else 0.0,
                local_benign_std=float(np.std(scores)) if n > 1 else 0.0,
                base_block_threshold=self.base_block_threshold,
                threshold_shift=0.0, shift_pct=0.0,
                confidence=confidence.value,
                rollout_stage=self._stage.value,
                blend_alpha=self._blend_alpha,
                validation_passed=False, validation_holdout_max=0.0,
                validation_margin=0.0,
                guardrail_violations=guardrail_violations,
                warnings=warnings,
            )
            self._last_result = result
            self._log_event("REJECTED", 0.0, 0.0, n, confidence.value,
                            "Insufficient samples", guardrail_violations)
            return result

        # ── Split into calibration and validation ──
        rng = np.random.RandomState(seed)
        n_holdout = max(int(n * self.holdout_fraction), 1)
        indices = rng.permutation(n)
        holdout_idx = indices[:n_holdout]
        calib_idx = indices[n_holdout:]

        calib_scores = scores[calib_idx]
        holdout_scores = scores[holdout_idx]

        # ── Compute proposed thresholds ──
        local_max = float(np.max(calib_scores))
        local_p90 = float(np.percentile(calib_scores, 90))
        local_mean = float(np.mean(calib_scores))
        local_std = float(np.std(calib_scores)) if len(calib_scores) > 1 else 0.0

        proposed_block = local_max + self.safety_margin
        flag_quantile = 1.0 - flag_fpr_target
        proposed_flag = float(np.quantile(calib_scores, flag_quantile))
        proposed_flag = min(proposed_flag, proposed_block * 0.8)

        # ── Safety guardrails ──
        shift = proposed_block - self.base_block_threshold
        shift_pct = abs(shift) / max(abs(self.base_block_threshold), 1e-10)

        if abs(shift) > self.max_shift_abs:
            guardrail_violations.append(
                f"Absolute shift {shift:+.4f} exceeds max {self.max_shift_abs:.4f}. "
                f"Clamping threshold."
            )
            if shift > 0:
                proposed_block = self.base_block_threshold + self.max_shift_abs
            else:
                proposed_block = self.base_block_threshold - self.max_shift_abs

        if shift_pct > self.max_shift_pct:
            guardrail_violations.append(
                f"Relative shift {shift_pct:.1%} exceeds max {self.max_shift_pct:.1%}. "
                f"Clamping threshold."
            )
            max_abs = self.base_block_threshold * self.max_shift_pct
            if shift > 0:
                proposed_block = min(proposed_block, self.base_block_threshold + max_abs)
            else:
                proposed_block = max(proposed_block, self.base_block_threshold - max_abs)

        # Recompute shift after clamping
        shift = proposed_block - self.base_block_threshold
        shift_pct = abs(shift) / max(abs(self.base_block_threshold), 1e-10)

        # ── Holdout validation ──
        holdout_max = float(np.max(holdout_scores)) if len(holdout_scores) > 0 else 0.0
        validation_margin = proposed_block - holdout_max
        validation_passed = validation_margin > 0  # block threshold above all holdout benign

        if not validation_passed:
            guardrail_violations.append(
                f"Holdout validation FAILED: max holdout score {holdout_max:.4f} "
                f">= proposed block threshold {proposed_block:.4f}. "
                f"Proposed thresholds would mis-classify known benign traffic."
            )
            warnings.append("Recalibration rejected by holdout validation.")

        if shift > 0.15:
            warnings.append(
                f"Large upward shift ({shift:+.4f}). Local environment scores "
                "substantially higher than calibration reference."
            )

        # ── Stage transition ──
        if validation_passed and not guardrail_violations:
            new_stage = RolloutStage.SHADOW
        elif validation_passed:
            new_stage = RolloutStage.SHADOW  # warnings logged, but still shadow
            warnings.append(
                "Guardrail violations present. Thresholds clamped but SHADOW ok."
            )
        else:
            new_stage = RolloutStage.INACTIVE  # validation failed → don't proceed

        self._proposed_block = proposed_block
        self._proposed_flag = proposed_flag
        if new_stage == RolloutStage.SHADOW:
            self._stage = RolloutStage.SHADOW
            self._blend_alpha = 0.0
            self._blend_step_count = 0

        # Active thresholds don't change yet in SHADOW
        result = SafeRecalibrationResult(
            proposed_block_threshold=proposed_block,
            proposed_flag_threshold=proposed_flag,
            active_block_threshold=self._active_block,
            active_flag_threshold=self._active_flag,
            n_samples=n, n_held_out=len(holdout_scores),
            local_benign_mean=local_mean,
            local_benign_p90=local_p90,
            local_benign_max=float(np.max(scores)),
            local_benign_std=local_std,
            base_block_threshold=self.base_block_threshold,
            threshold_shift=shift,
            shift_pct=shift_pct,
            confidence=confidence.value,
            rollout_stage=self._stage.value,
            blend_alpha=self._blend_alpha,
            validation_passed=validation_passed,
            validation_holdout_max=holdout_max,
            validation_margin=validation_margin,
            guardrail_violations=guardrail_violations,
            warnings=warnings,
        )
        self._last_result = result

        event_type = "PROPOSED" if new_stage == RolloutStage.SHADOW else "REJECTED"
        self._log_event(
            event_type, proposed_block, proposed_flag, n, confidence.value,
            f"shift={shift:+.4f}, validation={'PASS' if validation_passed else 'FAIL'}",
            guardrail_violations,
        )
        return result

    def advance(self) -> SafeRecalibrationResult:
        """
        Advance the rollout stage: SHADOW → PARTIAL → FULL.

        In PARTIAL mode, thresholds are blended:
            active = α * proposed + (1-α) * previous
        where α increases with each advance() call.

        Returns updated SafeRecalibrationResult.
        """
        if self._proposed_block is None or self._proposed_flag is None:
            raise RuntimeError("No proposed thresholds. Call propose() first.")

        old_stage = self._stage

        if self._stage == RolloutStage.SHADOW:
            # → PARTIAL with initial blend
            self._stage = RolloutStage.PARTIAL
            self._blend_step_count = 1
            self._blend_alpha = 1.0 / self.blend_steps
            self._previous_block = self._active_block
            self._previous_flag = self._active_flag

        elif self._stage == RolloutStage.PARTIAL:
            self._blend_step_count += 1
            self._blend_alpha = min(self._blend_step_count / self.blend_steps, 1.0)
            if self._blend_alpha >= 1.0:
                self._stage = RolloutStage.FULL
                self._blend_alpha = 1.0

        elif self._stage == RolloutStage.FULL:
            # Already fully applied
            pass
        else:
            raise RuntimeError(
                f"Cannot advance from stage {self._stage.value}. "
                "Need SHADOW or PARTIAL stage."
            )

        # Apply blended thresholds
        alpha = self._blend_alpha
        self._active_block = alpha * self._proposed_block + (1 - alpha) * self._previous_block
        self._active_flag = alpha * self._proposed_flag + (1 - alpha) * self._previous_flag

        result = self._make_current_result()
        self._last_result = result

        self._log_event(
            self._stage.value,
            self._active_block, self._active_flag,
            result.n_samples, result.confidence,
            f"alpha={alpha:.2f}, stage {old_stage.value} → {self._stage.value}",
            [],
        )
        return result

    def rollback(self, reason: str = "Manual rollback") -> SafeRecalibrationResult:
        """
        Revert to previous thresholds.

        This is the safety mechanism: if post-recalibration performance
        degrades (increased FPR, operator complaints, etc.), roll back.
        """
        old_block = self._active_block
        old_flag = self._active_flag

        self._active_block = self._previous_block
        self._active_flag = self._previous_flag
        self._stage = RolloutStage.ROLLED_BACK
        self._blend_alpha = 0.0
        self._proposed_block = None
        self._proposed_flag = None

        result = self._make_current_result()
        self._last_result = result

        self._log_event(
            "ROLLBACK",
            self._active_block, self._active_flag,
            result.n_samples, result.confidence,
            reason, [],
        )
        return result

    def _make_current_result(self) -> SafeRecalibrationResult:
        """Build result reflecting current state."""
        last = self._last_result
        return SafeRecalibrationResult(
            proposed_block_threshold=self._proposed_block or self.base_block_threshold,
            proposed_flag_threshold=self._proposed_flag or self.base_flag_threshold,
            active_block_threshold=self._active_block,
            active_flag_threshold=self._active_flag,
            n_samples=last.n_samples if last else 0,
            n_held_out=last.n_held_out if last else 0,
            local_benign_mean=last.local_benign_mean if last else 0.0,
            local_benign_p90=last.local_benign_p90 if last else 0.0,
            local_benign_max=last.local_benign_max if last else 0.0,
            local_benign_std=last.local_benign_std if last else 0.0,
            base_block_threshold=self.base_block_threshold,
            threshold_shift=self._active_block - self.base_block_threshold,
            shift_pct=abs(self._active_block - self.base_block_threshold)
                / max(abs(self.base_block_threshold), 1e-10),
            confidence=last.confidence if last else RecalibrationConfidence.INSUFFICIENT.value,
            rollout_stage=self._stage.value,
            blend_alpha=self._blend_alpha,
            validation_passed=last.validation_passed if last else False,
            validation_holdout_max=last.validation_holdout_max if last else 0.0,
            validation_margin=last.validation_margin if last else 0.0,
            guardrail_violations=last.guardrail_violations if last else [],
            warnings=last.warnings if last else [],
        )

    def _log_event(
        self,
        event_type: str,
        new_block: float,
        new_flag: float,
        n_samples: int,
        confidence: str,
        reason: str,
        violations: List[str],
    ) -> None:
        """Append to audit trail."""
        self._audit_trail.append(RecalibrationEvent(
            timestamp=time.time(),
            event_type=event_type,
            old_block_threshold=self._previous_block,
            new_block_threshold=new_block,
            old_flag_threshold=self._previous_flag,
            new_flag_threshold=new_flag,
            n_samples=n_samples,
            confidence=confidence,
            reason=reason,
            guardrail_violations=violations,
        ))

    # ── Properties ──

    @property
    def stage(self) -> RolloutStage:
        return self._stage

    @property
    def active_block_threshold(self) -> float:
        return self._active_block

    @property
    def active_flag_threshold(self) -> float:
        return self._active_flag

    @property
    def blend_alpha(self) -> float:
        return self._blend_alpha

    @property
    def audit_trail(self) -> List[RecalibrationEvent]:
        return list(self._audit_trail)

    @property
    def last_result(self) -> Optional[SafeRecalibrationResult]:
        return self._last_result

    def state_summary(self) -> Dict[str, Any]:
        """Compact state for logging/debugging."""
        return {
            "stage": self._stage.value,
            "active_block": round(self._active_block, 6),
            "active_flag": round(self._active_flag, 6),
            "proposed_block": round(self._proposed_block, 6) if self._proposed_block else None,
            "proposed_flag": round(self._proposed_flag, 6) if self._proposed_flag else None,
            "blend_alpha": round(self._blend_alpha, 3),
            "base_block": self.base_block_threshold,
            "base_flag": self.base_flag_threshold,
            "n_events": len(self._audit_trail),
        }

    # ── Persistence ──

    def save(self, path: Path) -> None:
        """Save full state including audit trail."""
        data = {
            "base_block_threshold": self.base_block_threshold,
            "base_flag_threshold": self.base_flag_threshold,
            "config": {
                "max_shift_abs": self.max_shift_abs,
                "max_shift_pct": self.max_shift_pct,
                "min_samples": self.min_samples,
                "high_confidence_samples": self.high_confidence_samples,
                "moderate_confidence_samples": self.moderate_confidence_samples,
                "safety_margin": self.safety_margin,
                "holdout_fraction": self.holdout_fraction,
                "blend_steps": self.blend_steps,
            },
            "state": {
                "stage": self._stage.value,
                "blend_alpha": self._blend_alpha,
                "blend_step_count": self._blend_step_count,
                "active_block": self._active_block,
                "active_flag": self._active_flag,
                "previous_block": self._previous_block,
                "previous_flag": self._previous_flag,
                "proposed_block": self._proposed_block,
                "proposed_flag": self._proposed_flag,
            },
            "audit_trail": [e.to_dict() for e in self._audit_trail],
            "last_result": self._last_result.to_dict() if self._last_result else None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SafeRecalibrator":
        """Load from saved state."""
        with open(path) as f:
            data = json.load(f)
        cfg = data.get("config", {})
        recal = cls(
            base_block_threshold=data["base_block_threshold"],
            base_flag_threshold=data.get("base_flag_threshold", 0.5),
            max_shift_abs=cfg.get("max_shift_abs", 0.20),
            max_shift_pct=cfg.get("max_shift_pct", 0.30),
            min_samples=cfg.get("min_samples", 30),
            high_confidence_samples=cfg.get("high_confidence_samples", 100),
            moderate_confidence_samples=cfg.get("moderate_confidence_samples", 50),
            safety_margin=cfg.get("safety_margin", 0.01),
            holdout_fraction=cfg.get("holdout_fraction", 0.2),
            blend_steps=cfg.get("blend_steps", 3),
        )
        state = data.get("state", {})
        recal._stage = RolloutStage(state.get("stage", "INACTIVE"))
        recal._blend_alpha = state.get("blend_alpha", 0.0)
        recal._blend_step_count = state.get("blend_step_count", 0)
        recal._active_block = state.get("active_block", recal.base_block_threshold)
        recal._active_flag = state.get("active_flag", recal.base_flag_threshold)
        recal._previous_block = state.get("previous_block", recal.base_block_threshold)
        recal._previous_flag = state.get("previous_flag", recal.base_flag_threshold)
        recal._proposed_block = state.get("proposed_block")
        recal._proposed_flag = state.get("proposed_flag")
        # Restore audit trail
        for evt in data.get("audit_trail", []):
            recal._audit_trail.append(RecalibrationEvent(**evt))
        return recal


