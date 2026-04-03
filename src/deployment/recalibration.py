# src/deployment/recalibration.py
"""
Local recalibration support for new deployment environments.

Allows a firewall operator to collect local benign traffic samples,
derive environment-specific thresholds and normalization, and export
deployment-ready artifacts — WITHOUT retraining the detector.

This is critical because the ISCX FPR problem stems from score distribution
differences across environments, not classifier quality.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RecalibrationResult:
    """Output of local recalibration."""
    local_block_threshold: float
    local_flag_threshold: float
    n_local_samples: int
    local_benign_mean: float
    local_benign_p90: float
    local_benign_max: float
    base_threshold: float
    threshold_shift: float
    confidence: str  # "high" / "moderate" / "low"
    warnings: list

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LocalRecalibrator:
    """
    Derives local deployment thresholds from benign traffic samples.

    Workflow:
    1. Deploy the firewall with STRICT mode initially
    2. Collect scores from traffic that passes (presumed benign)
    3. Call recalibrate(local_benign_scores) to derive local thresholds
    4. Switch to locally-calibrated thresholds

    This does NOT retrain the model — only adjusts the decision boundary.

    Usage:
        recal = LocalRecalibrator(base_block_threshold=0.7447)
        result = recal.recalibrate(local_benign_scores)
        # Use result.local_block_threshold for deployment
    """

    def __init__(
        self,
        base_block_threshold: float,
        base_flag_threshold: float = 0.5,
        min_samples: int = 30,
        safety_margin: float = 0.01,
    ):
        self.base_block_threshold = base_block_threshold
        self.base_flag_threshold = base_flag_threshold
        self.min_samples = min_samples
        self.safety_margin = safety_margin
        self._last_result: Optional[RecalibrationResult] = None

    def recalibrate(
        self,
        local_benign_scores: np.ndarray,
        flag_fpr_target: float = 0.05,
    ) -> RecalibrationResult:
        """
        Compute local thresholds from benign traffic samples.

        Parameters
        ----------
        local_benign_scores : array
            Session-level scores from known-benign local traffic.
        flag_fpr_target : float
            Target FPR for the flag threshold (more lenient than block).

        Returns
        -------
        RecalibrationResult with local thresholds and diagnostics.
        """
        scores = np.asarray(local_benign_scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        n = len(scores)

        warnings = []
        if n < self.min_samples:
            warnings.append(
                f"Only {n} samples (need >= {self.min_samples}). "
                "Local thresholds may be unreliable."
            )
        if n < 10:
            warnings.append("Very few samples. Using base threshold as fallback.")
            self._last_result = RecalibrationResult(
                local_block_threshold=self.base_block_threshold,
                local_flag_threshold=self.base_flag_threshold,
                n_local_samples=n,
                local_benign_mean=float(np.mean(scores)) if n > 0 else 0.0,
                local_benign_p90=float(np.percentile(scores, 90)) if n > 0 else 0.0,
                local_benign_max=float(np.max(scores)) if n > 0 else 0.0,
                base_threshold=self.base_block_threshold,
                threshold_shift=0.0,
                confidence="low",
                warnings=warnings,
            )
            return self._last_result

        local_max = float(np.max(scores))
        local_p90 = float(np.percentile(scores, 90))
        local_mean = float(np.mean(scores))

        # Block threshold: max(benign) + margin (same logic as val calibration)
        local_block = local_max + self.safety_margin

        # Flag threshold: at the flag_fpr_target quantile
        flag_quantile = 1.0 - flag_fpr_target
        local_flag = float(np.quantile(scores, flag_quantile))

        # Ensure flag < block
        local_flag = min(local_flag, local_block * 0.8)

        # Confidence assessment
        fpr_resolution = 1.0 / max(n, 1)
        if n >= 100:
            confidence = "high"
        elif n >= 50:
            confidence = "moderate"
        else:
            confidence = "low"

        threshold_shift = local_block - self.base_block_threshold

        if abs(threshold_shift) > 0.3:
            warnings.append(
                f"Large threshold shift ({threshold_shift:+.4f}). "
                "Local environment may be very different from calibration."
            )

        self._last_result = RecalibrationResult(
            local_block_threshold=local_block,
            local_flag_threshold=local_flag,
            n_local_samples=n,
            local_benign_mean=local_mean,
            local_benign_p90=local_p90,
            local_benign_max=local_max,
            base_threshold=self.base_block_threshold,
            threshold_shift=threshold_shift,
            confidence=confidence,
            warnings=warnings,
        )
        return self._last_result

    @property
    def last_result(self) -> Optional[RecalibrationResult]:
        return self._last_result

    def save(self, path: Path) -> None:
        data = {
            "base_block_threshold": self.base_block_threshold,
            "base_flag_threshold": self.base_flag_threshold,
            "min_samples": self.min_samples,
            "safety_margin": self.safety_margin,
            "last_result": self._last_result.to_dict() if self._last_result else None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LocalRecalibrator":
        with open(path) as f:
            data = json.load(f)
        return cls(
            base_block_threshold=data["base_block_threshold"],
            base_flag_threshold=data.get("base_flag_threshold", 0.5),
            min_samples=data.get("min_samples", 30),
            safety_margin=data.get("safety_margin", 0.01),
        )

