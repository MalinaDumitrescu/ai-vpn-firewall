# src/deployment/drift_monitor.py
"""
Domain-shift and score-drift monitor for VPN firewall deployment.

Detects when production traffic deviates from the calibration reference
distribution, enabling proactive threshold adjustment or alert escalation.

Operates on UNLABELED scores only — no label leakage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import ks_2samp


class DriftLevel(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    HIGH = "HIGH"


@dataclass
class DriftReport:
    """Result of a drift check."""
    level: DriftLevel
    ks_statistic: float
    ks_pvalue: float
    psi: float
    quantile_deltas: Dict[str, float]  # {p25, p50, p75, p90}
    reference_n: int
    current_n: int
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        return d


def _compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index."""
    eps = 1e-8
    edges = np.linspace(
        min(reference.min(), current.min()) - eps,
        max(reference.max(), current.max()) + eps,
        n_bins + 1,
    )
    ref_counts = np.histogram(reference, bins=edges)[0].astype(float) + eps
    cur_counts = np.histogram(current, bins=edges)[0].astype(float) + eps

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


class DriftMonitor:
    """
    Monitors score distribution drift against a reference distribution.

    Usage:
        monitor = DriftMonitor()
        monitor.fit(val_benign_scores)
        report = monitor.check(recent_scores)
    """

    def __init__(
        self,
        ks_warning_threshold: float = 0.05,  # KS p-value < this → WARNING
        ks_high_threshold: float = 0.001,     # KS p-value < this → HIGH
        psi_warning_threshold: float = 0.10,
        psi_high_threshold: float = 0.25,
        n_bins_psi: int = 10,
    ):
        self.ks_warning = ks_warning_threshold
        self.ks_high = ks_high_threshold
        self.psi_warning = psi_warning_threshold
        self.psi_high = psi_high_threshold
        self.n_bins_psi = n_bins_psi

        self._reference: Optional[np.ndarray] = None
        self._ref_quantiles: Dict[str, float] = {}
        self._history: List[DriftReport] = []

    def fit(self, reference_scores: np.ndarray) -> "DriftMonitor":
        """
        Set reference distribution from validation benign session scores.

        Parameters
        ----------
        reference_scores : array
            Session-level scores from benign validation sessions.
            These are the calibration reference.
        """
        self._reference = np.asarray(reference_scores, dtype=float)
        self._reference = self._reference[np.isfinite(self._reference)]
        if len(self._reference) < 5:
            raise ValueError("Reference distribution needs at least 5 scores.")

        self._ref_quantiles = {
            "p25": float(np.percentile(self._reference, 25)),
            "p50": float(np.percentile(self._reference, 50)),
            "p75": float(np.percentile(self._reference, 75)),
            "p90": float(np.percentile(self._reference, 90)),
        }
        return self

    def check(self, current_scores: np.ndarray) -> DriftReport:
        """
        Check current score batch against reference.

        Parameters
        ----------
        current_scores : array
            Recent session-level scores (all traffic, not just benign).
            In practice, use low-confidence scores as a proxy for benign.
        """
        if self._reference is None:
            raise RuntimeError("DriftMonitor not fitted. Call fit() first.")

        current = np.asarray(current_scores, dtype=float)
        current = current[np.isfinite(current)]
        if len(current) < 3:
            return DriftReport(
                level=DriftLevel.OK,
                ks_statistic=0.0, ks_pvalue=1.0, psi=0.0,
                quantile_deltas={}, reference_n=len(self._reference),
                current_n=len(current),
                recommendation="Insufficient data for drift check.",
            )

        # KS test
        ks_stat, ks_p = ks_2samp(self._reference, current)

        # PSI
        psi = _compute_psi(self._reference, current, self.n_bins_psi)

        # Quantile deltas
        cur_q = {
            "p25": float(np.percentile(current, 25)),
            "p50": float(np.percentile(current, 50)),
            "p75": float(np.percentile(current, 75)),
            "p90": float(np.percentile(current, 90)),
        }
        q_deltas = {k: cur_q[k] - self._ref_quantiles[k] for k in cur_q}

        # Determine level
        if ks_p < self.ks_high or psi > self.psi_high:
            level = DriftLevel.HIGH
            rec = ("Significant score distribution shift detected. "
                   "Switch to STRICT mode and consider local recalibration.")
        elif ks_p < self.ks_warning or psi > self.psi_warning:
            level = DriftLevel.WARNING
            rec = ("Moderate score distribution shift detected. "
                   "Monitor closely. Consider widening FLAG region.")
        else:
            level = DriftLevel.OK
            rec = "Score distribution within expected range. No action needed."

        report = DriftReport(
            level=level,
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_p),
            psi=float(psi),
            quantile_deltas=q_deltas,
            reference_n=len(self._reference),
            current_n=len(current),
            recommendation=rec,
        )
        self._history.append(report)
        return report

    @property
    def history(self) -> List[DriftReport]:
        return list(self._history)

    def save(self, path: Path) -> None:
        """Save reference distribution and config."""
        data = {
            "reference_scores": self._reference.tolist() if self._reference is not None else [],
            "reference_quantiles": self._ref_quantiles,
            "config": {
                "ks_warning": self.ks_warning,
                "ks_high": self.ks_high,
                "psi_warning": self.psi_warning,
                "psi_high": self.psi_high,
                "n_bins_psi": self.n_bins_psi,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DriftMonitor":
        """Load from saved artifacts."""
        with open(path) as f:
            data = json.load(f)
        cfg = data.get("config", {})
        monitor = cls(
            ks_warning_threshold=cfg.get("ks_warning", 0.05),
            ks_high_threshold=cfg.get("ks_high", 0.001),
            psi_warning_threshold=cfg.get("psi_warning", 0.10),
            psi_high_threshold=cfg.get("psi_high", 0.25),
            n_bins_psi=cfg.get("n_bins_psi", 10),
        )
        ref = data.get("reference_scores", [])
        if ref:
            monitor.fit(np.array(ref, dtype=float))
        return monitor


