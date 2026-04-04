# src/deployment/enhanced_drift_monitor.py
"""
Enhanced drift monitor for VPN firewall deployment (Part F).

Extends the base DriftMonitor with:
- Feature-level drift detection (per-feature KS tests)
- Rolling window drift tracking with trend detection
- Multi-signal assessment (score-level + feature-level)
- Alert severity escalation with history
- Exportable dashboard data for monitoring UIs

All operations use UNLABELED data only — no label leakage.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from scipy.stats import ks_2samp

from src.deployment.drift_monitor import DriftMonitor, DriftReport, DriftLevel, _compute_psi


# ──────────────────────────────────────────────────────
# Feature-level drift types
# ──────────────────────────────────────────────────────

@dataclass
class FeatureDriftResult:
    """Drift check result for a single feature."""
    feature_name: str
    ks_statistic: float
    ks_pvalue: float
    psi: float
    ref_mean: float
    ref_std: float
    cur_mean: float
    cur_std: float
    mean_shift: float  # absolute shift in means (normalized by ref std)
    level: str         # OK / WARNING / HIGH

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnhancedDriftReport:
    """
    Comprehensive drift report combining score-level and feature-level signals.
    """
    timestamp: float
    # Score-level
    score_drift: DriftReport
    # Feature-level
    feature_drifts: List[FeatureDriftResult]
    n_features_warning: int
    n_features_high: int
    worst_feature: Optional[str]
    worst_feature_ks: float
    # Composite assessment
    composite_level: str  # OK / WARNING / HIGH / CRITICAL
    composite_score: float  # 0..1 severity metric
    # Trend
    trend_direction: str  # STABLE / WORSENING / IMPROVING
    trend_slope: float    # positive = worsening
    # Recommendation
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['score_drift'] = self.score_drift.to_dict()
        d['feature_drifts'] = [fd.to_dict() for fd in self.feature_drifts]
        return d


class DriftTrend(str, Enum):
    STABLE = "STABLE"
    WORSENING = "WORSENING"
    IMPROVING = "IMPROVING"


# ──────────────────────────────────────────────────────
# Enhanced drift monitor
# ──────────────────────────────────────────────────────

class EnhancedDriftMonitor:
    """
    Production-grade drift monitor with multi-signal detection.

    Monitors both score distributions and individual feature distributions
    to detect drift before it impacts classification quality.

    Usage:
        monitor = EnhancedDriftMonitor(feature_names=['sz_coef_variation', ...])
        monitor.fit(ref_scores, ref_features)
        report = monitor.check(cur_scores, cur_features)

    Parameters
    ----------
    feature_names : list of str
        Names of features to monitor.
    window_size : int
        Number of recent checks to keep for trend detection.
    ks_warning_threshold : float
        KS p-value below this → WARNING.
    ks_high_threshold : float
        KS p-value below this → HIGH.
    psi_warning_threshold : float
        PSI above this → WARNING.
    psi_high_threshold : float
        PSI above this → HIGH.
    feature_ks_warning : float
        Per-feature KS statistic above this → feature WARNING.
    feature_ks_high : float
        Per-feature KS statistic above this → feature HIGH.
    critical_feature_pct : float
        If this fraction of features are HIGH, composite → CRITICAL.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        window_size: int = 20,
        ks_warning_threshold: float = 0.05,
        ks_high_threshold: float = 0.001,
        psi_warning_threshold: float = 0.10,
        psi_high_threshold: float = 0.25,
        feature_ks_warning: float = 0.15,
        feature_ks_high: float = 0.30,
        critical_feature_pct: float = 0.50,
        n_bins_psi: int = 10,
    ):
        self.feature_names = feature_names or []
        self.window_size = window_size
        self.n_bins_psi = n_bins_psi

        # Score-level thresholds
        self.ks_warning = ks_warning_threshold
        self.ks_high = ks_high_threshold
        self.psi_warning = psi_warning_threshold
        self.psi_high = psi_high_threshold

        # Feature-level thresholds
        self.feature_ks_warning = feature_ks_warning
        self.feature_ks_high = feature_ks_high
        self.critical_feature_pct = critical_feature_pct

        # Internal state
        self._score_monitor = DriftMonitor(
            ks_warning_threshold=ks_warning_threshold,
            ks_high_threshold=ks_high_threshold,
            psi_warning_threshold=psi_warning_threshold,
            psi_high_threshold=psi_high_threshold,
            n_bins_psi=n_bins_psi,
        )
        self._ref_features: Optional[np.ndarray] = None  # (n_ref, n_features)
        self._ref_feature_stats: Dict[str, Dict[str, float]] = {}
        self._history: Deque[EnhancedDriftReport] = deque(maxlen=window_size)
        self._composite_scores: Deque[float] = deque(maxlen=window_size)
        self._fitted = False
        self._alert_escalation_count = 0
        self._consecutive_warnings = 0

    def fit(
        self,
        reference_scores: np.ndarray,
        reference_features: Optional[np.ndarray] = None,
    ) -> "EnhancedDriftMonitor":
        """
        Set reference distributions from validation data.

        Parameters
        ----------
        reference_scores : array, shape (n_samples,)
            Session-level scores from validation benign sessions.
        reference_features : array, shape (n_samples, n_features), optional
            Feature matrix for the same reference sessions.
        """
        self._score_monitor.fit(reference_scores)

        if reference_features is not None:
            ref = np.asarray(reference_features, dtype=float)
            if ref.ndim == 1:
                ref = ref.reshape(-1, 1)
            self._ref_features = ref

            # Compute per-feature reference statistics
            n_feat = ref.shape[1]
            names = self.feature_names if len(self.feature_names) == n_feat \
                else [f"feature_{i}" for i in range(n_feat)]
            self.feature_names = names

            for i, name in enumerate(names):
                col = ref[:, i]
                col_clean = col[np.isfinite(col)]
                if len(col_clean) > 0:
                    self._ref_feature_stats[name] = {
                        "mean": float(np.mean(col_clean)),
                        "std": float(np.std(col_clean)) if len(col_clean) > 1 else 1.0,
                        "p25": float(np.percentile(col_clean, 25)),
                        "p50": float(np.percentile(col_clean, 50)),
                        "p75": float(np.percentile(col_clean, 75)),
                    }

        self._fitted = True
        return self

    def check(
        self,
        current_scores: np.ndarray,
        current_features: Optional[np.ndarray] = None,
    ) -> EnhancedDriftReport:
        """
        Run comprehensive drift check.

        Parameters
        ----------
        current_scores : array, shape (n_current,)
            Recent session-level scores.
        current_features : array, shape (n_current, n_features), optional
            Feature matrix for current sessions.

        Returns
        -------
        EnhancedDriftReport with composite assessment.
        """
        if not self._fitted:
            raise RuntimeError("EnhancedDriftMonitor not fitted. Call fit() first.")

        ts = time.time()

        # 1. Score-level drift
        score_report = self._score_monitor.check(current_scores)

        # 2. Feature-level drift
        feature_drifts = []
        if current_features is not None and self._ref_features is not None:
            cur = np.asarray(current_features, dtype=float)
            if cur.ndim == 1:
                cur = cur.reshape(-1, 1)

            n_feat = min(cur.shape[1], self._ref_features.shape[1])
            for i in range(n_feat):
                name = self.feature_names[i] if i < len(self.feature_names) \
                    else f"feature_{i}"
                ref_col = self._ref_features[:, i]
                cur_col = cur[:, i]

                ref_clean = ref_col[np.isfinite(ref_col)]
                cur_clean = cur_col[np.isfinite(cur_col)]

                if len(ref_clean) < 5 or len(cur_clean) < 3:
                    continue

                ks_stat, ks_p = ks_2samp(ref_clean, cur_clean)
                psi = _compute_psi(ref_clean, cur_clean, self.n_bins_psi)

                ref_mean = float(np.mean(ref_clean))
                ref_std = max(float(np.std(ref_clean)), 1e-10)
                cur_mean = float(np.mean(cur_clean))
                cur_std = float(np.std(cur_clean))
                mean_shift = abs(cur_mean - ref_mean) / ref_std

                # Feature-level severity
                if ks_stat >= self.feature_ks_high:
                    feat_level = DriftLevel.HIGH.value
                elif ks_stat >= self.feature_ks_warning:
                    feat_level = DriftLevel.WARNING.value
                else:
                    feat_level = DriftLevel.OK.value

                feature_drifts.append(FeatureDriftResult(
                    feature_name=name,
                    ks_statistic=float(ks_stat),
                    ks_pvalue=float(ks_p),
                    psi=float(psi),
                    ref_mean=ref_mean,
                    ref_std=ref_std,
                    cur_mean=cur_mean,
                    cur_std=cur_std,
                    mean_shift=float(mean_shift),
                    level=feat_level,
                ))

        # 3. Aggregate feature-level signals
        n_feat_warn = sum(1 for fd in feature_drifts if fd.level == DriftLevel.WARNING.value)
        n_feat_high = sum(1 for fd in feature_drifts if fd.level == DriftLevel.HIGH.value)
        worst_feat = None
        worst_feat_ks = 0.0
        if feature_drifts:
            worst = max(feature_drifts, key=lambda fd: fd.ks_statistic)
            worst_feat = worst.feature_name
            worst_feat_ks = worst.ks_statistic

        # 4. Compute composite severity
        composite_score = self._compute_composite_score(
            score_report, feature_drifts, n_feat_high
        )

        # 5. Determine composite level
        n_total_feat = len(feature_drifts) if feature_drifts else 1
        feat_high_pct = n_feat_high / max(n_total_feat, 1)

        if composite_score >= 0.8 or (score_report.level == DriftLevel.HIGH
                                       and feat_high_pct >= self.critical_feature_pct):
            composite_level = "CRITICAL"
        elif score_report.level == DriftLevel.HIGH or composite_score >= 0.5:
            composite_level = DriftLevel.HIGH.value
        elif score_report.level == DriftLevel.WARNING or composite_score >= 0.25:
            composite_level = DriftLevel.WARNING.value
        else:
            composite_level = DriftLevel.OK.value

        # 6. Trend detection
        trend_dir, trend_slope = self._detect_trend()

        # 7. Track escalation
        if composite_level in ("HIGH", "CRITICAL"):
            self._consecutive_warnings += 1
            if self._consecutive_warnings >= 3:
                self._alert_escalation_count += 1
        elif composite_level == "WARNING":
            self._consecutive_warnings += 1
        else:
            self._consecutive_warnings = 0

        # 8. Recommendation
        recommendation = self._generate_recommendation(
            composite_level, trend_dir, n_feat_high, worst_feat,
        )

        report = EnhancedDriftReport(
            timestamp=ts,
            score_drift=score_report,
            feature_drifts=feature_drifts,
            n_features_warning=n_feat_warn,
            n_features_high=n_feat_high,
            worst_feature=worst_feat,
            worst_feature_ks=worst_feat_ks,
            composite_level=composite_level,
            composite_score=round(composite_score, 4),
            trend_direction=trend_dir,
            trend_slope=round(trend_slope, 6),
            recommendation=recommendation,
        )

        self._history.append(report)
        self._composite_scores.append(composite_score)
        return report

    def _compute_composite_score(
        self,
        score_report: DriftReport,
        feature_drifts: List[FeatureDriftResult],
        n_feat_high: int,
    ) -> float:
        """
        Compute a 0..1 composite severity score from all signals.

        Weights:
        - Score KS statistic: 40%
        - Score PSI: 20%
        - Feature drift fraction: 25%
        - Worst feature KS: 15%
        """
        # Score KS: map p-value to severity (low p → high severity)
        # Use KS statistic directly (0..1 range)
        score_ks_severity = min(score_report.ks_statistic / 0.3, 1.0)

        # Score PSI: map to severity
        score_psi_severity = min(score_report.psi / 0.25, 1.0)

        # Feature drift fraction
        n_total = len(feature_drifts) if feature_drifts else 1
        n_drifted = sum(1 for fd in feature_drifts
                        if fd.level in (DriftLevel.WARNING.value, DriftLevel.HIGH.value))
        feat_frac_severity = n_drifted / max(n_total, 1)

        # Worst feature KS
        worst_ks = max((fd.ks_statistic for fd in feature_drifts), default=0.0)
        worst_feat_severity = min(worst_ks / 0.3, 1.0)

        composite = (
            0.40 * score_ks_severity
            + 0.20 * score_psi_severity
            + 0.25 * feat_frac_severity
            + 0.15 * worst_feat_severity
        )
        return min(composite, 1.0)

    def _detect_trend(self) -> tuple:
        """
        Detect drift trend from recent composite scores.

        Returns (direction: str, slope: float).
        Positive slope = worsening drift.
        """
        scores = list(self._composite_scores)
        if len(scores) < 3:
            return DriftTrend.STABLE.value, 0.0

        # Simple linear regression on the last N composite scores
        x = np.arange(len(scores), dtype=float)
        y = np.array(scores, dtype=float)
        # Center x for numerical stability
        x_centered = x - x.mean()
        denom = float(np.sum(x_centered ** 2))
        if denom < 1e-12:
            return DriftTrend.STABLE.value, 0.0

        slope = float(np.sum(x_centered * (y - y.mean())) / denom)

        if slope > 0.01:
            return DriftTrend.WORSENING.value, slope
        elif slope < -0.01:
            return DriftTrend.IMPROVING.value, slope
        else:
            return DriftTrend.STABLE.value, slope

    def _generate_recommendation(
        self,
        composite_level: str,
        trend: str,
        n_feat_high: int,
        worst_feat: Optional[str],
    ) -> str:
        """Generate human-readable recommendation."""
        parts = []

        if composite_level == "CRITICAL":
            parts.append(
                "CRITICAL: Major distribution shift detected in both scores and features. "
                "Immediately switch to STRICT mode. Consider halting predictions until "
                "root cause is identified."
            )
        elif composite_level == DriftLevel.HIGH.value:
            parts.append(
                "HIGH: Significant drift detected. Switch to STRICT mode and "
                "initiate local recalibration with fresh benign samples."
            )
        elif composite_level == DriftLevel.WARNING.value:
            parts.append(
                "WARNING: Moderate drift detected. Monitor closely. "
                "Consider widening the FLAG zone as a precautionary measure."
            )
        else:
            parts.append("OK: Distributions within expected range.")

        if n_feat_high > 0 and worst_feat:
            parts.append(
                f"Feature-level: {n_feat_high} feature(s) show HIGH drift. "
                f"Worst: '{worst_feat}'. Check if extraction pipeline changed."
            )

        if trend == DriftTrend.WORSENING.value:
            parts.append("TREND: Drift is WORSENING over recent checks. Escalation likely.")
        elif trend == DriftTrend.IMPROVING.value:
            parts.append("TREND: Drift is IMPROVING. Current measures may be working.")

        if self._alert_escalation_count > 0:
            parts.append(
                f"Alert escalation triggered {self._alert_escalation_count} time(s). "
                "Persistent drift requires operator intervention."
            )

        return " ".join(parts)

    # ── Properties ──

    @property
    def history(self) -> List[EnhancedDriftReport]:
        return list(self._history)

    @property
    def consecutive_warnings(self) -> int:
        return self._consecutive_warnings

    @property
    def alert_escalation_count(self) -> int:
        return self._alert_escalation_count

    def dashboard_data(self) -> Dict[str, Any]:
        """
        Export data suitable for a monitoring dashboard.

        Returns a dict with time-series of composite scores,
        current status, feature-level details, and trend.
        """
        last = self._history[-1] if self._history else None
        return {
            "current_status": last.composite_level if last else "UNKNOWN",
            "composite_score": last.composite_score if last else 0.0,
            "score_ks": last.score_drift.ks_statistic if last else 0.0,
            "score_psi": last.score_drift.psi if last else 0.0,
            "n_features_warning": last.n_features_warning if last else 0,
            "n_features_high": last.n_features_high if last else 0,
            "worst_feature": last.worst_feature if last else None,
            "trend": last.trend_direction if last else "UNKNOWN",
            "trend_slope": last.trend_slope if last else 0.0,
            "consecutive_warnings": self._consecutive_warnings,
            "alert_escalations": self._alert_escalation_count,
            "composite_score_history": list(self._composite_scores),
            "n_checks": len(self._history),
            "recommendation": last.recommendation if last else "",
        }

    # ── Persistence ──

    def save(self, path: Path) -> None:
        """Save monitor state to JSON."""
        data = {
            "feature_names": self.feature_names,
            "window_size": self.window_size,
            "config": {
                "ks_warning": self.ks_warning,
                "ks_high": self.ks_high,
                "psi_warning": self.psi_warning,
                "psi_high": self.psi_high,
                "feature_ks_warning": self.feature_ks_warning,
                "feature_ks_high": self.feature_ks_high,
                "critical_feature_pct": self.critical_feature_pct,
                "n_bins_psi": self.n_bins_psi,
            },
            "ref_feature_stats": self._ref_feature_stats,
            "score_monitor_ref": (
                self._score_monitor._reference.tolist()
                if self._score_monitor._reference is not None else []
            ),
            "ref_features": (
                self._ref_features.tolist()
                if self._ref_features is not None else []
            ),
            "consecutive_warnings": self._consecutive_warnings,
            "alert_escalation_count": self._alert_escalation_count,
            "composite_scores": list(self._composite_scores),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "EnhancedDriftMonitor":
        """Load monitor from saved state."""
        with open(path) as f:
            data = json.load(f)
        cfg = data.get("config", {})
        monitor = cls(
            feature_names=data.get("feature_names", []),
            window_size=data.get("window_size", 20),
            ks_warning_threshold=cfg.get("ks_warning", 0.05),
            ks_high_threshold=cfg.get("ks_high", 0.001),
            psi_warning_threshold=cfg.get("psi_warning", 0.10),
            psi_high_threshold=cfg.get("psi_high", 0.25),
            feature_ks_warning=cfg.get("feature_ks_warning", 0.15),
            feature_ks_high=cfg.get("feature_ks_high", 0.30),
            critical_feature_pct=cfg.get("critical_feature_pct", 0.50),
            n_bins_psi=cfg.get("n_bins_psi", 10),
        )
        # Restore score monitor
        ref_scores = data.get("score_monitor_ref", [])
        ref_features = data.get("ref_features", [])
        if ref_scores:
            ref_scores_arr = np.array(ref_scores, dtype=float)
            ref_features_arr = np.array(ref_features, dtype=float) if ref_features else None
            monitor.fit(ref_scores_arr, ref_features_arr)
        # Restore state
        monitor._consecutive_warnings = data.get("consecutive_warnings", 0)
        monitor._alert_escalation_count = data.get("alert_escalation_count", 0)
        for cs in data.get("composite_scores", []):
            monitor._composite_scores.append(cs)
        return monitor


