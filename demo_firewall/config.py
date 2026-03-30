# demo_firewall/config.py
"""
Deployment configuration for the VPN detection firewall pipeline.

Defines deployment modes, feature sets, and operational parameters.
All validated thresholds come from the tuned ensemble experiment
(balanced_bagging_firewall_tuned_ensemble).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import yaml


# ──────────────────────────────────────────────────────
# Feature sets
# ──────────────────────────────────────────────────────

COMPACT_FEATURES: List[str] = [
    "sz_coef_variation",
    "sz_p25_median_ratio",
    "sz_p75_median_ratio",
    "sz_iqr_norm_median",
    "dispersion_symmetry",
    "direction_balance_bytes",
    "direction_balance_packets",
]

DIRECTION_FEATURES: List[str] = [
    "direction_balance_bytes",
    "direction_balance_packets",
]

REDUCED_FEATURES: List[str] = [
    f for f in COMPACT_FEATURES if f not in DIRECTION_FEATURES
]

# ──────────────────────────────────────────────────────
# Flow extraction defaults
# ──────────────────────────────────────────────────────

DEFAULT_WINDOW_N: int = 100
DEFAULT_MIN_PACKETS: int = 10
DEFAULT_EPS: float = 1e-6

# ──────────────────────────────────────────────────────
# Ensemble model families
# ──────────────────────────────────────────────────────

MODEL_FAMILIES: List[str] = ["xgb", "lgbm", "cat"]
BAGS_PER_FAMILY: int = 3


# ──────────────────────────────────────────────────────
# Deployment modes
# ──────────────────────────────────────────────────────

class DeploymentMode(str, Enum):
    """
    Deployment operating modes for the firewall classifier.

    STRICT:   Zero block-FPR enforced. p90 session aggregation.
              Maximum benign threshold. Default for production.
    BALANCED: Optimize recall under ≤0.1% FPR constraint.
              Suitable for monitored deployments.
    RESEARCH: Raw probability output only. No thresholding.
              For offline analysis and experimentation.
    """
    STRICT = "strict"
    BALANCED = "balanced"
    RESEARCH = "research"


@dataclass(frozen=True)
class ModeConfig:
    """Per-mode operational parameters."""
    target_fpr: float
    aggregation_rule: str          # "p90" | "weighted_top5_mean" | "mean"
    enforce_zero_block_fpr: bool
    description: str


MODE_CONFIGS: Dict[DeploymentMode, ModeConfig] = {
    DeploymentMode.STRICT: ModeConfig(
        target_fpr=0.0,
        aggregation_rule="p90",
        enforce_zero_block_fpr=True,
        description="Zero block-FPR. p90 aggregation. Max-benign threshold.",
    ),
    DeploymentMode.BALANCED: ModeConfig(
        target_fpr=0.001,
        aggregation_rule="weighted_top5_mean",
        enforce_zero_block_fpr=False,
        description="Recall-optimized under ≤0.1% FPR constraint.",
    ),
    DeploymentMode.RESEARCH: ModeConfig(
        target_fpr=1.0,       # No constraint
        aggregation_rule="mean",
        enforce_zero_block_fpr=False,
        description="Raw probability output. No thresholding applied.",
    ),
}


# ──────────────────────────────────────────────────────
# Artifact paths (relative to project root)
# ──────────────────────────────────────────────────────

@dataclass
class ArtifactPaths:
    """Resolved paths to all required model artifacts."""
    ensemble_dir: Path
    features_dir: Path

    @property
    def model_paths(self) -> Dict[str, List[Path]]:
        """Returns {family: [bag0.pkl, bag1.pkl, bag2.pkl]}."""
        out: Dict[str, List[Path]] = {}
        for family in MODEL_FAMILIES:
            out[family] = [
                self.ensemble_dir / f"model_{family}_bag{i}.pkl"
                for i in range(BAGS_PER_FAMILY)
            ]
        return out

    @property
    def isotonic_calibrator_path(self) -> Path:
        return self.ensemble_dir / "isotonic_calibrator.pkl"

    @property
    def platt_calibrator_path(self) -> Path:
        return self.ensemble_dir / "platt_calibrator.pkl"

    @property
    def metrics_path(self) -> Path:
        return self.ensemble_dir / "metrics.json"

    @property
    def feature_columns_json(self) -> Path:
        return self.features_dir / "feature_columns.json"

    @property
    def scaler_pkl(self) -> Path:
        return self.features_dir / "scaler.pkl"

    def validate(self) -> List[str]:
        """Return list of missing artifact paths."""
        missing = []
        for family, paths in self.model_paths.items():
            for p in paths:
                if not p.exists():
                    missing.append(str(p))
        for p in [
            self.isotonic_calibrator_path,
            self.feature_columns_json,
            self.scaler_pkl,
        ]:
            if not p.exists():
                missing.append(str(p))
        return missing


def default_artifact_paths(repo_root: Path) -> ArtifactPaths:
    """Construct default artifact paths from repo root."""
    return ArtifactPaths(
        ensemble_dir=repo_root / "artifacts" / "balanced_bagging_firewall_tuned_ensemble",
        features_dir=repo_root / "artifacts" / "features",
    )


# ──────────────────────────────────────────────────────
# Threshold configuration
# ──────────────────────────────────────────────────────

@dataclass
class ThresholdConfig:
    """
    Threshold provenance record.

    Stores how the threshold was computed and on what data,
    enabling leakage detection.
    """
    block_threshold: float
    flag_threshold: float
    source_split: str                     # "val" — must be benign-only reference
    aggregation_rule: str
    calibration_method: str               # "isotonic" | "platt" | "raw"
    computed_on_benign_only: bool = True   # Safety flag

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_threshold": self.block_threshold,
            "flag_threshold": self.flag_threshold,
            "source_split": self.source_split,
            "aggregation_rule": self.aggregation_rule,
            "calibration_method": self.calibration_method,
            "computed_on_benign_only": self.computed_on_benign_only,
        }


def load_thresholds_yaml(path: Path) -> Dict[str, Any]:
    """Load threshold configuration from YAML."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


