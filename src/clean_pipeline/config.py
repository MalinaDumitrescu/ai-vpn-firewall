# src/clean_pipeline/config.py
"""
Configuration for the CLEAN pipeline.

Single source of truth for all pipeline parameters. Can be loaded
from YAML or constructed programmatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class CleanPipelineConfig:
    """Full configuration for a clean pipeline run."""

    # ── Data paths ──
    vnat_h5: Optional[Path] = None
    iscx_parquet: Optional[Path] = None
    usbvpn_raw_dir: Optional[Path] = None

    # ── Output paths ──
    output_dir: Path = Path("artifacts/clean_pipeline")
    splits_dir: Path = Path("data/splits")

    # ── Window ──
    max_packets: int = 300
    min_packets: int = 3

    # ── Feature family ──
    feature_family: str = "direction_invariant_augmented"

    # ── Splitting ──
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    min_captures_per_class_per_split: int = 2
    splitter_version: int = 2  # 1 = legacy greedy, 2 = constrained (recommended)

    # ── Scaling ──
    apply_quantile_scaling: bool = True
    quantile_n: int = 1000

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.splits_dir = Path(self.splits_dir)
        if self.vnat_h5 is not None:
            self.vnat_h5 = Path(self.vnat_h5)
        if self.iscx_parquet is not None:
            self.iscx_parquet = Path(self.iscx_parquet)
        if self.usbvpn_raw_dir is not None:
            self.usbvpn_raw_dir = Path(self.usbvpn_raw_dir)


def load_clean_config(yaml_path: Path) -> CleanPipelineConfig:
    """Load configuration from a YAML file."""
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    data = raw.get("data", {})
    window = raw.get("window", {})
    features = raw.get("features", {})
    splitting = raw.get("splitting", {})
    output = raw.get("output", {})
    scaling = raw.get("scaling", {})

    return CleanPipelineConfig(
        vnat_h5=Path(data["vnat_h5"]) if data.get("vnat_h5") else None,
        iscx_parquet=Path(data["iscx_parquet"]) if data.get("iscx_parquet") else None,
        usbvpn_raw_dir=Path(data["usbvpn_raw_dir"]) if data.get("usbvpn_raw_dir") else None,
        output_dir=Path(output.get("artifacts_dir", "artifacts/clean_pipeline")),
        splits_dir=Path(output.get("splits_dir", "data/splits")),
        max_packets=int(window.get("max_packets", 300)),
        min_packets=int(window.get("min_packets", 3)),
        feature_family=str(features.get("family", "direction_invariant_augmented")),
        seed=int(splitting.get("seed", 42)),
        train_ratio=float(splitting.get("train_ratio", 0.70)),
        val_ratio=float(splitting.get("val_ratio", 0.15)),
        test_ratio=float(splitting.get("test_ratio", 0.15)),
        min_captures_per_class_per_split=int(
            splitting.get("min_captures_per_class_per_split", 2)
        ),
        splitter_version=int(splitting.get("splitter_version", 2)),
        apply_quantile_scaling=bool(scaling.get("apply_quantile_scaling", True)),
        quantile_n=int(scaling.get("quantile_n", 1000)),
    )


def default_config() -> CleanPipelineConfig:
    """
    Build default config with auto-detected paths.

    Searches for the repo root (directory containing pyproject.toml)
    and resolves dataset paths relative to it.
    """
    here = Path(__file__).resolve().parent
    for candidate in [here.parent.parent, Path.cwd()]:
        if (candidate / "pyproject.toml").exists():
            root = candidate
            break
    else:
        root = Path.cwd()

    return CleanPipelineConfig(
        vnat_h5=root / "data" / "raw" / "vnat" / "VNAT_Dataframe_release_1.h5",
        iscx_parquet=root / "data" / "processed" / "iscx" / "flows.parquet",
        usbvpn_raw_dir=root / "data" / "raw" / "usbvpn",
        output_dir=root / "artifacts" / "clean_pipeline",
        splits_dir=root / "data" / "splits",
    )


