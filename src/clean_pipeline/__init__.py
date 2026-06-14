# src/clean_pipeline/__init__.py
"""
TRACK B — Clean unified feature pipeline for cross-dataset VPN detection.

This module provides scientifically honest feature extraction from raw
packet-level data across all three datasets (VNAT, ISCX, USBVPN).

Key invariant: every feature used in a CLEAN experiment must be computed
from the same formula applied to the same semantic input across all datasets.

Modules
-------
vnat_loader         Load VNAT from HDF5
iscx_loader         Load ISCX from pre-processed parquet
usbvpn_parser       Parse USBVPN from raw JSON
merge_datasets      Merge all datasets into unified schema
feature_families    Feature registry & safety classifications
feature_extractor   Compute features from packet arrays
splitter            Capture-level cross-dataset splitting
config              Pipeline configuration
run_pipeline        End-to-end orchestrator
"""

from src.clean_pipeline.config import CleanPipelineConfig, default_config, load_clean_config
from src.clean_pipeline.feature_extractor import extract_features_batch, extract_flow_features
from src.clean_pipeline.feature_families import (
    FAMILY_REGISTRY,
    FEATURE_REGISTRY,
    FeatureSafety,
    get_family,
    get_family_safety,
)
from src.clean_pipeline.merge_datasets import merge_all_datasets
from src.clean_pipeline.run_pipeline import run_clean_pipeline
from src.clean_pipeline.splitter import CleanSplitConfig, make_clean_split

__all__ = [
    "CleanPipelineConfig",
    "CleanSplitConfig",
    "default_config",
    "extract_features_batch",
    "extract_flow_features",
    "FAMILY_REGISTRY",
    "FEATURE_REGISTRY",
    "FeatureSafety",
    "get_family",
    "get_family_safety",
    "load_clean_config",
    "make_clean_split",
    "merge_all_datasets",
    "run_clean_pipeline",
]
