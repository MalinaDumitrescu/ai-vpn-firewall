"""
Clean-pipeline data-leakage validation helpers.

These utilities are used by:
  - tests/test_data_leakage_clean_pipeline.py
  - notebooks/validation_data_leakage_audit.ipynb

They operate exclusively on artifacts produced by the CURRENT clean pipeline
(`src/clean_pipeline/run_pipeline.py`). Legacy-pipeline paths are not used.
"""

from src.clean_pipeline.validation.leakage_checks import (
    METADATA_COLS,
    FORBIDDEN_FEATURE_SUBSTRINGS,
    CleanArtifactPaths,
    locate_clean_artifacts,
    load_split_capture_sets,
    load_features_dataframe,
    model_feature_columns,
    hash_feature_rows,
    capture_overlap_matrix,
)
from src.clean_pipeline.validation.preprocessing_metadata import (
    ensure_preprocessing_metadata,
    PreprocessingMetadata,
)
from src.clean_pipeline.validation.threshold_provenance import (
    ensure_policy_threshold_provenance,
)

__all__ = [
    "METADATA_COLS",
    "FORBIDDEN_FEATURE_SUBSTRINGS",
    "CleanArtifactPaths",
    "locate_clean_artifacts",
    "load_split_capture_sets",
    "load_features_dataframe",
    "model_feature_columns",
    "hash_feature_rows",
    "capture_overlap_matrix",
    "ensure_preprocessing_metadata",
    "PreprocessingMetadata",
    "ensure_policy_threshold_provenance",
]

