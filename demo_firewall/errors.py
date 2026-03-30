# demo_firewall/errors.py
"""
Custom exception hierarchy for the VPN detection firewall pipeline.

Safety-critical errors that prevent silent misclassification.
"""
from __future__ import annotations


class FirewallPipelineError(Exception):
    """Base class for all firewall pipeline errors."""
    pass


class CalibrationError(FirewallPipelineError):
    """
    Raised when probability calibration cannot be performed safely.

    Triggers:
    - Only one class present in calibration split (label collapse).
    - Calibrated outputs violate [0, 1] bounds.
    - Isotonic regression produces degenerate (constant) mapping.
    """
    pass


class ThresholdLeakageError(FirewallPipelineError):
    """
    Raised when the blocking threshold was computed on contaminated data.

    Triggers:
    - Threshold derived from a split containing positive (VPN) labels
      in the benign reference pool.
    - Threshold provenance metadata missing or invalid.
    """
    pass


class ModelLoadError(FirewallPipelineError):
    """
    Raised when ensemble model artifacts cannot be loaded.

    Triggers:
    - Missing .pkl files for any bag/family.
    - Version mismatch between saved model and current library.
    - Corrupted pickle payload.
    """
    pass


class FeatureExtractionError(FirewallPipelineError):
    """
    Raised when feature extraction produces invalid results.

    Triggers:
    - Flow has fewer packets than min_packets.
    - Non-finite values after extraction.
    - Missing required compact features.
    """
    pass


class InsufficientDataError(FirewallPipelineError):
    """
    Raised when there is insufficient data for a reliable decision.

    Triggers:
    - Session has zero valid flows after filtering.
    - Capture contains no extractable packets.
    """
    pass

