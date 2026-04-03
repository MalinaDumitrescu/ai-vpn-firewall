# tests/test_clean_pipeline.py
"""
Tests for the CLEAN pipeline.

Verifies:
  - Feature extraction produces correct values from synthetic data
  - All family features are computed
  - Merger schema is correct
  - Splitter preserves capture integrity
  - Config loading works
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clean_pipeline.feature_extractor import extract_flow_features, extract_features_batch
from src.clean_pipeline.feature_families import (
    FEATURE_REGISTRY,
    FAMILY_REGISTRY,
    FeatureSafety,
    get_family,
    get_family_safety,
    family_has_risky_features,
    PERMANENTLY_EXCLUDED,
)
from src.clean_pipeline.splitter import CleanSplitConfig, make_clean_split
from src.clean_pipeline.config import CleanPipelineConfig, default_config


# ──────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────

def _make_synthetic_flow(n_packets: int = 50, seed: int = 0):
    """Create a synthetic flow for testing."""
    rng = np.random.default_rng(seed)
    timestamps = np.sort(rng.uniform(0, 10, n_packets))
    sizes = rng.integers(40, 1500, n_packets)
    directions = rng.choice([0, 1], n_packets)
    return timestamps, sizes, directions


def _make_synthetic_flows_df(
    n_flows: int = 100,
    n_captures: int = 10,
    n_datasets: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic flows DataFrame for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_flows):
        cap_idx = i % n_captures
        ds_idx = cap_idx % n_datasets
        ds_name = ["vnat", "iscx", "usbvpn"][ds_idx % 3]
        label = 1 if cap_idx % 3 == 0 else 0

        ts, sz, dr = _make_synthetic_flow(n_packets=rng.integers(10, 80), seed=seed + i)
        rows.append({
            "flow_id": f"{ds_name}::{i}",
            "capture_id": f"{ds_name}_cap_{cap_idx}",
            "source_file": f"file_{i}.pcap",
            "dataset": ds_name,
            "label": label,
            "timestamps": ts.tolist(),
            "sizes": sz.tolist(),
            "directions": dr.tolist(),
            "app": f"app_{cap_idx}",
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────
# Feature extraction tests
# ──────────────────────────────────────────────────────

class TestExtractFlowFeatures:
    def test_returns_all_registered_features(self):
        ts, sz, dr = _make_synthetic_flow(50, seed=1)
        feat = extract_flow_features(ts, sz, dr)

        # Should have all SAFE features at minimum
        safe_features = [
            name for name, spec in FEATURE_REGISTRY.items()
            if spec.safety == FeatureSafety.SAFE
        ]
        for f in safe_features:
            assert f in feat, f"Missing SAFE feature: {f}"

    def test_total_packets_correct(self):
        ts, sz, dr = _make_synthetic_flow(50, seed=2)
        feat = extract_flow_features(ts, sz, dr)
        assert feat["total_packets"] == 50.0

    def test_total_bytes_correct(self):
        ts = np.array([0.0, 1.0, 2.0])
        sz = np.array([100, 200, 300])
        dr = np.array([1, 0, 1])
        feat = extract_flow_features(ts, sz, dr)
        assert feat["total_bytes"] == 600.0

    def test_mean_pkt_len_correct(self):
        ts = np.array([0.0, 1.0, 2.0, 3.0])
        sz = np.array([100, 200, 300, 400])
        dr = np.array([1, 0, 1, 0])
        feat = extract_flow_features(ts, sz, dr)
        assert feat["mean_pkt_len"] == 250.0

    def test_iat_mean_correct(self):
        ts = np.array([0.0, 1.0, 3.0, 6.0])
        sz = np.array([100, 100, 100, 100])
        dr = np.array([1, 1, 1, 1])
        feat = extract_flow_features(ts, sz, dr)
        # IATs: [1, 2, 3], mean = 2.0
        assert abs(feat["iat_mean"] - 2.0) < 1e-6

    def test_flow_duration_correct(self):
        ts = np.array([10.0, 12.0, 15.0])
        sz = np.array([100, 100, 100])
        dr = np.array([1, 0, 1])
        feat = extract_flow_features(ts, sz, dr)
        assert abs(feat["flow_duration"] - 5.0) < 1e-6

    def test_direction_invariant_features(self):
        """Direction-invariant features should be the same regardless of dir labeling."""
        ts = np.array([0.0, 1.0, 2.0, 3.0])
        sz = np.array([100, 200, 300, 400])
        dr_a = np.array([1, 0, 1, 0])
        dr_b = np.array([0, 1, 0, 1])  # flipped

        feat_a = extract_flow_features(ts, sz, dr_a)
        feat_b = extract_flow_features(ts, sz, dr_b)

        for f in ["dir_pkt_ratio_minmax", "dir_bytes_ratio_minmax",
                   "dir_mean_pkt_max", "dir_mean_pkt_min"]:
            assert abs(feat_a[f] - feat_b[f]) < 1e-9, \
                f"{f} not direction-invariant: {feat_a[f]} != {feat_b[f]}"

    def test_window_truncation(self):
        ts, sz, dr = _make_synthetic_flow(200, seed=3)
        feat = extract_flow_features(ts, sz, dr, max_packets=50)
        assert feat["total_packets"] == 50.0

    def test_all_features_finite(self):
        ts, sz, dr = _make_synthetic_flow(30, seed=4)
        feat = extract_flow_features(ts, sz, dr)
        for name, val in feat.items():
            assert np.isfinite(val), f"Feature {name} is not finite: {val}"


# ──────────────────────────────────────────────────────
# Batch extraction tests
# ──────────────────────────────────────────────────────

class TestExtractFeaturesBatch:
    def test_batch_output_shape(self):
        df = _make_synthetic_flows_df(n_flows=20, n_captures=4, seed=10)
        result = extract_features_batch(
            df, family="safe_core_10", max_packets=50, progress=False
        )
        assert len(result) <= 20
        for col in ["flow_id", "capture_id", "dataset", "label"]:
            assert col in result.columns

        # All 10 safe core features should be present
        for f in get_family("safe_core_10"):
            assert f in result.columns, f"Missing feature: {f}"

    def test_no_excluded_features(self):
        df = _make_synthetic_flows_df(n_flows=10, n_captures=2, seed=11)
        result = extract_features_batch(
            df, family="safe_core_10", max_packets=50, progress=False
        )
        for col in PERMANENTLY_EXCLUDED:
            assert col not in result.columns

    def test_all_families_extractable(self):
        """Verify every registered family can be fully extracted."""
        df = _make_synthetic_flows_df(n_flows=10, n_captures=2, seed=12)
        for family_name in FAMILY_REGISTRY:
            result = extract_features_batch(
                df, family=family_name, max_packets=50, progress=False
            )
            expected = get_family(family_name)
            for f in expected:
                assert f in result.columns, \
                    f"Family '{family_name}' missing feature '{f}'"


# ──────────────────────────────────────────────────────
# Feature families tests
# ──────────────────────────────────────────────────────

class TestFeatureFamilies:
    def test_safe_core_10_all_safe(self):
        safety = get_family_safety("safe_core_10")
        for f, s in safety.items():
            assert s == FeatureSafety.SAFE, f"Feature {f} is {s}, expected SAFE"

    def test_direction_augmented_has_risky(self):
        assert family_has_risky_features("direction_augmented")

    def test_direction_invariant_augmented_all_safe(self):
        assert not family_has_risky_features("direction_invariant_augmented")

    def test_families_are_nested(self):
        """Each larger family should contain the previous."""
        c10 = set(get_family("safe_core_10"))
        dur = set(get_family("safe_core_plus_duration"))
        temp = set(get_family("safe_core_plus_temporal"))
        dinv = set(get_family("direction_invariant_augmented"))
        daug = set(get_family("direction_augmented"))

        assert c10 <= dur <= temp <= dinv <= daug

    def test_no_excluded_in_families(self):
        for family_name, features in FAMILY_REGISTRY.items():
            for f in features:
                assert f not in PERMANENTLY_EXCLUDED, \
                    f"Excluded feature '{f}' found in family '{family_name}'"


# ──────────────────────────────────────────────────────
# Splitter tests
# ──────────────────────────────────────────────────────

class TestSplitter:
    def test_split_assigns_all_rows(self):
        df = _make_synthetic_flows_df(n_flows=100, n_captures=10, seed=20)
        features = extract_features_batch(
            df, family="safe_core_10", max_packets=50, progress=False
        )
        result = make_clean_split(features, CleanSplitConfig(seed=42))
        assert "split" in result.columns
        assert result["split"].isna().sum() == 0
        assert set(result["split"].unique()) <= {"train", "val", "test"}

    def test_split_preserves_capture_integrity(self):
        """All flows from one capture must be in the same split."""
        df = _make_synthetic_flows_df(n_flows=200, n_captures=20, seed=21)
        features = extract_features_batch(
            df, family="safe_core_10", max_packets=50, progress=False
        )
        result = make_clean_split(features, CleanSplitConfig(seed=42))

        for cap_id, group in result.groupby("capture_id"):
            splits = group["split"].unique()
            assert len(splits) == 1, \
                f"Capture {cap_id} split across: {splits}"

    def test_split_reproducible(self):
        df = _make_synthetic_flows_df(n_flows=100, n_captures=10, seed=22)
        features = extract_features_batch(
            df, family="safe_core_10", max_packets=50, progress=False
        )
        r1 = make_clean_split(features.copy(), CleanSplitConfig(seed=42))
        r2 = make_clean_split(features.copy(), CleanSplitConfig(seed=42))

        assert list(r1["split"]) == list(r2["split"])


# ──────────────────────────────────────────────────────
# Config tests
# ──────────────────────────────────────────────────────

class TestConfig:
    def test_default_config_valid(self):
        cfg = default_config()
        assert cfg.max_packets > 0
        assert cfg.min_packets > 0
        assert cfg.feature_family in FAMILY_REGISTRY
        assert abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) < 0.01

    def test_config_dataclass(self):
        cfg = CleanPipelineConfig(
            max_packets=100,
            min_packets=5,
            feature_family="safe_core_10",
            seed=123,
        )
        assert cfg.max_packets == 100
        assert cfg.seed == 123

