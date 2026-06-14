"""
Split integrity validation suite for the CLEAN pipeline.
Covers: assignment, class presence, dominance, and splitter stability.
"""
import pytest
import numpy as np
import pandas as pd
from src.clean_pipeline.splitter import CleanSplitConfig, make_clean_split
from src.clean_pipeline.feature_families import get_family

def synthetic_flows_fixture():
    # Create a synthetic flows DataFrame with multiple datasets, captures, classes, and imbalance
    rng = np.random.default_rng(42)
    datasets = ["iscx", "usbvpn", "vnat"]
    n_captures = 60
    n_flows = 600
    rows = []
    for i in range(n_flows):
        cap_idx = i % n_captures
        ds = datasets[cap_idx % len(datasets)]
        label = 1 if (cap_idx % 7 == 0 or (i % 13 == 0)) else 0  # Scarce minority
        rows.append({
            "flow_id": f"{ds}::{i}",
            "capture_id": f"{ds}_cap_{cap_idx}",
            "dataset": ds,
            "label": label,
        })
    return pd.DataFrame(rows)

def test_every_flow_assigned_to_exactly_one_split():
    df = synthetic_flows_fixture()
    cfg = CleanSplitConfig(seed=123)
    result = make_clean_split(df.copy(), cfg)
    assert "split" in result.columns
    assert result["split"].isna().sum() == 0
    assert set(result["split"].unique()) <= {"train", "val", "test"}
    # No duplicated flow_id across splits
    if "flow_id" in result.columns:
        dup = result.groupby("flow_id")["split"].nunique()
        assert (dup <= 1).all(), f"Duplicated flow_id across splits: {dup[dup > 1]}"
    # No missing assignments
    assert result["split"].isna().sum() == 0

def test_each_dataset_split_contains_both_classes():
    df = synthetic_flows_fixture()
    cfg = CleanSplitConfig(seed=456)
    result = make_clean_split(df.copy(), cfg)
    for ds in result["dataset"].unique():
        for split in ("train", "val", "test"):
            sub = result[(result["dataset"] == ds) & (result["split"] == split)]
            if len(sub) == 0:
                continue  # skip empty combinations
            vpn_count = (sub["label"] == 1).sum()
            nonvpn_count = (sub["label"] == 0).sum()
            assert vpn_count > 0, f"No VPN in {ds}/{split}"
            assert nonvpn_count > 0, f"No nonVPN in {ds}/{split}"

def test_no_single_capture_dominates_split():
    df = synthetic_flows_fixture()
    # Make one large capture to test dominance
    df.loc[df["capture_id"] == "iscx_cap_0", "capture_id"] = "iscx_cap_BIG"
    cfg = CleanSplitConfig(seed=789, max_capture_share_per_split=0.40)
    result = make_clean_split(df.copy(), cfg)
    for ds in result["dataset"].unique():
        for split in ("train", "val", "test"):
            sub = result[(result["dataset"] == ds) & (result["split"] == split)]
            if len(sub) == 0:
                continue
            cap_sizes = sub.groupby("capture_id").size()
            max_share = cap_sizes.max() / len(sub)
            assert max_share <= cfg.max_capture_share_per_split + 1e-6, (
                f"Dominance: {ds}/{split} {max_share:.3f} > {cfg.max_capture_share_per_split}"
            )

def test_splitter_stability_across_random_seeds():
    df = synthetic_flows_fixture()
    for seed in range(20):
        cfg = CleanSplitConfig(seed=seed)
        result = make_clean_split(df.copy(), cfg)
        # No capture overlap
        for cap_id, group in result.groupby("capture_id"):
            splits = group["split"].unique()
            assert len(splits) == 1, f"Capture {cap_id} split across: {splits}"
        # All flows assigned once
        assert result["split"].isna().sum() == 0
        # Both classes present in val and test for each dataset
        for ds in result["dataset"].unique():
            for split in ("val", "test"):
                sub = result[(result["dataset"] == ds) & (result["split"] == split)]
                if len(sub) == 0:
                    continue
                vpn_count = (sub["label"] == 1).sum()
                nonvpn_count = (sub["label"] == 0).sum()
                assert vpn_count > 0, f"Seed {seed}: No VPN in {ds}/{split}"
                assert nonvpn_count > 0, f"Seed {seed}: No nonVPN in {ds}/{split}"
        # No dominance above threshold
        for ds in result["dataset"].unique():
            for split in ("train", "val", "test"):
                sub = result[(result["dataset"] == ds) & (result["split"] == split)]
                if len(sub) == 0:
                    continue
                cap_sizes = sub.groupby("capture_id").size()
                max_share = cap_sizes.max() / len(sub)
                assert max_share <= cfg.max_capture_share_per_split + 1e-6, (
                    f"Seed {seed}: Dominance {ds}/{split} {max_share:.3f} > {cfg.max_capture_share_per_split}"
                )

