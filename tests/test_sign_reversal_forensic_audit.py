from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.sign_reversal_forensic_audit import (
    apply_transform_variant,
    build_feature_construction_audit,
    build_preprocessing_comparison,
    build_strict_loose_reversal_report,
    compute_effects_table,
    summarize_reversals,
)


FEATURES = ["f1", "f2"]


def _make_df() -> pd.DataFrame:
    rows = []
    # f1 reverses: positive in iscx, negative in usbvpn/vnat
    # f2 stays positive everywhere
    specs = {
        "iscx": {
            1: [(3.0, 4.0), (4.0, 5.0)],
            0: [(1.0, 1.0), (2.0, 2.0)],
        },
        "usbvpn": {
            1: [(1.0, 4.0), (2.0, 5.0)],
            0: [(3.0, 1.0), (4.0, 2.0)],
        },
        "vnat": {
            1: [(1.0, 3.0), (2.0, 4.0)],
            0: [(3.0, 1.0), (4.0, 2.0)],
        },
    }
    idx = 0
    for dataset, by_label in specs.items():
        for label, values in by_label.items():
            for cap_i, pair in enumerate(values):
                rows.append(
                    {
                        "flow_id": f"{dataset}::{idx}",
                        "capture_id": f"{dataset}_cap_{label}_{cap_i}",
                        "dataset": dataset,
                        "label": label,
                        "source_file": f"{dataset}_{label}_{cap_i}.pcap",
                        "app": f"app_{cap_i}",
                        "f1": pair[0],
                        "f2": pair[1],
                    }
                )
                idx += 1
    return pd.DataFrame(rows)


def test_raw_reversal_detected_and_affine_scaling_preserves_it():
    df = _make_df()
    raw = compute_effects_table(df, FEATURES, analysis_name="all_flows", transform_name="raw", seed=42)
    raw_summary = summarize_reversals(raw)

    scaled_df = apply_transform_variant(df, FEATURES, "per_dataset_zscore", seed=42)
    scaled = compute_effects_table(scaled_df, FEATURES, analysis_name="all_flows", transform_name="per_dataset_zscore", seed=42)
    scaled_summary = summarize_reversals(scaled)

    raw_map = raw_summary.set_index("feature")
    scaled_map = scaled_summary.set_index("feature")

    assert bool(raw_map.at["f1", "consensus_reversal"])
    assert not bool(raw_map.at["f2", "consensus_reversal"])
    assert bool(scaled_map.at["f1", "consensus_reversal"])
    assert not bool(scaled_map.at["f2", "consensus_reversal"])

    preprocessing = build_preprocessing_comparison(
        {"raw": raw_summary, "per_dataset_zscore": scaled_summary},
        build_strict_loose_reversal_report(
            pd.DataFrame(
                [
                    {"dataset": ds, "feature": "f1", "metric": m, "estimate": 1.0, "ci_low": 0.5, "ci_high": 1.5, "weak_zone": 0.1, "strength_tag": ("positive strong" if ds == "iscx" else "negative strong")}
                    for ds in ["iscx", "usbvpn", "vnat"]
                    for m in ["cohen_d", "cliffs_delta", "signed_auc"]
                ]
                + [
                    {"dataset": ds, "feature": "f2", "metric": m, "estimate": 1.0, "ci_low": 0.5, "ci_high": 1.5, "weak_zone": 0.1, "strength_tag": "positive strong"}
                    for ds in ["iscx", "usbvpn", "vnat"]
                    for m in ["cohen_d", "cliffs_delta", "signed_auc"]
                ]
            )
        ),
        FEATURES,
    )
    prep_map = preprocessing.set_index("feature")
    assert not bool(prep_map.at["f1", "reversal_introduced_only_after_scaling"])


def test_feature_construction_audit_marks_safe_temporal_features_direction_safe():
    audit = build_feature_construction_audit(["total_packets", "iat_mean", "flow_duration"], max_packets=300)
    assert set(audit["direction_safe"]) == {"yes"}
    assert set(audit["computed_identically_across_datasets"]) == {"yes"}
