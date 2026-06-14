"""
Data-leakage validation suite for the CLEAN pipeline.

Five hard tests:
  1. test_no_capture_overlap_between_splits
  2. test_no_exact_duplicate_feature_rows_across_splits
  3. test_thresholds_are_validation_derived_only
  4. test_scaler_is_fit_only_on_training_data
  5. test_no_metadata_columns_in_model_features

All tests are deterministic and FAIL LOUDLY if required clean-pipeline
artifacts are missing. They are wired against
`src/clean_pipeline/validation/*`, NOT against the legacy `src/splits/*`.

Run only this file:
    pytest tests/test_data_leakage_clean_pipeline.py -v

Run a single test:
    pytest tests/test_data_leakage_clean_pipeline.py::test_no_capture_overlap_between_splits -v
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Set

import pandas as pd
import pytest

from src.clean_pipeline.validation import (
    FORBIDDEN_FEATURE_SUBSTRINGS,
    METADATA_COLS,
    capture_overlap_matrix,
    ensure_policy_threshold_provenance,
    ensure_preprocessing_metadata,
    hash_feature_rows,
    load_features_dataframe,
    load_split_capture_sets,
    locate_clean_artifacts,
    model_feature_columns,
)
from src.clean_pipeline.validation.preprocessing_metadata import stats_match_train


# ────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def paths():
    p = locate_clean_artifacts()
    p.assert_minimum_present()
    return p


@pytest.fixture(scope="module")
def features_df(paths) -> pd.DataFrame:
    return load_features_dataframe(paths)


@pytest.fixture(scope="module")
def split_captures(paths) -> Dict[str, Set[str]]:
    return load_split_capture_sets(paths)


@pytest.fixture(scope="module")
def summary_sink():
    """In-process collector for the per-check status; persisted by autouse fixture below."""
    return []


@pytest.fixture(autouse=True, scope="module")
def _persist_summary(paths, summary_sink):
    # Yield first so all tests can append.
    yield
    out_dir = paths.repo_root / "artifacts" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    (out_dir / "data_leakage_summary.json").write_text(
        json.dumps(summary_sink, indent=2), encoding="utf-8"
    )
    # CSV
    pd.DataFrame(summary_sink).to_csv(
        out_dir / "data_leakage_summary.csv", index=False
    )


def _record(summary_sink, check_name: str, status: str, details: str, counts: dict | None = None) -> None:
    summary_sink.append({
        "check_name": check_name,
        "status": status,
        "details": details,
        "counts": json.dumps(counts or {}, sort_keys=True),
    })


# ────────────────────────────────────────────────────────────────────────
# 1. Capture-level split-overlap
# ────────────────────────────────────────────────────────────────────────

def test_no_capture_overlap_between_splits(paths, split_captures, features_df, summary_sink):
    train = split_captures["train"]
    val = split_captures["val"]
    test = split_captures["test"]

    if not train or not val or not test:
        _record(summary_sink, "no_capture_overlap_between_splits", "FAIL",
                "One or more split capture lists are empty.",
                {"n_train": len(train), "n_val": len(val), "n_test": len(test)})
        pytest.fail(
            f"Empty split capture list(s): "
            f"train={len(train)} val={len(val)} test={len(test)}"
        )

    tv = train & val
    tt = train & test
    vt = val & test

    # Cross-check: the features parquet must agree with the txt manifests.
    parquet_pairs = (
        features_df[["capture_id", "split"]]
        .drop_duplicates()
        .groupby("split")["capture_id"]
        .apply(set)
        .to_dict()
    )
    parquet_disagreements = {}
    for s in ("train", "val", "test"):
        diff_txt_not_parquet = split_captures[s] - parquet_pairs.get(s, set())
        diff_parquet_not_txt = parquet_pairs.get(s, set()) - split_captures[s]
        if diff_txt_not_parquet or diff_parquet_not_txt:
            parquet_disagreements[s] = {
                "in_txt_not_parquet": sorted(diff_txt_not_parquet)[:5],
                "in_parquet_not_txt": sorted(diff_parquet_not_txt)[:5],
            }

    # Captures inside features.parquet may not span >1 split (capture integrity).
    multi_split_caps = (
        features_df.groupby("capture_id")["split"].nunique()
        .pipe(lambda s: s[s > 1])
    )

    ok = (
        not tv and not tt and not vt
        and not parquet_disagreements
        and multi_split_caps.empty
    )
    status = "PASS" if ok else "FAIL"
    _record(summary_sink, "no_capture_overlap_between_splits", status,
            "All pairwise capture intersections empty; txt manifests agree "
            "with features.parquet; no capture spans multiple splits."
            if ok else "Capture-level leakage detected; see counts.",
            {
                "n_train": len(train), "n_val": len(val), "n_test": len(test),
                "train_val_overlap": len(tv),
                "train_test_overlap": len(tt),
                "val_test_overlap":   len(vt),
                "parquet_vs_txt_disagreements": len(parquet_disagreements),
                "captures_spanning_multiple_splits": int(multi_split_caps.shape[0]),
            })

    assert not tv, f"train ∩ val captures non-empty ({len(tv)}): {sorted(tv)[:5]}"
    assert not tt, f"train ∩ test captures non-empty ({len(tt)}): {sorted(tt)[:5]}"
    assert not vt, f"val ∩ test captures non-empty ({len(vt)}): {sorted(vt)[:5]}"
    assert not parquet_disagreements, (
        f"features.parquet/split disagrees with txt capture lists: {parquet_disagreements}"
    )
    assert multi_split_caps.empty, (
        f"{multi_split_caps.shape[0]} capture(s) span multiple splits: "
        f"{multi_split_caps.head(5).to_dict()}"
    )


# ────────────────────────────────────────────────────────────────────────
# 2. Row-level duplicate leakage
# ────────────────────────────────────────────────────────────────────────

def test_no_exact_duplicate_feature_rows_across_splits(features_df, summary_sink):
    feat_cols = model_feature_columns(features_df)
    assert feat_cols, "No model feature columns identified — features.parquet schema unexpected."

    row_hash = hash_feature_rows(features_df, feat_cols)
    work = features_df[["split"]].copy()
    work["row_hash"] = row_hash.values

    # Within-split duplicates (informational, do NOT fail the test).
    within = (
        work.groupby(["split", "row_hash"]).size()
        .reset_index(name="n").query("n > 1")
    )

    # Cross-split duplicates: a single hash present in ≥2 distinct splits.
    splits_per_hash = work.groupby("row_hash")["split"].nunique()
    cross_hashes = splits_per_hash[splits_per_hash > 1].index
    cross = work[work["row_hash"].isin(cross_hashes)]
    n_cross_hashes = int(cross_hashes.size)
    n_cross_rows = int(cross.shape[0])

    ok = (n_cross_hashes == 0)
    status = "PASS" if ok else "FAIL"
    note = ""
    if not within.empty:
        note = (f" Note: {int(within['n'].sum() - within['split'].nunique())} "
                f"within-split duplicates (informational only).")
    _record(summary_sink, "no_exact_duplicate_feature_rows_across_splits", status,
            ("No identical feature row appears in more than one split." + note)
            if ok else
            f"{n_cross_hashes} hash(es) span multiple splits ({n_cross_rows} rows total).{note}",
            {
                "n_model_features": len(feat_cols),
                "n_within_split_dup_groups": int(within.shape[0]),
                "n_cross_split_dup_hashes": n_cross_hashes,
                "n_cross_split_dup_rows": n_cross_rows,
            })

    assert ok, (
        f"{n_cross_hashes} feature-row hash(es) appear in more than one split "
        f"({n_cross_rows} rows). Sample:\n"
        f"{cross.head(10).to_string()}"
    )


# ────────────────────────────────────────────────────────────────────────
# 3. Threshold provenance: validation-only
# ────────────────────────────────────────────────────────────────────────

def test_thresholds_are_validation_derived_only(paths, summary_sink):
    if not paths.evaluation_report.exists():
        _record(summary_sink, "thresholds_are_validation_derived_only", "FAIL",
                f"evaluation_report.json missing at {paths.evaluation_report}",
                {})
        pytest.fail(f"evaluation_report.json missing: {paths.evaluation_report}")

    provenance = ensure_policy_threshold_provenance(paths, overwrite=False)
    global_split = str(provenance.get("global_policy_fit_split", "")).lower()
    policies = provenance.get("policies", {}) or {}

    val_ok = global_split in {"val", "validation"}
    policy_violations = {
        name: meta.get("fit_split")
        for name, meta in policies.items()
        if str(meta.get("fit_split", "")).lower() not in {"val", "validation"}
    }
    ok = val_ok and not policy_violations

    _record(summary_sink, "thresholds_are_validation_derived_only",
            "PASS" if ok else "FAIL",
            "All policy thresholds carry fit_split == 'val'." if ok else
            "Threshold provenance violates val-only invariant.",
            {
                "global_policy_fit_split": global_split,
                "n_policies": len(policies),
                "n_violations": len(policy_violations),
                "policy_names": sorted(policies.keys()),
            })

    assert val_ok, (
        f"evaluation_report.policy_fit_split must be 'val', got {global_split!r}"
    )
    assert not policy_violations, (
        f"Policies with non-val fit_split: {policy_violations}"
    )


# ────────────────────────────────────────────────────────────────────────
# 4. Scaler / preprocessing fit-on-train-only
# ────────────────────────────────────────────────────────────────────────

def test_scaler_is_fit_only_on_training_data(paths, features_df, summary_sink):
    meta = ensure_preprocessing_metadata(paths, overwrite=False)

    fit_ok = (meta.fit_split == "train")
    cols_ok = (set(meta.feature_columns) == set(model_feature_columns(features_df)))
    transformed_ok = set(meta.transformed_splits) >= {"train", "val", "test"}

    mismatches = stats_match_train(meta, features_df)
    stats_ok = not mismatches

    # Sanity: stats must NOT match full-dataset stats (otherwise it was fit on all).
    full_df = features_df.copy()
    full_df["split"] = "train"  # trick: recompute on the whole frame
    full_mismatch = stats_match_train(meta, full_df)
    # If stats match BOTH train and full, the train and full means must already
    # be equal, which is degenerate. We require they differ for at least one
    # feature, confirming the metadata is genuinely train-only.
    not_fit_on_full = bool(full_mismatch)

    ok = fit_ok and cols_ok and transformed_ok and stats_ok and not_fit_on_full

    _record(summary_sink, "scaler_is_fit_only_on_training_data",
            "PASS" if ok else "FAIL",
            ("Preprocessing metadata declares fit_split=train, columns match, "
             "stats match recomputed train stats, and differ from full-dataset stats.")
            if ok else "Preprocessing metadata invariant violated; see counts.",
            {
                "fit_split": meta.fit_split,
                "transformer_type": meta.transformer_type,
                "n_feature_columns": len(meta.feature_columns),
                "n_stats_mismatches_vs_train": len(mismatches),
                "differs_from_full_dataset_stats": bool(not_fit_on_full),
            })

    assert fit_ok, f"meta.fit_split must be 'train', got {meta.fit_split!r}"
    assert cols_ok, (
        "meta.feature_columns disagree with current model_feature_columns."
    )
    assert transformed_ok, (
        f"meta.transformed_splits must include train/val/test; got {meta.transformed_splits}"
    )
    assert stats_ok, (
        f"Per-column stats deviate from recomputed train stats: {list(mismatches.keys())[:5]}"
    )
    assert not_fit_on_full, (
        "Train-split stats are indistinguishable from full-dataset stats; "
        "metadata may have been fit on the entire dataset."
    )


# ────────────────────────────────────────────────────────────────────────
# 5. No metadata columns in model features
# ────────────────────────────────────────────────────────────────────────

def test_no_metadata_columns_in_model_features(features_df, summary_sink):
    feat_cols = model_feature_columns(features_df)
    # Forbidden substring scan (case-insensitive).
    violations = []
    for col in feat_cols:
        cl = col.lower()
        for sub in FORBIDDEN_FEATURE_SUBSTRINGS:
            if sub in cl:
                violations.append({"column": col, "matched_substring": sub})
                break

    # And no exact-name metadata leak.
    name_overlap = sorted(set(feat_cols).intersection(METADATA_COLS))

    ok = not violations and not name_overlap
    _record(summary_sink, "no_metadata_columns_in_model_features",
            "PASS" if ok else "FAIL",
            f"All {len(feat_cols)} model feature columns are free of metadata substrings."
            if ok else f"{len(violations)} forbidden substring match(es); "
                      f"{len(name_overlap)} exact metadata name(s) found.",
            {
                "n_model_features": len(feat_cols),
                "n_substring_violations": len(violations),
                "n_exact_metadata_in_features": len(name_overlap),
                "violations_sample": violations[:5],
            })

    assert not violations, (
        f"Forbidden metadata substrings found in model features: {violations[:10]}"
    )
    assert not name_overlap, (
        f"Exact metadata columns leaked into model features: {name_overlap}"
    )

