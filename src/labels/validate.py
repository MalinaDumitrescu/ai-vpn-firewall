from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd

from src.labels.vnat import label_from_filename as vnat_label_from_filename
from src.labels.iscx import label_from_filename as iscx_label_from_filename


@dataclass
class LabelValidationReport:
    dataset: str
    rows: int
    unique_captures: int
    label_counts: Dict[int, int]
    unknown_prefix_rows: int
    mixed_label_captures: int
    duplicate_flow_ids: int
    notes: List[str]


def _pick_id_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns exist: {candidates}. Available: {list(df.columns)}")


def _derive_expected_labels(df: pd.DataFrame, dataset: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (expected_label, expected_app, expected_rule)
    Derived from file_names/capture_id deterministically.
    """
    src_col = _pick_id_col(df, ["file_names", "capture_id", "capture_name"])

    def _one(x: Any):
        s = str(x)
        if dataset == "vnat":
            lab = vnat_label_from_filename(s)
            return lab.label, lab.app, lab.rule
        if dataset == "iscx":
            lab = iscx_label_from_filename(s)
            return lab.label, lab.app, lab.rule
        raise ValueError(f"Unsupported dataset: {dataset}")

    tmp = df[src_col].map(_one)
    expected_label = tmp.map(lambda t: t[0]).astype("int64")
    expected_app = tmp.map(lambda t: t[1]).astype("string")
    expected_rule = tmp.map(lambda t: t[2]).astype("string")
    return expected_label, expected_app, expected_rule


def validate_labels_df(
    df: pd.DataFrame,
    dataset: str,
    require_label_col: bool = True,
    require_capture_consistency: bool = True,
) -> LabelValidationReport:
    notes: List[str] = []

    # basic columns
    flow_id_col = "flow_id" if "flow_id" in df.columns else None
    capture_id_col = _pick_id_col(df, ["capture_id", "capture_name", "file_names"])

    if require_label_col and "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the dataframe, but it is missing.")

    # duplicates
    duplicate_flow_ids = 0
    if flow_id_col:
        duplicate_flow_ids = int(df[flow_id_col].duplicated().sum())
        if duplicate_flow_ids:
            notes.append(f"Found {duplicate_flow_ids} duplicate flow_id values.")

    # derive expected labels
    unknown_prefix_rows = 0
    try:
        exp_label, exp_app, exp_rule = _derive_expected_labels(df, dataset=dataset)
    except ValueError as e:
        # unknown prefix will surface here
        # We count unknown prefix rows by attempting safer per-row derivation.
        msg = str(e)
        notes.append(msg)
        raise

    # if label exists, compare
    if "label" in df.columns:
        bad = (df["label"].astype("int64") != exp_label.astype("int64"))
        n_bad = int(bad.sum())
        if n_bad:
            ex = df.loc[bad, [capture_id_col, "label"]].head(10)
            raise ValueError(
                f"{dataset}: label mismatch between stored df['label'] and derived label from naming. "
                f"Examples:\n{ex}"
            )

    # app consistency (optional but useful)
    if "app" in df.columns:
        bad_app = (df["app"].astype(str) != exp_app.astype(str))
        n_bad_app = int(bad_app.sum())
        if n_bad_app:
            notes.append(f"{dataset}: {n_bad_app} rows have app != derived_app (keeping df['app'] as-is).")

    # capture-level label consistency
    mixed_label_captures = 0
    if require_capture_consistency:
        by_cap = df.groupby(capture_id_col)["label"].nunique() if "label" in df.columns else exp_label.groupby(df[capture_id_col]).nunique()
        mixed_label_captures = int((by_cap > 1).sum())
        if mixed_label_captures:
            ex_caps = by_cap[by_cap > 1].head(10).to_dict()
            raise ValueError(f"{dataset}: captures with mixed labels detected: {ex_caps}")

    # counts
    label_counts = df["label"].value_counts().to_dict() if "label" in df.columns else exp_label.value_counts().to_dict()

    return LabelValidationReport(
        dataset=dataset,
        rows=int(len(df)),
        unique_captures=int(df[capture_id_col].astype(str).nunique()),
        label_counts={int(k): int(v) for k, v in label_counts.items()},
        unknown_prefix_rows=unknown_prefix_rows,
        mixed_label_captures=mixed_label_captures,
        duplicate_flow_ids=duplicate_flow_ids,
        notes=notes,
    )


def validate_splits_against_flows(
    flows_df: pd.DataFrame,
    train_list: Path,
    val_list: Path,
    test_list: Path,
) -> Dict[str, Any]:
    """
    Label-independent split sanity:
      - no overlap
      - all capture IDs exist
    """
    capture_id_col = _pick_id_col(flows_df, ["capture_id", "capture_name", "file_names"])
    all_caps = set(map(str, flows_df[capture_id_col].astype(str).unique()))

    def read_list(p: Path) -> List[str]:
        xs = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return xs

    tr = read_list(train_list)
    va = read_list(val_list)
    te = read_list(test_list)

    s_tr, s_va, s_te = set(tr), set(va), set(te)

    overlap = {
        "train_val": sorted(s_tr & s_va)[:20],
        "train_test": sorted(s_tr & s_te)[:20],
        "val_test": sorted(s_va & s_te)[:20],
    }
    if any(overlap[k] for k in overlap):
        raise ValueError(f"Split lists overlap (showing up to 20 each): {overlap}")

    missing = {
        "train": sorted(s_tr - all_caps)[:20],
        "val": sorted(s_va - all_caps)[:20],
        "test": sorted(s_te - all_caps)[:20],
    }
    if any(missing[k] for k in missing):
        raise ValueError(f"Split lists contain capture_ids not found in flows (up to 20 each): {missing}")

    return {
        "counts": {"train": len(tr), "val": len(va), "test": len(te)},
        "overlap_sample": overlap,
        "missing_sample": missing,
    }
