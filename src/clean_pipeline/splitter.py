# src/clean_pipeline/splitter.py
"""
Cross-dataset capture-level splitter for the CLEAN pipeline.

Splits a merged feature DataFrame into train/val/test while preserving
capture integrity (all flows from one capture stay in the same split).

Supports:
  - Per-dataset splitting (e.g. VNAT captures split independently)
  - Stratification by label within each dataset
  - Minimum class counts per split
  - Reproducible via seed
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanSplitConfig:
    """Configuration for clean pipeline splitting."""
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    min_captures_per_class_per_split: int = 2
    min_flows_val: int = 50
    min_flows_test: int = 50


def _capture_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build capture-level summary from feature DataFrame.

    Returns DataFrame with columns:
        capture_id, dataset, label, n_flows
    """
    cap = (
        df.groupby("capture_id")
        .agg(
            dataset=("dataset", "first"),
            label=("label", "first"),
            n_flows=("label", "size"),
        )
        .reset_index()
    )
    cap["label"] = cap["label"].astype(int)
    cap["n_flows"] = cap["n_flows"].astype(int)
    return cap


def _assign_captures_greedy(
    captures: pd.DataFrame,
    train_r: float,
    val_r: float,
    seed: int,
) -> Dict[str, List[str]]:
    """
    Greedy capture assignment minimizing flow-ratio error.

    Processes captures largest-first, assigning each to the split
    that minimizes the global flow-ratio deviation.
    """
    rng = np.random.default_rng(seed)

    total_flows = int(captures["n_flows"].sum())
    targets = {
        "train": int(round(total_flows * train_r)),
        "val": int(round(total_flows * val_r)),
        "test": total_flows - int(round(total_flows * train_r)) - int(round(total_flows * val_r)),
    }

    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    flows: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    # Shuffle then sort by size (deterministic tie-breaking)
    caps = captures.sample(frac=1.0, random_state=seed).sort_values(
        "n_flows", ascending=False
    ).reset_index(drop=True)

    for _, row in caps.iterrows():
        cid = str(row["capture_id"])
        w = int(row["n_flows"])

        def score(s: str) -> int:
            return sum(
                abs((flows[k] + (w if k == s else 0)) - targets[k])
                for k in ("train", "val", "test")
            )

        best = min(score(s) for s in ("train", "val", "test"))
        cands = [s for s in ("train", "val", "test") if score(s) == best]
        chosen = cands[int(rng.integers(0, len(cands)))]

        out[chosen].append(cid)
        flows[chosen] += w

    return out


def make_clean_split(
    df: pd.DataFrame,
    cfg: CleanSplitConfig = CleanSplitConfig(),
) -> pd.DataFrame:
    """
    Assign a 'split' column to df based on capture-level splitting.

    Splits each (dataset, label) group independently to ensure
    balanced representation across datasets and classes.

    Parameters
    ----------
    df : DataFrame
        Feature DataFrame with: flow_id, capture_id, dataset, label
    cfg : CleanSplitConfig

    Returns
    -------
    df with an added 'split' column ("train" / "val" / "test")
    """
    cap = _capture_summary(df)

    all_assignments: Dict[str, str] = {}  # capture_id -> split

    # Split each (dataset, label) group separately
    for (ds, lbl), group in cap.groupby(["dataset", "label"]):
        n_caps = len(group)
        if n_caps == 0:
            continue

        # Ensure minimum captures per split
        min_per = cfg.min_captures_per_class_per_split
        if n_caps < min_per * 3:
            # Not enough captures for 3 splits -- put all in train
            print(f"  WARNING: {ds}/label={lbl} has only {n_caps} captures, "
                  f"all assigned to train")
            for cid in group["capture_id"]:
                all_assignments[str(cid)] = "train"
            continue

        # Greedy assignment
        assigns = _assign_captures_greedy(
            group,
            train_r=cfg.train_ratio,
            val_r=cfg.val_ratio,
            seed=cfg.seed + hash(f"{ds}_{lbl}") % 10000,
        )

        for split, cids in assigns.items():
            for cid in cids:
                all_assignments[cid] = split

    # Map to DataFrame
    df = df.copy()
    df["split"] = df["capture_id"].map(all_assignments)

    # Any unmapped captures (shouldn't happen) -> train
    unmapped = df["split"].isna().sum()
    if unmapped > 0:
        print(f"WARNING: {unmapped} flows with unmapped captures -> train")
        df["split"] = df["split"].fillna("train")

    # Summary
    print(f"\nSplit summary:")
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        n_caps = sub["capture_id"].nunique()
        print(f"  {split:5s}: {len(sub):6d} flows, {n_caps:4d} captures "
              f"(VPN={int((sub['label']==1).sum())}, "
              f"nonVPN={int((sub['label']==0).sum())})")

    # Per-dataset breakdown
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        breakdown = sub.groupby("split").size().to_dict()
        print(f"  {ds}: {breakdown}")

    return df


def save_split_manifest(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    pipeline_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save split lists and manifest JSON.

    Writes:
        output_dir/clean_train_captures.txt
        output_dir/clean_val_captures.txt
        output_dir/clean_test_captures.txt
        output_dir/clean_split_manifest.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "pipeline": "clean",
        "config": pipeline_config or {},
    }

    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        caps = sorted(sub["capture_id"].unique().tolist())

        # Write capture list
        list_path = output_dir / f"clean_{split}_captures.txt"
        list_path.write_text("\n".join(caps) + "\n", encoding="utf-8")

        # Per-dataset stats
        ds_stats = {}
        for ds in sorted(sub["dataset"].unique()):
            ds_sub = sub[sub["dataset"] == ds]
            ds_stats[ds] = {
                "n_captures": int(ds_sub["capture_id"].nunique()),
                "n_flows": int(len(ds_sub)),
                "vpn_flows": int((ds_sub["label"] == 1).sum()),
                "nonvpn_flows": int((ds_sub["label"] == 0).sum()),
            }

        manifest[split] = {
            "n_captures": int(len(caps)),
            "n_flows": int(len(sub)),
            "vpn_flows": int((sub["label"] == 1).sum()),
            "nonvpn_flows": int((sub["label"] == 0).sum()),
            "per_dataset": ds_stats,
            "capture_list": str(list_path.name),
        }

    # Checksums
    for split in ("train", "val", "test"):
        list_path = output_dir / f"clean_{split}_captures.txt"
        h = hashlib.sha256(list_path.read_bytes()).hexdigest()
        manifest[split]["sha256"] = h

    manifest_path = output_dir / "clean_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nSplit manifest saved to {manifest_path}")
    return manifest_path


