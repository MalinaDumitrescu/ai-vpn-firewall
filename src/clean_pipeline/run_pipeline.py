# src/clean_pipeline/run_pipeline.py
"""
CLEAN pipeline orchestrator -- MEMORY-SAFE STREAMING VERSION.

End-to-end:  load raw data -> extract features (streaming) -> split -> save.

CRITICAL DESIGN:
  This pipeline NEVER holds all raw packet arrays in memory at once.
  It processes flows one-by-one (or in small batches), extracts compact
  numeric features immediately, and discards the raw arrays.

  Peak memory usage ≈ O(one_json_flow + accumulated_feature_rows).
  On an 8 GB machine, this should use ~1-2 GB instead of 20+ GB.

Usage:
    python -m src.clean_pipeline.run_pipeline
    python -m src.clean_pipeline.run_pipeline --config configs/clean_pipeline.yaml
    python -m src.clean_pipeline.run_pipeline --family safe_core_plus_temporal
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.clean_pipeline.config import (
    CleanPipelineConfig,
    default_config,
    load_clean_config,
)
from src.clean_pipeline.feature_extractor import extract_flow_features
from src.clean_pipeline.feature_families import (
    get_family,
    get_family_safety,
    family_has_risky_features,
    FeatureSafety,
)
from src.clean_pipeline.splitter import (
    CleanSplitConfig,
    make_clean_split,
    save_split_manifest,
)


# ------------------------------------------------------
# Streaming feature extraction helpers
# ------------------------------------------------------

def _extract_one_flow(
    flow_dict: Dict[str, Any],
    family_cols: set,
    max_packets: int,
) -> Optional[Dict[str, float]]:
    """
    Extract features from a single flow dict (with timestamps/sizes/directions).
    Returns compact feature dict or None if flow is too short.
    """
    ts = flow_dict["timestamps"]
    sz = flow_dict["sizes"]
    dr = flow_dict["directions"]

    n = min(len(ts), len(sz), len(dr))
    if n < 3:
        return None

    feat = extract_flow_features(
        np.asarray(ts[:n], dtype=np.float64),
        np.asarray(sz[:n], dtype=np.float64),
        np.asarray(dr[:n], dtype=np.int32),
        max_packets=max_packets,
    )

    # Keep only family features
    return {k: v for k, v in feat.items() if k in family_cols}


def _process_vnat_streaming(
    h5_path: Path,
    family_cols: set,
    max_packets: int,
    min_packets: int,
) -> List[Dict]:
    """Process VNAT dataset with streaming: load chunk -> extract features -> discard."""
    from src.clean_pipeline.vnat_loader import iter_vnat_flows

    rows: List[Dict] = []
    count = 0
    vpn = 0

    for flow in iter_vnat_flows(h5_path, min_packets=min_packets):
        feat = _extract_one_flow(flow, family_cols, max_packets)
        if feat is None:
            continue

        feat["flow_id"] = flow["flow_id"]
        feat["capture_id"] = flow["capture_id"]
        feat["source_file"] = flow.get("source_file", "")
        feat["dataset"] = "vnat"
        feat["label"] = int(flow["label"])
        feat["app"] = flow.get("app", "")

        rows.append(feat)
        count += 1
        if flow["label"] == 1:
            vpn += 1

    gc.collect()
    print(f"  VNAT: {count} flows extracted (VPN={vpn}, nonVPN={count-vpn})")
    return rows


def _process_iscx_streaming(
    parquet_path: Path,
    family_cols: set,
    max_packets: int,
    min_packets: int,
) -> List[Dict]:
    """Process ISCX dataset with streaming feature extraction."""
    from src.clean_pipeline.iscx_loader import iter_iscx_flows

    rows: List[Dict] = []
    count = 0
    vpn = 0

    for flow in iter_iscx_flows(parquet_path, min_packets=min_packets):
        feat = _extract_one_flow(flow, family_cols, max_packets)
        if feat is None:
            continue

        feat["flow_id"] = flow["flow_id"]
        feat["capture_id"] = flow["capture_id"]
        feat["source_file"] = flow.get("source_file", "")
        feat["dataset"] = "iscx"
        feat["label"] = int(flow["label"])
        feat["app"] = flow.get("app", "")

        rows.append(feat)
        count += 1
        if flow["label"] == 1:
            vpn += 1

    gc.collect()
    print(f"  ISCX: {count} flows extracted (VPN={vpn}, nonVPN={count-vpn})")
    return rows


def _process_usbvpn_streaming(
    raw_dir: Path,
    family_cols: set,
    max_packets: int,
    min_packets: int,
) -> List[Dict]:
    """
    Process USBVPN with streaming JSON parsing + immediate feature extraction.

    This is the most memory-critical path because USBVPN raw JSONs total ~19 GB.
    Using ijson, we parse ONE flow at a time from each JSON file, extract features
    immediately, and discard the raw packet arrays.

    Peak memory: O(accumulated_feature_rows) -- no raw arrays kept.
    """
    from src.clean_pipeline.usbvpn_parser import iter_usbvpn_all_files

    rows: List[Dict] = []
    count = 0
    vpn = 0

    for flow, meta in iter_usbvpn_all_files(raw_dir, min_packets=min_packets):
        feat = _extract_one_flow(flow, family_cols, max_packets)
        if feat is None:
            continue

        feat["flow_id"] = meta["flow_id"]
        feat["capture_id"] = meta["capture_id"]
        feat["source_file"] = meta.get("source_file", "")
        feat["dataset"] = "usbvpn"
        feat["label"] = int(meta["label"])
        feat["app"] = meta.get("app", "")

        rows.append(feat)
        count += 1
        if meta["label"] == 1:
            vpn += 1

        # Periodic GC for very large file processing
        if count % 10000 == 0:
            gc.collect()

    gc.collect()
    print(f"  USBVPN: {count} flows extracted (VPN={vpn}, nonVPN={count-vpn})")
    return rows


# ------------------------------------------------------
# Main pipeline
# ------------------------------------------------------

def run_clean_pipeline(
    cfg: Optional[CleanPipelineConfig] = None,
    *,
    save_flows: bool = True,
    save_features: bool = True,
    save_splits: bool = True,
) -> pd.DataFrame:
    """
    Execute the full CLEAN pipeline in memory-safe streaming mode.

    Instead of loading all raw data -> merge -> extract, this pipeline:
    1. Processes each dataset independently
    2. Extracts features from raw arrays immediately (one flow at a time)
    3. Discards raw arrays after feature extraction
    4. Accumulates only compact feature rows (numbers, not arrays)
    """
    if cfg is None:
        cfg = default_config()

    t0 = time.time()
    family_cols = set(get_family(cfg.feature_family))
    family_list = list(get_family(cfg.feature_family))

    print("=" * 70)
    print("CLEAN PIPELINE -- MEMORY-SAFE STREAMING MODE")
    print(f"  Feature family: {cfg.feature_family} ({len(family_cols)} features)")
    print(f"  Window: max_packets={cfg.max_packets}, min_packets={cfg.min_packets}")
    print(f"  Seed: {cfg.seed}")
    print(f"  Output: {cfg.output_dir}")
    print("=" * 70)

    # Warn about risky features
    if family_has_risky_features(cfg.feature_family):
        safety = get_family_safety(cfg.feature_family)
        risky = [f for f, s in safety.items() if s == FeatureSafety.SEMANTICALLY_RISKY]
        print(f"\n[WARN]  Feature family '{cfg.feature_family}' contains "
              f"{len(risky)} SEMANTICALLY_RISKY features: {risky}")
        print("   Direction semantics differ across datasets -- use with caution.\n")

    # --------------------------------------
    # Step 1: Process each dataset independently
    # --------------------------------------
    all_rows: List[Dict] = []

    # --- VNAT ---
    if cfg.vnat_h5 is not None and cfg.vnat_h5.exists():
        print(f"\n[1/3] Processing VNAT (streaming)...")
        vnat_rows = _process_vnat_streaming(
            cfg.vnat_h5, family_cols, cfg.max_packets, cfg.min_packets
        )
        all_rows.extend(vnat_rows)
        del vnat_rows
        gc.collect()
    else:
        print("\n[1/3] VNAT: skipped (path not provided or missing)")

    # --- ISCX ---
    if cfg.iscx_parquet is not None and cfg.iscx_parquet.exists():
        print(f"\n[2/3] Processing ISCX (streaming)...")
        iscx_rows = _process_iscx_streaming(
            cfg.iscx_parquet, family_cols, cfg.max_packets, cfg.min_packets
        )
        all_rows.extend(iscx_rows)
        del iscx_rows
        gc.collect()
    else:
        print("\n[2/3] ISCX: skipped (path not provided or missing)")

    # --- USBVPN ---
    if cfg.usbvpn_raw_dir is not None and cfg.usbvpn_raw_dir.exists():
        print(f"\n[3/3] Processing USBVPN (streaming with ijson)...")
        usbvpn_rows = _process_usbvpn_streaming(
            cfg.usbvpn_raw_dir, family_cols, cfg.max_packets, cfg.min_packets
        )
        all_rows.extend(usbvpn_rows)
        del usbvpn_rows
        gc.collect()
    else:
        print("\n[3/3] USBVPN: skipped (path not provided or missing)")

    if not all_rows:
        raise ValueError("No flows extracted from any dataset. Check paths.")

    # --------------------------------------
    # Step 2: Build feature DataFrame
    # --------------------------------------
    print(f"\nBuilding feature DataFrame from {len(all_rows)} extracted flows...")
    features = pd.DataFrame(all_rows)
    del all_rows
    gc.collect()

    # Verify columns
    meta_cols = ["flow_id", "capture_id", "dataset", "label"]
    optional_meta = ["source_file", "app"]
    for col in optional_meta:
        if col in features.columns:
            meta_cols.append(col)

    available_feats = [c for c in family_list if c in features.columns]
    missing_feats = [c for c in family_list if c not in features.columns]
    if missing_feats:
        print(f"WARNING: {len(missing_feats)} family features not computed: {missing_feats}")

    features = features[meta_cols + available_feats].copy()

    for c in available_feats:
        features[c] = pd.to_numeric(features[c], errors="coerce").fillna(0.0)
    features = features.replace([np.inf, -np.inf], 0.0)

    n_dup = features["flow_id"].duplicated().sum()
    if n_dup > 0:
        print(f"WARNING: {n_dup} duplicate flow_ids -- making unique")
        features["flow_id"] = features["flow_id"] + "_" + features.index.astype(str)

    print(f"\n{'='*60}")
    print(f"EXTRACTED: {len(features)} flows x {len(available_feats)} features")
    for ds in sorted(features["dataset"].unique()):
        sub = features[features["dataset"] == ds]
        print(f"  {ds}: {len(sub)} flows "
              f"(VPN={int((sub['label']==1).sum())}, "
              f"nonVPN={int((sub['label']==0).sum())})")
    print(f"{'='*60}")

    # --------------------------------------
    # Step 3: Save flow metadata
    # --------------------------------------
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if save_flows:
        flows_dir = cfg.output_dir / "flows"
        flows_dir.mkdir(parents=True, exist_ok=True)
        flows_path = flows_dir / "merged_flows_meta.parquet"
        meta_df = features[["flow_id", "capture_id", "dataset", "label"]].copy()
        if "source_file" in features.columns:
            meta_df["source_file"] = features["source_file"]
        if "app" in features.columns:
            meta_df["app"] = features["app"]
        meta_df.to_parquet(flows_path, index=False)
        print(f"  Saved flow metadata -> {flows_path}")
        del meta_df
        gc.collect()

    # --------------------------------------
    # Step 4: Split
    # --------------------------------------
    print("\nCreating capture-level splits...")
    split_cfg = CleanSplitConfig(
        seed=cfg.seed,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        min_captures_per_class_per_split=cfg.min_captures_per_class_per_split,
        splitter_version=cfg.splitter_version,
    )
    features = make_clean_split(features, split_cfg)

    if save_splits:
        save_split_manifest(
            features,
            cfg.splits_dir,
            pipeline_config={
                "feature_family": cfg.feature_family,
                "max_packets": cfg.max_packets,
                "min_packets": cfg.min_packets,
                "seed": cfg.seed,
                "train_ratio": cfg.train_ratio,
                "val_ratio": cfg.val_ratio,
                "test_ratio": cfg.test_ratio,
                "splitter_version": cfg.splitter_version,
            },
        )

    # --------------------------------------
    # Step 5: Save features
    # --------------------------------------
    print("\nSaving artifacts...")

    if save_features:
        feat_path = cfg.output_dir / "features.parquet"
        features.to_parquet(feat_path, index=False)
        print(f"  Saved features -> {feat_path}")

        for split in ("train", "val", "test"):
            sub = features[features["split"] == split]
            split_path = cfg.output_dir / f"features_{split}.parquet"
            sub.to_parquet(split_path, index=False)
            print(f"  Saved {split} -> {split_path} ({len(sub)} rows)")

    # Save run metadata
    run_metadata = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(time.time() - t0, 1),
        "pipeline_mode": "streaming_memory_safe",
        "config": {
            "feature_family": cfg.feature_family,
            "max_packets": cfg.max_packets,
            "min_packets": cfg.min_packets,
            "seed": cfg.seed,
        },
        "datasets": {},
        "total_flows": int(len(features)),
        "feature_count": len(available_feats),
        "feature_names": available_feats,
        "metadata_columns": meta_cols,
        "splits": {},
    }

    for ds in sorted(features["dataset"].unique()):
        sub = features[features["dataset"] == ds]
        run_metadata["datasets"][ds] = {
            "n_flows": int(len(sub)),
            "vpn": int((sub["label"] == 1).sum()),
            "nonvpn": int((sub["label"] == 0).sum()),
            "n_captures": int(sub["capture_id"].nunique()),
        }

    for split in ("train", "val", "test"):
        sub = features[features["split"] == split]
        run_metadata["splits"][split] = {
            "n_flows": int(len(sub)),
            "vpn": int((sub["label"] == 1).sum()),
            "nonvpn": int((sub["label"] == 0).sum()),
            "n_captures": int(sub["capture_id"].nunique()),
        }

    meta_path = cfg.output_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"CLEAN PIPELINE -- Complete in {elapsed:.1f}s")
    print(f"  {len(features)} flows x {len(available_feats)} features")
    print(f"  Artifacts: {cfg.output_dir}")
    print(f"  Peak memory: feature DataFrame only (no raw packet arrays)")
    print(f"{'='*70}")

    return features


def main():
    parser = argparse.ArgumentParser(description="Run the CLEAN feature pipeline")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to clean pipeline YAML config"
    )
    parser.add_argument(
        "--family", type=str, default=None,
        help="Override feature family (e.g. safe_core_10, direction_invariant_augmented)"
    )
    parser.add_argument(
        "--max-packets", type=int, default=None,
        help="Override max packets per flow window"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save artifacts (dry run)"
    )
    args = parser.parse_args()

    if args.config:
        cfg = load_clean_config(Path(args.config))
    else:
        cfg = default_config()

    if args.family:
        cfg.feature_family = args.family
    if args.max_packets:
        cfg.max_packets = args.max_packets

    save = not args.no_save
    run_clean_pipeline(
        cfg,
        save_flows=save,
        save_features=save,
        save_splits=save,
    )


if __name__ == "__main__":
    main()


