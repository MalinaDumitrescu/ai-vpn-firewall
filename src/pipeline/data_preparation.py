import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from src.utils.paths import load_paths
from src.utils.logging import setup_logger
from src.features.extract import load_feature_config
from src.splits.io import load_splits

logger = setup_logger()

USBVPN_METADATA_COLS = ["source_file", "source_capture_id"]


def _load_pcap_features(
    flows_path: Path,
    features_path: Path,
    train_list: Path,
    val_list: Path,
    test_list: Path,
    cfg,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Load features.parquet if it already contains all COMPACT_FEATURES + a split column.
    Otherwise re-extract compact features from the raw flows.parquet, assign splits,
    and save back to features_path so the next call is fast.

    This is the correct data-loading path for PCAP-based datasets (VNAT, ISCX).
    Loading flows.parquet directly is WRONG because its timestamps/sizes/directions
    columns are raw packet arrays — they become all-zeros after numeric coercion,
    which makes every VNAT/ISCX flow look identical to the model (guaranteed overfitting).
    """
    from src.pipeline.feature_pipeline import COMPACT_FEATURES
    from src.features.extract import extract_features_from_flows

    # ── Fast path: valid features.parquet ────────────────────────────────────
    if features_path.exists():
        df = pd.read_parquet(features_path)
        missing = [c for c in COMPACT_FEATURES if c not in df.columns]
        if not missing:
            if "split" in df.columns:
                logger.info(
                    f"[{dataset_name}] Loaded features.parquet "
                    f"({len(df)} flows, splits already assigned)"
                )
                return df
            # Has features but no split column — apply splits and return
            logger.info(
                f"[{dataset_name}] features.parquet has COMPACT_FEATURES but "
                f"no split column — assigning splits now."
            )
            df = apply_split_lists(df, train_list, val_list, test_list)
            df.to_parquet(features_path, index=False)
            return df
        logger.warning(
            f"[{dataset_name}] features.parquet is stale (missing {missing}). "
            f"Re-extracting from flows.parquet."
        )

    # ── Slow path: extract from raw flows ────────────────────────────────────
    if not flows_path.exists():
        raise FileNotFoundError(
            f"[{dataset_name}] Missing both features.parquet and flows.parquet. "
            f"Expected flows at: {flows_path}"
        )

    logger.info(f"[{dataset_name}] Extracting compact features from flows.parquet ...")
    raw_flows = pd.read_parquet(flows_path)
    df_feats = extract_features_from_flows(raw_flows, cfg)

    # Merge back any metadata columns present in raw flows (app, connection_str, etc.)
    _skip = {"timestamps", "sizes", "directions",
             "capture_name", "row_id", "flow_key", "file_names",
             "packet_count", "packet_count_full", "min_packets_ok"}
    meta_cols = [
        c for c in raw_flows.columns
        if c not in df_feats.columns and c not in _skip
    ]
    if "flow_id" in raw_flows.columns and "flow_id" in df_feats.columns:
        raw_idx = raw_flows.set_index("flow_id")
        for col in meta_cols:
            if col in raw_idx.columns:
                df_feats[col] = df_feats["flow_id"].map(raw_idx[col])

    # Assign splits and cache
    df_feats = apply_split_lists(df_feats, train_list, val_list, test_list)
    df_feats.to_parquet(features_path, index=False)
    logger.info(
        f"[{dataset_name}] Saved re-extracted features to {features_path} "
        f"({len(df_feats)} flows)"
    )
    return df_feats




def apply_split_lists(
    df: pd.DataFrame,
    train_list: Path,
    val_list: Path,
    test_list: Path,
    *,
    split_col: str = "split"
) -> pd.DataFrame:
    """
    Applies canonical split lists to VNAT/ISCX based on capture_id.
    """
    splits = load_splits(train_list, val_list, test_list)

    cap_to_split = {}
    for split_name, caps in splits.items():
        for cid in caps:
            clean_cid = str(cid).replace(".pcapng", "").replace(".pcap", "").strip()
            cap_to_split[clean_cid] = split_name
    out = df.copy()
    out["temp_match_id"] = out["capture_id"].astype(str).str.replace(".pcapng", "").str.replace(".pcap", "").str.strip()
    out[split_col] = out["temp_match_id"].map(cap_to_split)

    out = out.dropna(subset=[split_col]).copy()
    out[split_col] = out[split_col].astype(str)
    out = out.drop(columns=["temp_match_id"])
    return out


def _validate_no_forbidden_model_metadata(df: pd.DataFrame) -> None:
    forbidden_present = [c for c in ["source_capture_id", "source_file"] if c in df.columns]
    if forbidden_present:
        logger.info(f"Metadata columns present for analysis only: {forbidden_present}")


def load_and_prepare_data(
    config_path: Optional[Path] = None,
    vnat_only: bool = False
) -> pd.DataFrame:
    """
    Loads VNAT, ISCX, and USBVPN into one aligned raw-feature pool.

    Important:
    - returns raw features, not pipeline-transformed features
    - preserves USBVPN metadata columns for analysis only
    - leaves feature exclusion to FeaturePipeline
    """
    paths = load_paths()
    if config_path is None:
        config_path = paths.configs_dir / "features.yaml"

    cfg = load_feature_config(config_path)
    all_dfs = []

    # --- 1. VNAT ---
    logger.info("Loading VNAT (PCAP-based)...")
    vnat_flows_path = paths.data_processed_dir / "vnat" / "flows.parquet"
    vnat_features_path = paths.data_processed_dir / "vnat" / "features.parquet"
    if vnat_flows_path.exists() or vnat_features_path.exists():
        vnat_feats = _load_pcap_features(
            flows_path=vnat_flows_path,
            features_path=vnat_features_path,
            train_list=paths.data_splits / "vnat_train_captures.txt",
            val_list=paths.data_splits / "vnat_val_captures.txt",
            test_list=paths.data_splits / "vnat_test_captures.txt",
            cfg=cfg,
            dataset_name="VNAT",
        )
        vnat_feats["dataset"] = "vnat"
        all_dfs.append(vnat_feats)
    else:
        logger.warning(f"VNAT not found at {vnat_flows_path}")

    if vnat_only:
        return all_dfs[0] if all_dfs else pd.DataFrame()

    # --- 2. ISCX ---
    logger.info("Loading ISCX (PCAP-based)...")
    iscx_flows_path = paths.data_processed_dir / "iscx" / "flows.parquet"
    iscx_features_path = paths.data_processed_dir / "iscx" / "features.parquet"
    if iscx_flows_path.exists() or iscx_features_path.exists():
        iscx_feats = _load_pcap_features(
            flows_path=iscx_flows_path,
            features_path=iscx_features_path,
            train_list=paths.data_splits / "iscx_train_captures.txt",
            val_list=paths.data_splits / "iscx_val_captures.txt",
            test_list=paths.data_splits / "iscx_test_captures.txt",
            cfg=cfg,
            dataset_name="ISCX",
        )
        iscx_feats["dataset"] = "iscx"
        all_dfs.append(iscx_feats)
    else:
        logger.warning(f"ISCX not found at {iscx_flows_path}")

    # --- 3. USBVPN ---
    logger.info("Loading USBVPN (JSON-based)...")
    usbvpn_path = paths.data_processed_dir / "usbvpn" / "flows.parquet"
    if usbvpn_path.exists():
        usb_feats = pd.read_parquet(usbvpn_path).copy()
        usb_feats["dataset"] = "usbvpn"

        if "split" not in usb_feats.columns:
            raise ValueError("USBVPN missing 'split' column. Re-run notebooks/09_usbvpn_integration.ipynb")

        usb_feats["capture_id"] = usb_feats["capture_id"].astype(str)
        usb_feats["split"] = usb_feats["split"].astype(str)

        unique_caps = usb_feats["capture_id"].nunique()
        if unique_caps < 10:
            logger.warning(
                f"USBVPN has only {unique_caps} capture groups. "
                "This is still too weak for robust evaluation."
            )

        if "source_capture_id" not in usb_feats.columns:
            logger.warning("USBVPN missing source_capture_id. Leave-one-source-file-out evaluation will not be possible.")

        all_dfs.append(usb_feats)
    else:
        logger.warning(f"USBVPN not found at {usbvpn_path}")

    if not all_dfs:
        raise ValueError("No datasets were loaded. Check processed parquet files.")

    df_all = pd.concat(all_dfs, ignore_index=True)

    if "q_min_packets_ok" in df_all.columns:
        df_all = df_all[df_all["q_min_packets_ok"].fillna(1) == 1].copy()

    df_all["split"] = df_all["split"].astype(str)
    df_all["dataset"] = df_all["dataset"].astype(str)
    df_all["label"] = df_all["label"].astype(int)

    # Dedup identical flows BEFORE training
    logger.info("Removing exact duplicate flows across feature columns...")
    
    # Identify feature columns (exclude metadata)
    exclude_cols = {"flow_id", "capture_id", "source_file", "source_capture_id", 
                   "split", "dataset", "label", "app", "connection_str"}
    
    feature_cols = [c for c in df_all.columns if c not in exclude_cols]
    
    # FIX: Ensure all numeric feature columns are properly typed for XGBoost BEFORE deduplication
    # XGBoost requires int, float, bool, or category dtypes, not object
    logger.info("Ensuring numeric dtypes for feature columns...")
    
    for col in feature_cols:
        if df_all[col].dtype == 'object':
            logger.warning(f"Converting object column '{col}' to numeric")
            # Convert to numeric, coercing errors to NaN
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
        
        # Ensure numeric columns are float64 (XGBoost compatible)
        if df_all[col].dtype in ['int64', 'int32', 'float32', 'float64']:
            df_all[col] = df_all[col].astype('float64')
    
    # Fill any NaN values that were created during conversion
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns
    df_all[numeric_cols] = df_all[numeric_cols].fillna(0.0)

    # DEBUG: Verify conversion worked
    object_feature_cols = [c for c in feature_cols if df_all[c].dtype == 'object']
    if object_feature_cols:
        logger.error(f"FAILED to convert these feature columns to numeric: {object_feature_cols}")
        for col in object_feature_cols[:3]:
            unique_vals = df_all[col].dropna().unique()[:5]
            logger.error(f"  {col} sample values: {unique_vals}")
    else:
        logger.info("✓ All feature columns successfully converted to numeric dtypes")

    # Drop duplicates WITHIN each (dataset, capture, label) group.
    # Flows from different captures that share identical feature values are
    # genuinely distinct observations and must NOT be collapsed.
    numeric_feature_cols = [c for c in feature_cols if df_all[c].dtype in ['int64', 'int32', 'float32', 'float64']]
    dedup_subset = ["dataset", "capture_id", "label"] + numeric_feature_cols
    initial_len = len(df_all)
    df_all = df_all.drop_duplicates(subset=dedup_subset, keep="first")
    final_len = len(df_all)
    
    if initial_len > final_len:
        logger.info(f"Removed {initial_len - final_len} duplicate flows ({(initial_len - final_len) / initial_len * 100:.2f}%)")

    _validate_no_forbidden_model_metadata(df_all)

    logger.info(f"Multi-Domain Pool Created: {df_all.shape}")
    logger.info(f"Datasets: {df_all['dataset'].value_counts().to_dict()}")
    logger.info(f"Splits: {df_all['split'].value_counts().to_dict()}")

    return df_all


def remove_cross_split_duplicates(
    df: pd.DataFrame,
    feature_cols: list,
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test"
) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate flows that appear across different splits (train/val/test).
    
    This is a POST-SPLIT deduplication that removes test/val samples if they are
    exact feature matches to training samples. This prevents data leakage from
    inflating test performance.
    
    Args:
        df: DataFrame with split column and feature columns
        feature_cols: List of feature column names to use for deduplication
        train_split: Name of training split (default: "train")
        val_split: Name of validation split (default: "val")
        test_split: Name of test split (default: "test")
    
    Returns:
        (df_deduped, num_removed): Deduplicated dataframe and count of removed flows
    """
    if "split" not in df.columns:
        raise ValueError("DataFrame must contain 'split' column")
    
    # Get train features as reference
    train_features = df[df["split"] == train_split][feature_cols].copy()
    train_features_set = set(map(tuple, train_features.values))
    
    initial_len = len(df)
    df_out = df.copy()
    
    # Remove val samples that match train
    if val_split in df_out["split"].unique():
        val_mask = df_out["split"] == val_split
        val_features = df_out[val_mask][feature_cols].copy()
        val_rows_to_keep = ~val_features.apply(lambda row: tuple(row) in train_features_set, axis=1)
        
        val_removed = (~val_rows_to_keep).sum()
        df_out.loc[val_mask, "row_to_keep"] = val_rows_to_keep
        
        if val_removed > 0:
            logger.info(f"Removed {val_removed} duplicate flows from {val_split} set (matching {train_split})")
    
    # Remove test samples that match train
    if test_split in df_out["split"].unique():
        test_mask = df_out["split"] == test_split
        test_features = df_out[test_mask][feature_cols].copy()
        test_rows_to_keep = ~test_features.apply(lambda row: tuple(row) in train_features_set, axis=1)
        
        test_removed = (~test_rows_to_keep).sum()
        df_out.loc[test_mask, "row_to_keep"] = test_rows_to_keep
        
        if test_removed > 0:
            logger.info(f"Removed {test_removed} duplicate flows from {test_split} set (matching {train_split})")
    
    # Keep rows where row_to_keep is True or not set (train and other splits)
    df_out["row_to_keep"] = df_out.get("row_to_keep", True)
    df_out = df_out[df_out["row_to_keep"] == True].drop(columns=["row_to_keep"])
    
    final_len = len(df_out)
    removed_count = initial_len - final_len
    
    if removed_count > 0:
        removal_pct = (removed_count / initial_len) * 100
        logger.info(f"Total cross-split duplicates removed: {removed_count} ({removal_pct:.2f}%)")
    
    return df_out, removed_count

