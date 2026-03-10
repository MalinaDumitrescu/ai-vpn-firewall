import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

from src.utils.paths import load_paths
from src.utils.logging import setup_logger
from src.features.extract import load_feature_config, extract_features_from_flows
from src.splits.io import load_splits

logger = setup_logger()

def apply_split_lists(
    df: pd.DataFrame, 
    train_list: Path, 
    val_list: Path, 
    test_list: Path, 
    *, 
    split_col: str = "split"
) -> pd.DataFrame:
    """
    Applies canonical split lists to a DataFrame based on capture_id.
    """
    splits = load_splits(train_list, val_list, test_list)

    cap_to_split = {}
    for split_name, caps in splits.items():
        for cid in caps:
            cap_to_split[str(cid)] = split_name

    out = df.copy()
    out["capture_id"] = out["capture_id"].astype(str)
    out[split_col] = out["capture_id"].map(cap_to_split)

    # Filter out captures that are not in any split (optional, but good for safety)
    # or raise error if strict coverage is needed. 
    # Here we just return rows that have a split.
    out = out.dropna(subset=[split_col])
    return out

def load_and_prepare_data(
    config_path: Optional[Path] = None,
    vnat_only: bool = False
) -> pd.DataFrame:
    """
    Loads VNAT and ISCX datasets, extracts features, applies splits, 
    and returns a combined DataFrame.
    """
    paths = load_paths()
    if config_path is None:
        config_path = paths.configs_dir / "features.yaml"
    
    cfg = load_feature_config(config_path)

    # --- VNAT ---
    logger.info("Loading and processing VNAT...")
    vnat_flows = pd.read_parquet(paths.data_processed / "vnat" / "flows.parquet")
    vnat_feats = extract_features_from_flows(vnat_flows, cfg)
    vnat_feats["dataset"] = "vnat"
    
    if "q_min_packets_ok" in vnat_feats.columns:
        vnat_feats = vnat_feats[vnat_feats["q_min_packets_ok"] == 1.0].copy()

    vnat_feats = apply_split_lists(
        vnat_feats,
        paths.data_splits / "vnat_train_captures.txt",
        paths.data_splits / "vnat_val_captures.txt",
        paths.data_splits / "vnat_test_captures.txt",
    )

    if vnat_only:
        return vnat_feats

    # --- ISCX ---
    logger.info("Loading and processing ISCX...")
    iscx_flows = pd.read_parquet(paths.data_processed / "iscx" / "flows.parquet")
    iscx_feats = extract_features_from_flows(iscx_flows, cfg)
    iscx_feats["dataset"] = "iscx"

    if "q_min_packets_ok" in iscx_feats.columns:
        iscx_feats = iscx_feats[iscx_feats["q_min_packets_ok"] == 1.0].copy()

    iscx_feats = apply_split_lists(
        iscx_feats,
        paths.data_splits / "iscx_train_captures.txt",
        paths.data_splits / "iscx_val_captures.txt",
        paths.data_splits / "iscx_test_captures.txt",
    )

    # --- Combine ---
    df_all = pd.concat([vnat_feats, iscx_feats], ignore_index=True)
    df_all["split"] = df_all["split"].astype(str)
    
    logger.info(f"Data loaded. Shape: {df_all.shape}")
    logger.info(f"Split counts:\n{df_all['split'].value_counts()}")
    
    return df_all
