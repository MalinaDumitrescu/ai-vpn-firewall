import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils.paths import load_paths
from src.utils.logging import setup_logger
from src.features.extract import load_feature_config, extract_features_from_flows
from src.splits.io import load_splits

logger = setup_logger()

USBVPN_METADATA_COLS = ["source_file", "source_capture_id"]


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
            cap_to_split[str(cid)] = split_name

    out = df.copy()
    out["capture_id"] = out["capture_id"].astype(str)
    out[split_col] = out["capture_id"].map(cap_to_split)

    out = out.dropna(subset=[split_col]).copy()
    out[split_col] = out[split_col].astype(str)
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
    vnat_path = paths.data_processed_dir / "vnat" / "flows.parquet"
    if vnat_path.exists():
        vnat_flows = pd.read_parquet(vnat_path)
        vnat_feats = extract_features_from_flows(vnat_flows, cfg)
        vnat_feats["dataset"] = "vnat"
        vnat_feats = apply_split_lists(
            vnat_feats,
            paths.data_splits / "vnat_train_captures.txt",
            paths.data_splits / "vnat_val_captures.txt",
            paths.data_splits / "vnat_test_captures.txt",
        )
        all_dfs.append(vnat_feats)
    else:
        logger.warning(f"VNAT not found at {vnat_path}")

    if vnat_only:
        return all_dfs[0] if all_dfs else pd.DataFrame()

    # --- 2. ISCX ---
    logger.info("Loading ISCX (PCAP-based)...")
    iscx_path = paths.data_processed_dir / "iscx" / "flows.parquet"
    if iscx_path.exists():
        iscx_flows = pd.read_parquet(iscx_path)
        iscx_feats = extract_features_from_flows(iscx_flows, cfg)
        iscx_feats["dataset"] = "iscx"
        iscx_feats = apply_split_lists(
            iscx_feats,
            paths.data_splits / "iscx_train_captures.txt",
            paths.data_splits / "iscx_val_captures.txt",
            paths.data_splits / "iscx_test_captures.txt",
        )
        all_dfs.append(iscx_feats)
    else:
        logger.warning(f"ISCX not found at {iscx_path}")

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
        df_all = df_all[df_all["q_min_packets_ok"] == 1].copy()

    df_all["split"] = df_all["split"].astype(str)
    df_all["dataset"] = df_all["dataset"].astype(str)
    df_all["label"] = df_all["label"].astype(int)

    _validate_no_forbidden_model_metadata(df_all)

    logger.info(f"Multi-Domain Pool Created: {df_all.shape}")
    logger.info(f"Datasets: {df_all['dataset'].value_counts().to_dict()}")
    logger.info(f"Splits: {df_all['split'].value_counts().to_dict()}")

    return df_all