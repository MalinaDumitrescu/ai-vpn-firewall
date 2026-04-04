# src/clean_pipeline/merge_datasets.py
"""
Dataset merger for the CLEAN pipeline.

Loads all three datasets (VNAT, ISCX, USBVPN) into the unified flow
schema and concatenates them into a single DataFrame.

Unified schema columns:
    flow_id       str   -- globally unique  (prefix: vnat::, iscx::, usbvpn::)
    capture_id    str   -- capture-level grouping key
    source_file   str   -- original file name
    dataset       str   -- "vnat" | "iscx" | "usbvpn"
    label         int   -- 0=nonVPN, 1=VPN
    timestamps    list[float]  -- epoch seconds per packet
    sizes         list[int]    -- absolute byte counts per packet
    directions    list[int]    -- 0 or 1 per packet
    app           str   -- application / activity label
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def merge_all_datasets(
    *,
    vnat_h5: Optional[Path] = None,
    iscx_parquet: Optional[Path] = None,
    usbvpn_raw_dir: Optional[Path] = None,
    min_packets: int = 3,
) -> pd.DataFrame:
    """
    Load and merge all available datasets into the unified flow schema.

      WARNING: This function loads ALL raw packet arrays into memory.
    On machines with < 16 GB RAM, use run_clean_pipeline() instead,
    which processes data in streaming mode.

    Pass None for any dataset to skip it.

    Parameters
    ----------
    vnat_h5 : Path or None
        Path to VNAT_Dataframe_release_1.h5
    iscx_parquet : Path or None
        Path to data/processed/iscx/flows.parquet
    usbvpn_raw_dir : Path or None
        Path to data/raw/usbvpn/
    min_packets : int
        Minimum packets per flow (applied at load time).

    Returns
    -------
    DataFrame with unified schema.
    """
    import warnings
    warnings.warn(
        "merge_all_datasets() loads ALL raw packet arrays into memory. "
        "On machines with < 16 GB RAM, this WILL crash. "
        "Use run_clean_pipeline() instead (streaming mode).",
        ResourceWarning,
        stacklevel=2,
    )
    frames: List[pd.DataFrame] = []

    if vnat_h5 is not None and vnat_h5.exists():
        from src.clean_pipeline.vnat_loader import load_vnat_raw
        df_vnat = load_vnat_raw(vnat_h5, min_packets=min_packets)
        frames.append(df_vnat)
        print(f"  VNAT: {len(df_vnat)} flows")
    else:
        print("  VNAT: skipped (path not provided or missing)")

    if iscx_parquet is not None and iscx_parquet.exists():
        from src.clean_pipeline.iscx_loader import load_iscx_from_parquet
        df_iscx = load_iscx_from_parquet(iscx_parquet, min_packets=min_packets)
        frames.append(df_iscx)
        print(f"  ISCX: {len(df_iscx)} flows")
    else:
        print("  ISCX: skipped (path not provided or missing)")

    if usbvpn_raw_dir is not None and usbvpn_raw_dir.exists():
        from src.clean_pipeline.usbvpn_parser import load_usbvpn_raw
        df_usbvpn = load_usbvpn_raw(usbvpn_raw_dir, min_packets=min_packets)
        frames.append(df_usbvpn)
        print(f"  USBVPN: {len(df_usbvpn)} flows")
    else:
        print("  USBVPN: skipped (path not provided or missing)")

    if not frames:
        raise ValueError("No datasets loaded. Check paths.")

    # Harmonize columns before concat -- USBVPN has vpn_protocol, others don't
    common_cols = [
        "flow_id", "capture_id", "source_file", "dataset",
        "label", "timestamps", "sizes", "directions", "app",
    ]

    harmonized: List[pd.DataFrame] = []
    for df in frames:
        for col in common_cols:
            if col not in df.columns:
                df[col] = ""
        harmonized.append(df[common_cols].copy())

    merged = pd.concat(harmonized, ignore_index=True)

    # Validate uniqueness of flow_id
    n_dup = merged["flow_id"].duplicated().sum()
    if n_dup > 0:
        print(f"WARNING: {n_dup} duplicate flow_ids detected -- making unique")
        merged["flow_id"] = merged["flow_id"] + "_" + merged.index.astype(str)

    # Summary
    print(f"\n{'='*60}")
    print(f"MERGED: {len(merged)} total flows")
    for ds in merged["dataset"].unique():
        sub = merged[merged["dataset"] == ds]
        print(f"  {ds}: {len(sub)} flows "
              f"(VPN={int((sub['label']==1).sum())}, "
              f"nonVPN={int((sub['label']==0).sum())})")
    print(f"{'='*60}\n")

    return merged


def _resolve_default_paths() -> Dict[str, Optional[Path]]:
    """Resolve default dataset paths relative to repo root."""
    # Try to find repo root
    here = Path(__file__).resolve().parent
    for candidate in [here.parent.parent, Path.cwd()]:
        if (candidate / "pyproject.toml").exists():
            root = candidate
            break
    else:
        root = Path.cwd()

    return {
        "vnat_h5": root / "data" / "raw" / "vnat" / "VNAT_Dataframe_release_1.h5",
        "iscx_parquet": root / "data" / "processed" / "iscx" / "flows.parquet",
        "usbvpn_raw_dir": root / "data" / "raw" / "usbvpn",
    }


if __name__ == "__main__":
    paths = _resolve_default_paths()
    merged = merge_all_datasets(
        vnat_h5=paths["vnat_h5"],
        iscx_parquet=paths["iscx_parquet"],
        usbvpn_raw_dir=paths["usbvpn_raw_dir"],
        min_packets=3,
    )
    print(merged.groupby(["dataset", "label"]).size())
    print(f"Mean packets: {merged['sizes'].apply(len).mean():.0f}")

