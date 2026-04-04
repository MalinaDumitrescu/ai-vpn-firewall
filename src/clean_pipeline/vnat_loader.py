# src/clean_pipeline/vnat_loader.py
"""
VNAT dataset loader for the CLEAN pipeline.

Reads raw HDF5 data (VNAT_Dataframe_release_1.h5) and outputs the
unified flow schema with packet-level arrays.

VNAT uses canonical IP sorting for direction:
  direction=1 → A→B where A < B lexicographically
  direction=0 → B→A

MEMORY-SAFE: Provides both streaming (iter) and batch (DataFrame) modes.
For 8 GB RAM systems, use iter_vnat_flows() to process flows one at a time.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import pandas as pd


def _parse_vnat_row(idx: int, row) -> Dict[str, Any]:
    """Parse a single VNAT HDF5 row into a unified flow dict."""
    file_name = str(row["file_names"])
    label = 1 if file_name.lower().startswith("vpn") else 0
    app = file_name.replace(".pcap", "").replace(".pcapng", "")

    timestamps = row["timestamps"]
    sizes = row["sizes"]
    directions = row["directions"]

    # Ensure they are lists
    if not isinstance(timestamps, (list, tuple)):
        timestamps = list(timestamps)
    if not isinstance(sizes, (list, tuple)):
        sizes = list(sizes)
    if not isinstance(directions, (list, tuple)):
        directions = list(directions)

    n_pkts = min(len(timestamps), len(sizes), len(directions))
    timestamps = [float(t) for t in timestamps[:n_pkts]]
    sizes = [abs(int(s)) for s in sizes[:n_pkts]]
    directions = [int(d) for d in directions[:n_pkts]]

    return {
        "flow_id": f"vnat::{idx}",
        "capture_id": file_name,
        "source_file": file_name,
        "dataset": "vnat",
        "label": label,
        "timestamps": timestamps,
        "sizes": sizes,
        "directions": directions,
        "app": app,
        "n_packets": n_pkts,
    }


def iter_vnat_flows(
    h5_path: Path,
    min_packets: int = 3,
    chunk_size: int = 500,
) -> Generator[Dict[str, Any], None, None]:
    """
    Iterate VNAT flows from HDF5.

    VNAT HDF5 is ~1 GB (FrameFixed format) — safe to read fully on 8 GB RAM.
    The memory-critical dataset is USBVPN (19 GB), not VNAT.

    Yields dicts with: flow_id, capture_id, source_file, dataset, label,
                       timestamps, sizes, directions, app, n_packets.
    """
    print(f"  VNAT: reading HDF5 (FrameFixed format)...")
    df = pd.read_hdf(str(h5_path), key="/data")
    print(f"  VNAT HDF5: {len(df)} rows, columns: {list(df.columns)}")

    yielded = 0
    for idx, row in df.iterrows():
        parsed = _parse_vnat_row(idx, row)
        if parsed["n_packets"] < min_packets:
            continue
        yield parsed
        yielded += 1

    del df
    gc.collect()
    print(f"  VNAT: yielded {yielded} flows")


def load_vnat_raw(
    h5_path: Path,
    min_packets: int = 3,
) -> pd.DataFrame:
    """
    Load VNAT HDF5 into unified flow DataFrame.

    ⚠  On machines with < 16 GB RAM, prefer iter_vnat_flows() and
    streaming feature extraction instead.

    Parameters
    ----------
    h5_path : Path
        Path to VNAT_Dataframe_release_1.h5
    min_packets : int
        Minimum packets per flow.

    Returns
    -------
    DataFrame with columns:
        flow_id, capture_id, source_file, dataset, label,
        timestamps, sizes, directions, app
    """
    rows: List[Dict[str, Any]] = []

    for flow in iter_vnat_flows(h5_path, min_packets):
        rows.append(flow)

    df = pd.DataFrame(rows)
    # Drop the helper column
    if "n_packets" in df.columns:
        df = df.drop(columns=["n_packets"])

    print(f"VNAT: loaded {len(df)} flows from HDF5 "
          f"(VPN={int((df['label'] == 1).sum())}, "
          f"nonVPN={int((df['label'] == 0).sum())})")
    return df


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/raw/vnat/VNAT_Dataframe_release_1.h5"
    )


    # Memory-safe streaming test
    count = 0
    for flow in iter_vnat_flows(path):
        count += 1
    print(f"Total flows: {count}")
