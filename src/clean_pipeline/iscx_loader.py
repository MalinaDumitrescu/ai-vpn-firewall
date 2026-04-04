# src/clean_pipeline/iscx_loader.py
"""
ISCX dataset loader for the CLEAN pipeline.

Option A: Load from pre-processed flows.parquet (has raw packet arrays).
Option B: Re-build from PCAPs using FlowBuilder (if needed).

ISCX uses canonical IP sorting for direction (same as VNAT).

MEMORY-SAFE: Provides both streaming (iter) and batch (DataFrame) modes.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, Generator, List

import pandas as pd


def iter_iscx_flows(
    flows_parquet: Path,
    min_packets: int = 3,
    chunk_size: int = 2000,
) -> Generator[Dict[str, Any], None, None]:
    """
    Iterate ISCX flows from parquet in a memory-conscious way.

    Reads the parquet in chunks using pyarrow batches to avoid loading
    the entire file into memory at once.

    Yields dicts with: flow_id, capture_id, source_file, dataset, label,
                       timestamps, sizes, directions, app.
    """
    # ISCX parquet is small (~5 MB) so we can read it all at once,
    # but we still yield one flow at a time to stay consistent with
    # the streaming API and avoid keeping raw arrays longer than needed.
    df_raw = pd.read_parquet(flows_parquet)

    required = {"timestamps", "sizes", "directions", "label"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(
            f"ISCX flows.parquet missing required columns: {missing}. "
            "Rebuild flows from PCAPs."
        )

    print(f"  ISCX parquet: {len(df_raw)} rows")
    yielded = 0

    for idx, row in df_raw.iterrows():
        timestamps = row["timestamps"]
        sizes = row["sizes"]
        directions = row["directions"]

        if not isinstance(timestamps, (list, tuple)):
            timestamps = list(timestamps)
        if not isinstance(sizes, (list, tuple)):
            sizes = list(sizes)
        if not isinstance(directions, (list, tuple)):
            directions = list(directions)

        n_pkts = min(len(timestamps), len(sizes), len(directions))
        if n_pkts < min_packets:
            continue

        timestamps = [float(t) for t in timestamps[:n_pkts]]
        sizes = [abs(int(s)) for s in sizes[:n_pkts]]
        directions = [int(d) for d in directions[:n_pkts]]

        capture_id = str(row.get("capture_id", row.get("capture_name", f"iscx_{idx}")))
        source_file = str(row.get("file_names", capture_id))
        label = int(row["label"])
        app = str(row.get("app", "iscx"))
        flow_id = str(row.get("flow_id", f"iscx::{idx}"))

        yield {
            "flow_id": flow_id,
            "capture_id": capture_id,
            "source_file": source_file,
            "dataset": "iscx",
            "label": label,
            "timestamps": timestamps,
            "sizes": sizes,
            "directions": directions,
            "app": app,
        }
        yielded += 1

    del df_raw
    gc.collect()
    print(f"  ISCX: yielded {yielded} flows (streaming)")


def load_iscx_from_parquet(
    flows_parquet: Path,
    min_packets: int = 3,
) -> pd.DataFrame:
    """
    Load ISCX from pre-processed flows.parquet into unified schema.

    Parameters
    ----------
    flows_parquet : Path
        Path to data/processed/iscx/flows.parquet
    min_packets : int
        Minimum packets per flow.

    Returns
    -------
    DataFrame with unified schema.
    """
    rows: List[Dict[str, Any]] = []
    for flow in iter_iscx_flows(flows_parquet, min_packets):
        rows.append(flow)

    df = pd.DataFrame(rows)
    print(f"ISCX: loaded {len(df)} flows from parquet "
          f"(VPN={int((df['label'] == 1).sum())}, "
          f"nonVPN={int((df['label'] == 0).sum())})")
    return df


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/processed/iscx/flows.parquet"
    )
    df = load_iscx_from_parquet(path)
    print(df.groupby("label").size())
    print(f"Mean packets per flow: {df['sizes'].apply(len).mean():.0f}")

