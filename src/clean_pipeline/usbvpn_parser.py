# src/clean_pipeline/usbvpn_parser.py
"""
USBVPN raw JSON re-parser -- MEMORY-SAFE STREAMING VERSION.

Re-parses raw JSON files from data/raw/usbvpn/ using **streaming JSON
parsing** (ijson) so that multi-GB files never need to be loaded into
memory in their entirety.

USBVPN format:
  Each JSON file is a list of flow dicts:
    {
      "ip_proto": "udp" | "tcp",
      "port_src": int,
      "port_dst": int,
      "x_packets": [
        {
          "bytes": str (SIGNED! negative = reverse direction),
          "timestamp_start": str (ISO datetime),
          "timestamp_end": str (ISO datetime),
          "packets": str ("1" usually),
          "ip_header_len": str
        },
        ...
      ]
    }

Direction convention:
  - Positive bytes -> direction 1 (src -> dst)
  - Negative bytes -> direction 0 (dst -> src)
  - We take abs(bytes) for size

NOTE: This direction convention is NOT the same as VNAT/ISCX's canonical
IP sort. For SAFE features, we only use direction-INVARIANT statistics.
"""
from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------

def _parse_timestamp(ts_str: str) -> float:
    """Parse USBVPN timestamp string to epoch seconds."""
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Single-flow packet extraction
# ---------------------------------------------------------------------------

def _parse_flow_packets(
    x_packets: List[Dict[str, str]],
) -> Tuple[List[float], List[int], List[int]]:
    """
    Extract (timestamps, sizes, directions) from x_packets array.

    Returns:
        timestamps: list of epoch floats
        sizes: list of absolute byte counts
        directions: list of 0/1
    """
    timestamps: List[float] = []
    sizes: List[int] = []
    directions: List[int] = []

    for pkt in x_packets:
        raw_bytes = int(pkt.get("bytes", "0"))
        ts = _parse_timestamp(pkt.get("timestamp_start", ""))

        timestamps.append(ts)
        sizes.append(abs(raw_bytes))
        directions.append(1 if raw_bytes >= 0 else 0)

    return timestamps, sizes, directions


# ---------------------------------------------------------------------------
# Streaming JSON iterator -- MEMORY-SAFE
# ---------------------------------------------------------------------------

def _iter_flows_streaming(
    json_path: Path,
    min_packets: int = 3,
) -> Generator[Dict[str, Any], None, None]:
    """
    Yield one flow dict at a time from a USBVPN JSON file using ijson
    streaming parser.  Memory usage: O(single_flow), not O(file_size).

    Each yielded dict has keys:
        timestamps, sizes, directions, ip_proto, port_src, port_dst, n_packets
    """
    try:
        import ijson
    except ImportError:
        raise ImportError(
            "ijson is required for streaming USBVPN parsing. "
            "Install it: pip install ijson"
        )

    with open(json_path, "rb") as f:
        # ijson.items(f, "item") iterates over top-level array elements
        for flow_dict in ijson.items(f, "item"):
            x_packets = flow_dict.get("x_packets", [])
            if not x_packets:
                continue

            timestamps, sizes, directions = _parse_flow_packets(x_packets)

            if len(timestamps) < min_packets:
                continue

            yield {
                "timestamps": timestamps,
                "sizes": sizes,
                "directions": directions,
                "ip_proto": str(flow_dict.get("ip_proto", "unknown")),
                "port_src": int(flow_dict.get("port_src", 0)),
                "port_dst": int(flow_dict.get("port_dst", 0)),
                "n_packets": len(timestamps),
            }


def _iter_flows_small_file(
    json_path: Path,
    min_packets: int = 3,
) -> Generator[Dict[str, Any], None, None]:
    """
    Fallback for small files (< 50 MB): use standard json.load().
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return

    for flow_dict in data:
        x_packets = flow_dict.get("x_packets", [])
        if not x_packets:
            continue

        timestamps, sizes, directions = _parse_flow_packets(x_packets)

        if len(timestamps) < min_packets:
            continue

        yield {
            "timestamps": timestamps,
            "sizes": sizes,
            "directions": directions,
            "ip_proto": str(flow_dict.get("ip_proto", "unknown")),
            "port_src": int(flow_dict.get("port_src", 0)),
            "port_dst": int(flow_dict.get("port_dst", 0)),
            "n_packets": len(timestamps),
        }

    del data
    gc.collect()


# ---------------------------------------------------------------------------
# Public: iterate flows from one JSON file (auto-selects strategy)
# ---------------------------------------------------------------------------

_SMALL_FILE_THRESHOLD_MB = 50


def iter_usbvpn_flows(
    json_path: Path,
    min_packets: int = 3,
) -> Generator[Dict[str, Any], None, None]:
    """
    Iterate flows from a single USBVPN JSON file.

    Automatically uses streaming (ijson) for large files and standard
    json.load() for small files.

    Yields flow dicts with: timestamps, sizes, directions, ip_proto,
    port_src, port_dst, n_packets.
    """
    file_size_mb = json_path.stat().st_size / (1024 * 1024)

    if file_size_mb > _SMALL_FILE_THRESHOLD_MB:
        yield from _iter_flows_streaming(json_path, min_packets)
    else:
        yield from _iter_flows_small_file(json_path, min_packets)


# ---------------------------------------------------------------------------
# Public: iterate ALL USBVPN files with metadata
# ---------------------------------------------------------------------------

def iter_usbvpn_all_files(
    raw_dir: Path,
    min_packets: int = 3,
) -> Generator[Tuple[Dict[str, Any], Dict[str, Any]], None, None]:
    """
    Iterate over ALL USBVPN flows across all JSON files.

    Yields (flow_dict, metadata_dict) tuples where metadata contains:
        label, source_file, capture_id, app, vpn_protocol

    This is the main entry point for memory-safe USBVPN loading.
    """
    flow_counter = 0

    # --- VPN flows ---
    vpn_dir = raw_dir / "vpn"
    if vpn_dir.exists():
        for protocol_dir in sorted(vpn_dir.iterdir()):
            if not protocol_dir.is_dir():
                continue
            protocol_name = protocol_dir.name
            for json_path in sorted(protocol_dir.glob("*.json")):
                source_file = f"vpn/{protocol_name}/{json_path.name}"
                app_name = json_path.stem
                capture_id = f"usbvpn_vpn_{protocol_name}_{app_name}"
                file_mb = json_path.stat().st_size / (1024 * 1024)
                print(f"    Parsing {source_file} ({file_mb:.1f} MB)...")

                file_count = 0
                for flow in iter_usbvpn_flows(json_path, min_packets):
                    meta = {
                        "flow_id": f"usbvpn::{flow_counter}",
                        "capture_id": capture_id,
                        "source_file": source_file,
                        "dataset": "usbvpn",
                        "label": 1,
                        "app": app_name,
                        "vpn_protocol": protocol_name,
                    }
                    yield flow, meta
                    flow_counter += 1
                    file_count += 1

                print(f"      -> {file_count} flows")
                gc.collect()

    # --- Non-VPN flows ---
    nonvpn_dir = raw_dir / "nonvpn"
    if nonvpn_dir.exists():
        for json_path in sorted(nonvpn_dir.glob("*.json")):
            source_file = f"nonvpn/{json_path.name}"
            app_name = json_path.stem
            capture_id = f"usbvpn_nonvpn_{app_name}"
            file_mb = json_path.stat().st_size / (1024 * 1024)
            print(f"    Parsing {source_file} ({file_mb:.1f} MB)...")

            file_count = 0
            for flow in iter_usbvpn_flows(json_path, min_packets):
                meta = {
                    "flow_id": f"usbvpn::{flow_counter}",
                    "capture_id": capture_id,
                    "source_file": source_file,
                    "dataset": "usbvpn",
                    "label": 0,
                    "app": app_name,
                    "vpn_protocol": "none",
                }
                yield flow, meta
                flow_counter += 1
                file_count += 1

            print(f"      -> {file_count} flows")
            gc.collect()

    print(f"  USBVPN: total {flow_counter} flows parsed (streaming)")


# ---------------------------------------------------------------------------
# Legacy non-streaming loader (kept for reference, DO NOT USE on 8 GB RAM)
# ---------------------------------------------------------------------------

def parse_usbvpn_json(json_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a single USBVPN JSON file into a list of flow dicts.

    [WARN] WARNING: Uses json.load() -- do NOT use on files > 50 MB
    on machines with < 16 GB RAM.  Use iter_usbvpn_flows() instead.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of flows, got {type(data)} in {json_path}")

    flows = []
    for flow_dict in data:
        x_packets = flow_dict.get("x_packets", [])
        if not x_packets:
            continue

        timestamps, sizes, directions = _parse_flow_packets(x_packets)

        flows.append({
            "timestamps": timestamps,
            "sizes": sizes,
            "directions": directions,
            "ip_proto": flow_dict.get("ip_proto", "unknown"),
            "port_src": int(flow_dict.get("port_src", 0)),
            "port_dst": int(flow_dict.get("port_dst", 0)),
            "n_packets": len(timestamps),
        })

    return flows


def load_usbvpn_raw(
    raw_dir: Path,
    min_packets: int = 3,
) -> pd.DataFrame:
    """
    Load all USBVPN raw JSONs into a unified flow DataFrame.

    [WARN] WARNING: Loads ALL flows into memory at once.
    For machines with < 16 GB RAM, use iter_usbvpn_all_files() instead
    and extract features in streaming batches.
    """
    rows: List[Dict[str, Any]] = []

    for flow, meta in iter_usbvpn_all_files(raw_dir, min_packets):
        rows.append({
            **meta,
            "timestamps": flow["timestamps"],
            "sizes": flow["sizes"],
            "directions": flow["directions"],
        })

    df = pd.DataFrame(rows)
    print(f"USBVPN: parsed {len(df)} flows from raw JSON "
          f"(VPN={int((df['label'] == 1).sum())}, "
          f"nonVPN={int((df['label'] == 0).sum())})")
    return df


if __name__ == "__main__":
    import sys
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/usbvpn")

    # Count flows using streaming (memory-safe test)
    count = 0
    for flow, meta in iter_usbvpn_all_files(raw):
        count += 1
    print(f"Total flows: {count}")
