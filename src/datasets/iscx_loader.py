# src/datasets/iscx_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any, List

import pandas as pd

from src.datasets.pcap_reader import iter_packets  # you must have / implement this
from src.flow.builder import FlowBuilder          # your existing builder


@dataclass(frozen=True)
class IscxConfig:
    vpn_dir: Path
    nonvpn_dir: Path


def _list_pcaps(root: Path) -> List[Path]:
    exts = {".pcap", ".pcapng"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def build_iscx_flows(*, vpn_dir: Path, nonvpn_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def process_one(pcap_path: Path, label: int) -> None:
        capture_id = pcap_path.name
        file_names = pcap_path.name  # keep same naming style as VNAT

        fb = FlowBuilder()

        for pkt in iter_packets(pcap_path):
            # pkt must contain: ts, src_ip, dst_ip, src_port, dst_port, proto, size
            fb.add_packet(
                ts=pkt["ts"],
                src_ip=pkt["src_ip"],
                src_port=pkt["src_port"],
                dst_ip=pkt["dst_ip"],
                dst_port=pkt["dst_port"],
                proto=pkt["proto"],
                size=pkt["size"],
            )

        # fb.finalize() should return list of flow dicts:
        # {
        #   "connection": (src_ip, src_port, dst_ip, dst_port, proto),
        #   "timestamps": [...],
        #   "sizes": [...],
        #   "directions": [...],
        # }
        flows = fb.finalize()

        for i, f in enumerate(flows):
            rows.append(
                {
                    "capture_id": capture_id,
                    "capture_name": capture_id,
                    "row_id": int(i),
                    "flow_id": f"{capture_id}::{i}",
                    "flow_key": _conn_to_str(f["connection"]),
                    "connection_str": _conn_to_str(f["connection"]),
                    "timestamps": f["timestamps"],
                    "sizes": f["sizes"],
                    "directions": f["directions"],
                    "file_names": file_names,
                    "app": "iscx",          # optional; ISCX doesn’t encode app same way
                    "label": int(label),
                }
            )

    for p in _list_pcaps(vpn_dir):
        process_one(p, label=1)

    for p in _list_pcaps(nonvpn_dir):
        process_one(p, label=0)

    return pd.DataFrame(rows)


def _conn_to_str(conn: Tuple[str, int, str, int, int]) -> str:
    src_ip, src_port, dst_ip, dst_port, proto = conn
    return f"{src_ip}:{int(src_port)}-{dst_ip}:{int(dst_port)}-p{int(proto)}"
