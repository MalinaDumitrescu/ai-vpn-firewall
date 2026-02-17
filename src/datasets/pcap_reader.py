# src/datasets/pcap_reader.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Any

try:
    from scapy.all import PcapReader
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
except ImportError as e:
    raise ImportError("Install scapy: pip install scapy") from e


def iter_packets(pcap_path: str | Path) -> Iterator[Dict[str, Any]]:
    p = Path(pcap_path)
    if not p.exists():
        raise FileNotFoundError(f"PCAP not found: {p}")

    with PcapReader(str(p)) as reader:
        for pkt in reader:
            ts = float(getattr(pkt, "time", 0.0))

            if IP in pkt:
                ip = pkt[IP]
                src_ip, dst_ip = str(ip.src), str(ip.dst)
            elif IPv6 in pkt:
                ip = pkt[IPv6]
                src_ip, dst_ip = str(ip.src), str(ip.dst)
            else:
                continue

            tcp_flags = None

            if TCP in pkt:
                l4 = pkt[TCP]
                proto = 6
                try:
                    tcp_flags = int(l4.flags)
                except Exception:
                    tcp_flags = None
            elif UDP in pkt:
                l4 = pkt[UDP]
                proto = 17
            else:
                continue

            try:
                src_port = int(l4.sport)
                dst_port = int(l4.dport)
            except Exception:
                continue

            size = int(len(pkt)) if pkt is not None else 0

            yield {
                "ts": ts,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "proto": proto,
                "size": size,
                "tcp_flags": tcp_flags,
            }
