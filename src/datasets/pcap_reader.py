# src/datasets/pcap_reader.py
"""
Canonical PCAP → packet-dict reader for VPN firewall inference.

PACKET SIZE CONVENTION
----------------------
The robust9_firewall training data (USBVPN flows.parquet, ISCX clean_pipeline)
uses IP-layer packet length: ``len(pkt[IP])``.

Evidence: ``sz_all_mean`` minimum is 28 bytes in USBVPN training parquet
(20-byte IP header + 8-byte UDP header). An L2/Ethernet frame would have a
floor of ≥ 42 bytes, and transport payload alone would have a floor of 0.

If live captures use ``len(pkt)`` (the full Scapy packet = Ethernet frame),
each packet is reported ~14 bytes larger than the training convention. This
shifts the entire ``sz_all_*`` distribution out-of-distribution and collapses
robust9 probabilities → all sessions PASS, even known VPN.

Supported size modes
--------------------
- ``"ip_field"`` (DEFAULT, IP-declared total length)  = ``int(pkt[IP].len)``
- ``"ip_layer"`` (IP layer including header)          = ``len(pkt[IP])`` / ``len(pkt[IPv6])``
- ``"frame"``    (legacy / L2 Ethernet)               = ``len(pkt)``
- ``"payload"``  (transport payload only)             = ``len(pkt[TCP].payload)`` etc.

``"ip_field"`` is preferred over ``"ip_layer"`` because the IP-declared length
is robust to truncated captures (e.g. snaplen-clipped pcaps) — Scapy's
``len(pkt[IP])`` reflects bytes actually present, while ``pkt[IP].len`` is the
on-the-wire value.

For backward compatibility, ``"ip"`` is accepted as an alias for ``"ip_field"``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Any, Literal

try:
    from scapy.all import PcapReader
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
except ImportError as e:
    raise ImportError("Install scapy: pip install scapy") from e


SizeMode = Literal["ip_field", "ip_layer", "frame", "payload", "ip"]
DEFAULT_SIZE_MODE: SizeMode = "ip_field"


def _packet_size(pkt: Any, ip_layer: Any, l4: Any, mode: str) -> int:
    """Return packet size in bytes according to the requested convention."""
    # Backward-compatible alias
    if mode == "ip":
        mode = "ip_field"

    if mode == "frame":
        # Full Scapy packet length, typically Ethernet frame on live captures.
        return int(len(pkt))

    if mode == "payload":
        # Transport payload only (no headers).
        try:
            return int(len(l4.payload))
        except Exception:
            return 0

    if mode == "ip_layer":
        # Bytes actually present in the IP layer (Scapy's interpretation).
        try:
            return int(len(ip_layer))
        except Exception:
            return int(len(pkt))

    # Default: ip_field — IP-declared total length (most robust on truncated captures).
    ln = getattr(ip_layer, "len", None)
    if ln is not None:
        try:
            return int(ln)
        except Exception:
            pass
    try:
        return int(len(ip_layer))
    except Exception:
        return int(len(pkt))


def iter_packets(
    pcap_path: str | Path,
    *,
    size_mode: str = DEFAULT_SIZE_MODE,
) -> Iterator[Dict[str, Any]]:
    """
    Iterate packets from a pcap file, yielding feature-extraction dicts.

    Parameters
    ----------
    pcap_path : str or Path
        Path to .pcap / .pcapng file.
    size_mode : {"ip_field", "ip_layer", "frame", "payload"}, or alias "ip"
        Packet size definition. Default ``"ip_field"`` matches robust9 training.

    Yields
    ------
    dict with keys: ts, src_ip, dst_ip, src_port, dst_port, proto, size, tcp_flags
    """
    valid = ("ip_field", "ip_layer", "frame", "payload", "ip")
    if size_mode not in valid:
        raise ValueError(
            f"Invalid size_mode={size_mode!r}. Must be one of {valid}. "
            f"'ip' is accepted as an alias for 'ip_field'."
        )

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

            size = _packet_size(pkt, ip, l4, size_mode)

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
