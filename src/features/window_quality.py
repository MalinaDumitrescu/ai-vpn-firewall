from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class WindowQuality:
    packet_count: int
    min_packets_ok: bool


def quality_features(q: WindowQuality) -> Dict[str, float]:
    """
    Quality indicators are important for:
    - early detection
    - auditability (you can show how many packets you actually had)
    """
    return {
        "q_packet_count": float(q.packet_count),
        "q_min_packets_ok": 1.0 if q.min_packets_ok else 0.0,
    }
