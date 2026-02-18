from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import re


_VPN_PREFIX = "vpn_"
_NONVPN_PREFIX = "nonvpn_"


def normalize_name(s: str) -> str:
    """Lowercase, strip, normalize separators, keep basename only."""
    s = str(s).strip().lower()
    s = s.replace("\\", "/").split("/")[-1]
    return s


def derive_binary_label(name: str) -> Optional[int]:
    """
    Returns:
      1 if vpn_*
      0 if nonvpn_*
      None if unknown
    """
    s = normalize_name(name)
    if s.startswith(_VPN_PREFIX):
        return 1
    if s.startswith(_NONVPN_PREFIX):
        return 0
    return None


def derive_app(name: str) -> str:
    """
    VNAT naming you showed:
      vpn_youtube_capture2.pcap
      nonvpn_sftp_newcapture1.pcap
    We strip vpn_/nonvpn_, strip extension, then take first token.
    """
    s = normalize_name(name)
    s = re.sub(r"\.(pcap|pcapng)$", "", s)
    s = re.sub(r"^(vpn_|nonvpn_)", "", s)
    # split by "_" only; keep hyphens inside token (skype-chat)
    return s.split("_", 1)[0] if s else "unknown"


def derive_vpn_type(name: str) -> Optional[str]:
    """
    VNAT example names do NOT include tunnel tech (openvpn/wireguard/etc),
    only 'vpn_' vs 'nonvpn_' and app. So return None for now.

    If later you add files like vpn_openvpn_youtube_*.pcap, we can extend this.
    """
    _ = name
    return None


@dataclass(frozen=True)
class VnatLabel:
    label: int               # 0/1
    label_name: str          # "nonvpn"/"vpn"
    app: str                 # youtube, ssh, skype-chat, ...
    vpn_type: Optional[str]  # None for now
    rule: str                # audit string


def label_from_filename(file_name: str) -> VnatLabel:
    """
    Deterministic labeling from VNAT file_names / capture_id.
    Raises ValueError on unknown naming.
    """
    s = normalize_name(file_name)
    y = derive_binary_label(s)
    if y is None:
        raise ValueError(f"VNAT: cannot derive label (expected vpn_/nonvpn_ prefix): {file_name}")

    app = derive_app(s)
    vpn_type = derive_vpn_type(s)

    if y == 1:
        return VnatLabel(label=1, label_name="vpn", app=app, vpn_type=vpn_type, rule="prefix:vpn_")
    return VnatLabel(label=0, label_name="nonvpn", app=app, vpn_type=None, rule="prefix:nonvpn_")


def label_row(row: Dict[str, Any]) -> VnatLabel:
    """
    Convenience for dict-like rows.
    Prefers file_names, falls back to capture_id/capture_name.
    """
    for key in ("file_names", "capture_id", "capture_name"):
        if key in row and row[key] is not None:
            return label_from_filename(str(row[key]))
    raise ValueError("VNAT: row has no file_names/capture_id/capture_name to label from.")
