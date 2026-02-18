from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import re


def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("\\", "/").split("/")[-1]
    return s


def derive_binary_label(name: str) -> Optional[int]:
    s = normalize_name(name)
    if s.startswith("vpn_"):
        return 1
    if s.startswith("nonvpn_"):
        return 0
    return None


def derive_app(name: str) -> str:
    """
    ISCX examples after your prefixing:
      nonvpn_aim_chat_3a.pcap
      vpn_email2a.pcap
      vpn_bittorrent.pcap
    Strip extension and prefix, then first token.
    """
    s = normalize_name(name)
    s = re.sub(r"\.(pcap|pcapng)$", "", s)
    s = re.sub(r"^(vpn_|nonvpn_)", "", s)
    return s.split("_", 1)[0] if s else "unknown"


def derive_vpn_type(name: str) -> Optional[str]:
    """
    ISCX files don't encode tunnel tech either (at least in your structure),
    so keep None for now.
    """
    _ = name
    return None


@dataclass(frozen=True)
class IscxLabel:
    label: int
    label_name: str
    app: str
    vpn_type: Optional[str]
    rule: str


def label_from_filename(file_name: str) -> IscxLabel:
    s = normalize_name(file_name)
    y = derive_binary_label(s)
    if y is None:
        raise ValueError(
            f"ISCX: cannot derive label (expected vpn_/nonvpn_ prefix). "
            f"If you didn’t prefix file_names in processing, fix that first. Got: {file_name}"
        )

    app = derive_app(s)
    vpn_type = derive_vpn_type(s)

    if y == 1:
        return IscxLabel(label=1, label_name="vpn", app=app, vpn_type=vpn_type, rule="prefix:vpn_")
    return IscxLabel(label=0, label_name="nonvpn", app=app, vpn_type=None, rule="prefix:nonvpn_")


def label_row(row: Dict[str, Any]) -> IscxLabel:
    for key in ("file_names", "capture_id", "capture_name"):
        if key in row and row[key] is not None:
            return label_from_filename(str(row[key]))
    raise ValueError("ISCX: row has no file_names/capture_id/capture_name to label from.")
