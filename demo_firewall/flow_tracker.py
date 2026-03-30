# demo_firewall/flow_tracker.py
"""
Stage 1 & 2 — Flow construction and feature extraction.

Converts raw pcap input (file or packet iterator) into flow-level
feature DataFrames ready for ensemble inference.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from src.flow.builder import FlowBuilder
from src.datasets.pcap_reader import iter_packets
from src.features.extract import (
    FeatureConfig,
    extract_features_from_flows,
)
from src.utils.logging import setup_logger

from demo_firewall.config import (
    COMPACT_FEATURES,
    DIRECTION_FEATURES,
    DEFAULT_WINDOW_N,
    DEFAULT_MIN_PACKETS,
    DEFAULT_EPS,
)
from demo_firewall.errors import FeatureExtractionError, InsufficientDataError

logger = setup_logger(name="firewall.flow_tracker")


class FlowTracker:
    """
    Builds bidirectional flows from packets and extracts compact features.

    Supports two input modes:
    - pcap file path  (via `from_pcap`)
    - live packet dicts (via `add_packet` + `finalize`)

    Parameters
    ----------
    capture_id : str or None
        Session/capture identifier. Auto-generated if None.
    label : int
        Ground-truth label (use -1 for unlabeled / live inference).
    window_n : int
        Maximum packets per flow window.
    min_packets : int
        Minimum packets for a valid flow.
    eps : float
        Epsilon for numerical stability.
    drop_direction_features : bool
        If True, remove direction_balance_bytes/packets from output
        (domain-fingerprinting mitigation).
    inactivity_timeout : float
        Seconds of inactivity before splitting a flow.
    """

    def __init__(
        self,
        capture_id: Optional[str] = None,
        label: int = -1,
        window_n: int = DEFAULT_WINDOW_N,
        min_packets: int = DEFAULT_MIN_PACKETS,
        eps: float = DEFAULT_EPS,
        drop_direction_features: bool = False,
        inactivity_timeout: float = 120.0,
    ):
        self.capture_id = capture_id or str(uuid.uuid4())
        self.label = label
        self.window_n = window_n
        self.min_packets = min_packets
        self.eps = eps
        self.drop_direction_features = drop_direction_features

        self._builder = FlowBuilder(
            inactivity_timeout=inactivity_timeout,
            close_on_fin_rst=True,
        )
        self._feature_cfg = FeatureConfig(
            N=window_n,
            min_packets=min_packets,
            eps=eps,
        )

    # ─────────────────────────────────────────────
    # Input: packet-by-packet
    # ─────────────────────────────────────────────

    def add_packet(
        self,
        *,
        ts: float,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
        proto: int,
        size: int,
        tcp_flags: Any = None,
    ) -> None:
        """Feed a single packet to the flow builder."""
        self._builder.add_packet(
            ts=ts,
            src_ip=src_ip,
            src_port=src_port,
            dst_ip=dst_ip,
            dst_port=dst_port,
            proto=proto,
            size=size,
            tcp_flags=tcp_flags,
        )

    def finalize(self) -> pd.DataFrame:
        """
        Finalize all active flows and extract features.

        Returns
        -------
        pd.DataFrame
            Flow-level features with columns matching COMPACT_FEATURES
            plus metadata (flow_id, capture_id, label).
        """
        raw_flows = self._builder.finalize()
        return self._extract_features(raw_flows)

    # ─────────────────────────────────────────────
    # Input: pcap file
    # ─────────────────────────────────────────────

    @classmethod
    def from_pcap(
        cls,
        pcap_path: str | Path,
        capture_id: Optional[str] = None,
        label: int = -1,
        drop_direction_features: bool = False,
        min_packets: int = DEFAULT_MIN_PACKETS,
        window_n: int = DEFAULT_WINDOW_N,
        inactivity_timeout: float = 120.0,
    ) -> pd.DataFrame:
        """
        One-shot: read a pcap file, build flows, extract features.

        Returns
        -------
        pd.DataFrame
            Flow-level feature matrix.
        """
        pcap_path = Path(pcap_path)
        cid = capture_id or pcap_path.stem

        tracker = cls(
            capture_id=cid,
            label=label,
            window_n=window_n,
            min_packets=min_packets,
            drop_direction_features=drop_direction_features,
            inactivity_timeout=inactivity_timeout,
        )

        n_packets = 0
        for pkt in iter_packets(pcap_path):
            tracker.add_packet(**pkt)
            n_packets += 1

        logger.info(f"Read {n_packets} packets from {pcap_path.name}")

        return tracker.finalize()

    # ─────────────────────────────────────────────
    # Input: packet iterator (live capture)
    # ─────────────────────────────────────────────

    @classmethod
    def from_packet_iter(
        cls,
        packets: Iterator[Dict[str, Any]],
        capture_id: Optional[str] = None,
        label: int = -1,
        drop_direction_features: bool = False,
        min_packets: int = DEFAULT_MIN_PACKETS,
        window_n: int = DEFAULT_WINDOW_N,
    ) -> pd.DataFrame:
        """
        Build flows from an arbitrary packet iterator.

        Parameters
        ----------
        packets : Iterator[Dict]
            Each dict must contain: ts, src_ip, src_port, dst_ip, dst_port,
            proto, size, and optionally tcp_flags.
        """
        tracker = cls(
            capture_id=capture_id,
            label=label,
            window_n=window_n,
            min_packets=min_packets,
            drop_direction_features=drop_direction_features,
        )

        n_packets = 0
        for pkt in packets:
            tracker.add_packet(**pkt)
            n_packets += 1

        logger.info(f"Processed {n_packets} packets from iterator")
        return tracker.finalize()

    # ─────────────────────────────────────────────
    # Internal: feature extraction
    # ─────────────────────────────────────────────

    def _extract_features(self, raw_flows: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert raw flow dicts to feature DataFrame."""
        if not raw_flows:
            raise InsufficientDataError(
                f"No flows extracted for capture '{self.capture_id}'. "
                "Check that the pcap contains valid TCP/UDP packets."
            )

        # Build flows DataFrame in the format expected by extract_features_from_flows
        flow_records = []
        for i, flow in enumerate(raw_flows):
            flow_records.append({
                "flow_id": f"{self.capture_id}_flow{i:04d}",
                "capture_id": self.capture_id,
                "label": self.label,
                "timestamps": flow["timestamps"],
                "sizes": flow["sizes"],
                "directions": flow["directions"],
            })

        df_flows = pd.DataFrame(flow_records)

        # Extract features
        try:
            df_features = extract_features_from_flows(df_flows, self._feature_cfg)
        except Exception as e:
            raise FeatureExtractionError(
                f"Feature extraction failed for capture '{self.capture_id}': {e}"
            ) from e

        # Filter flows below min_packets
        if "q_packet_count" in df_features.columns:
            valid_mask = df_features["q_packet_count"] >= self.min_packets
            n_dropped = (~valid_mask).sum()
            if n_dropped > 0:
                logger.info(
                    f"Dropped {n_dropped}/{len(df_features)} flows "
                    f"below min_packets={self.min_packets}"
                )
            df_features = df_features[valid_mask].copy()

        if len(df_features) == 0:
            raise InsufficientDataError(
                f"All flows for capture '{self.capture_id}' had fewer than "
                f"{self.min_packets} packets. Cannot make a prediction."
            )

        # Optionally drop direction features (domain fingerprinting mitigation)
        if self.drop_direction_features:
            cols_to_drop = [c for c in DIRECTION_FEATURES if c in df_features.columns]
            if cols_to_drop:
                df_features = df_features.drop(columns=cols_to_drop)
                logger.info(
                    f"Dropped direction features: {cols_to_drop} "
                    "(domain fingerprinting mitigation)"
                )

        # Validate required features exist
        expected = (
            [f for f in COMPACT_FEATURES if f not in DIRECTION_FEATURES]
            if self.drop_direction_features
            else COMPACT_FEATURES
        )
        missing = [c for c in expected if c not in df_features.columns]
        if missing:
            raise FeatureExtractionError(
                f"Missing compact features after extraction: {missing}"
            )

        # Add source metadata columns required by FeaturePipeline
        if "source_file" not in df_features.columns:
            df_features["source_file"] = "live"
        if "source_capture_id" not in df_features.columns:
            df_features["source_capture_id"] = self.capture_id

        # Validate finite values
        numeric_cols = [c for c in expected if c in df_features.columns]
        arr = df_features[numeric_cols].to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise FeatureExtractionError(
                f"Non-finite values in extracted features for capture '{self.capture_id}'"
            )

        logger.info(
            f"Extracted {len(df_features)} valid flows with "
            f"{len(expected)} features for capture '{self.capture_id}'"
        )

        return df_features

