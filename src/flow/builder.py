# src/flow/builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

FiveTuple = Tuple[str, int, str, int, int]  # ip_a, port_a, ip_b, port_b, proto
Endpoint = Tuple[str, int]


@dataclass
class _FlowState:
    connection: FiveTuple            # canonical (stable) orientation A->B
    timestamps: List[float]
    sizes: List[int]
    directions: List[int]            # 1 = A->B, 0 = B->A
    last_ts: float
    closed: bool


class FlowBuilder:
    """
    Builds bidirectional flows from packets.

    - Canonical keying: A<->B always maps to the same flow key (stable orientation).
    - direction=1 if packet is A->B in canonical orientation, else 0.
    - Splits flows on inactivity timeout (finalizes old flow, starts new).
    - Optional: FIN/RST closes TCP flows early if tcp_flags is provided.
    - Optional: different inactivity timeouts for TCP and UDP.
    """

    def __init__(
        self,
        inactivity_timeout: float = 120.0,
        *,
        tcp_timeout: Optional[float] = None,
        udp_timeout: Optional[float] = None,
        close_on_fin_rst: bool = True,
    ):
        base = float(inactivity_timeout)
        self.tcp_timeout = float(tcp_timeout) if tcp_timeout is not None else base
        self.udp_timeout = float(udp_timeout) if udp_timeout is not None else base
        self.close_on_fin_rst = bool(close_on_fin_rst)

        # canonical FiveTuple -> flow_id
        self._key_to_flow: Dict[FiveTuple, int] = {}

        # active flows by id
        self._flows: Dict[int, _FlowState] = {}

        # completed flows for output
        self._done: List[_FlowState] = []

        self._next_id: int = 0

    @staticmethod
    def _endpoint(ip: str, port: int) -> Endpoint:
        return (str(ip), int(port))

    @classmethod
    def _canonicalize(
        cls,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
        proto: int,
    ) -> Tuple[FiveTuple, int]:
        """
        Returns (canonical_five_tuple, direction)

        Canonical orientation is defined by sorting endpoints:
          A = min((src_ip,src_port), (dst_ip,dst_port))
          B = max(...)

        direction=1 if packet matches A->B, else 0.
        """
        a = cls._endpoint(src_ip, src_port)
        b = cls._endpoint(dst_ip, dst_port)

        proto_i = int(proto)

        if a <= b:
            key: FiveTuple = (a[0], a[1], b[0], b[1], proto_i)
            direction = 1  # src is A
        else:
            key = (b[0], b[1], a[0], a[1], proto_i)
            direction = 0  # src is B (reverse vs canonical)
        return key, direction

    @staticmethod
    def _tcp_fin_or_rst(tcp_flags: Any) -> bool:
        """
        Accepts:
        - int bitmask (FIN=0x01, RST=0x04)
        - scapy flag objects / strings like "FA", "R", "RST"
        - None
        """
        if tcp_flags is None:
            return False

        if isinstance(tcp_flags, int):
            flags = tcp_flags
        else:
            # scapy sometimes gives flags as string-like
            s = str(tcp_flags).upper()
            # quick string test
            return ("FIN" in s) or ("RST" in s) or ("F" in s) or ("R" in s)

        FIN = 0x01
        RST = 0x04
        return bool(flags & FIN) or bool(flags & RST)

    def _timeout_for_proto(self, proto: int) -> float:
        return self.tcp_timeout if int(proto) == 6 else self.udp_timeout

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
        t = float(ts)
        proto_i = int(proto)

        key, direction = self._canonicalize(src_ip, src_port, dst_ip, dst_port, proto_i)
        flow_id = self._key_to_flow.get(key)

        # existing flow
        if flow_id is not None and flow_id in self._flows:
            st = self._flows[flow_id]

            # if already closed, finalize it and start new
            if st.closed:
                self._finalize_flow(flow_id)
                flow_id = None
            else:
                timeout = self._timeout_for_proto(proto_i)

                # inactivity split
                if (t - st.last_ts) > timeout:
                    self._finalize_flow(flow_id)
                    flow_id = None
                else:
                    st.timestamps.append(t)
                    st.sizes.append(int(size))
                    st.directions.append(int(direction))
                    st.last_ts = t

                    # optional TCP close
                    if (
                        self.close_on_fin_rst
                        and proto_i == 6
                        and self._tcp_fin_or_rst(tcp_flags)
                    ):
                        st.closed = True
                        self._finalize_flow(flow_id)
                    return

        # create new flow
        new_id = self._next_id
        self._next_id += 1

        st = _FlowState(
            connection=key,
            timestamps=[t],
            sizes=[int(size)],
            directions=[int(direction)],
            last_ts=t,
            closed=False,
        )

        self._flows[new_id] = st
        self._key_to_flow[key] = new_id

        # if first packet is FIN/RST, close immediately
        if self.close_on_fin_rst and proto_i == 6 and self._tcp_fin_or_rst(tcp_flags):
            self._finalize_flow(new_id)

    def _finalize_flow(self, flow_id: int) -> None:
        st = self._flows.pop(flow_id, None)
        if st is None:
            return

        key = st.connection
        if self._key_to_flow.get(key) == flow_id:
            self._key_to_flow.pop(key, None)

        self._done.append(st)

    def finalize(self) -> List[Dict[str, Any]]:
        """
        Finalize ALL remaining active flows and return a list of dicts.
        """
        for fid in list(self._flows.keys()):
            self._finalize_flow(fid)

        return [
            {
                "connection": st.connection,
                "timestamps": st.timestamps,
                "sizes": st.sizes,
                "directions": st.directions,
            }
            for st in self._done
        ]
