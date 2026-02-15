# src/flow/builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional


FiveTuple = Tuple[str, int, str, int, int]  # src_ip, src_port, dst_ip, dst_port, proto


@dataclass
class _FlowState:
    connection: FiveTuple
    timestamps: List[float]
    sizes: List[int]
    directions: List[int]   # 1 forward, 0 reverse
    last_ts: float


class FlowBuilder:
    """
    Builds bidirectional flows from packets.

    - First packet defines forward direction (connection)
    - Packets matching forward tuple => direction=1
    - Packets matching reverse tuple => direction=0
    - Flow expires if inactive for inactivity_timeout seconds
    """

    def __init__(self, inactivity_timeout: float = 120.0):
        self.inactivity_timeout = float(inactivity_timeout)
        self._tuple_to_flow: Dict[FiveTuple, int] = {}
        self._flows: Dict[int, _FlowState] = {}
        self._next_id: int = 0

    @staticmethod
    def _rev(t: FiveTuple) -> FiveTuple:
        return (t[2], t[3], t[0], t[1], t[4])

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
    ) -> None:
        t = float(ts)
        fwd: FiveTuple = (str(src_ip), int(src_port), str(dst_ip), int(dst_port), int(proto))
        rev: FiveTuple = self._rev(fwd)

        # FIX: don't use "or" because flow_id=0 is valid
        flow_id: Optional[int] = None
        if fwd in self._tuple_to_flow:
            flow_id = self._tuple_to_flow[fwd]
        elif rev in self._tuple_to_flow:
            flow_id = self._tuple_to_flow[rev]

        if flow_id is not None and flow_id in self._flows:
            st = self._flows[flow_id]

            # Timeout: start a new flow if too much idle time
            if (t - st.last_ts) > self.inactivity_timeout:
                self._expire_flow(flow_id)
                flow_id = None
            else:
                direction = 1 if fwd == st.connection else 0
                st.timestamps.append(t)
                st.sizes.append(int(size))
                st.directions.append(int(direction))
                st.last_ts = t
                return

        # Create new flow
        new_id = self._next_id
        self._next_id += 1

        st = _FlowState(
            connection=fwd,
            timestamps=[t],
            sizes=[int(size)],
            directions=[1],
            last_ts=t,
        )
        self._flows[new_id] = st

        # Map both directions
        self._tuple_to_flow[fwd] = new_id
        self._tuple_to_flow[self._rev(fwd)] = new_id

    def _expire_flow(self, flow_id: int) -> None:
        st = self._flows.pop(flow_id, None)
        if st is None:
            return

        fwd = st.connection
        rev = self._rev(fwd)

        if self._tuple_to_flow.get(fwd) == flow_id:
            self._tuple_to_flow.pop(fwd, None)
        if self._tuple_to_flow.get(rev) == flow_id:
            self._tuple_to_flow.pop(rev, None)

    def finalize(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for fid in sorted(self._flows.keys()):
            st = self._flows[fid]
            out.append(
                {
                    "connection": st.connection,
                    "timestamps": st.timestamps,
                    "sizes": st.sizes,
                    "directions": st.directions,
                }
            )
        return out
