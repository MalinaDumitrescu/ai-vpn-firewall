# src/deployment/adaptive_threshold.py
"""
Adaptive threshold adjustment based on a benign-score reference buffer.

In real deployment, a firewall does NOT know labels. The adaptive threshold
uses low-confidence (presumed-benign) scores from recent traffic to detect
when the score distribution has shifted, and raises the block threshold
conservatively to reduce false positives.

Key design choices:
- threshold can only go UP (more conservative) by default
- downward relaxation requires explicit opt-in
- all adaptation is explainable and logged
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np


@dataclass
class ThresholdState:
    """Snapshot of adaptive threshold state."""
    base_threshold: float
    current_threshold: float
    buffer_size: int
    buffer_count: int
    buffer_p90: Optional[float]
    buffer_max: Optional[float]
    n_adjustments: int
    frozen: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdaptiveThreshold:
    """
    Adapts the block threshold based on a rolling buffer of presumed-benign
    session scores.

    In deployment, sessions scoring below a "benign ceiling" (e.g., below the
    flag threshold) are assumed benign and added to the buffer. The adapted
    threshold is derived from the buffer's score distribution.

    Usage:
        at = AdaptiveThreshold(base_threshold=0.7447)
        # Feed presumed-benign scores:
        at.update(0.12)
        at.update(0.03)
        ...
        # Use adapted threshold:
        thr = at.current_threshold
    """

    def __init__(
        self,
        base_threshold: float,
        buffer_size: int = 200,
        safety_margin: float = 0.02,
        adaptation_rate: float = 0.1,  # EMA blending weight for new info
        allow_relaxation: bool = False,
        frozen: bool = False,
    ):
        """
        Parameters
        ----------
        base_threshold : float
            Val-calibrated block threshold (the anchor).
        buffer_size : int
            Rolling buffer capacity.
        safety_margin : float
            Added above buffer max/p90 for conservatism.
        adaptation_rate : float
            EMA weight for blending buffer-derived threshold with base.
        allow_relaxation : bool
            If False, threshold can only increase (more conservative).
            If True, threshold can decrease toward base when drift resolves.
        frozen : bool
            If True, no adaptation occurs (strict mode).
        """
        self.base_threshold = base_threshold
        self.buffer_size = buffer_size
        self.safety_margin = safety_margin
        self.adaptation_rate = adaptation_rate
        self.allow_relaxation = allow_relaxation
        self.frozen = frozen

        self._buffer: Deque[float] = deque(maxlen=buffer_size)
        self._current_threshold = base_threshold
        self._n_adjustments = 0
        self._history: List[Dict[str, Any]] = []

    def update(self, score: float) -> None:
        """
        Add a presumed-benign session score to the buffer.

        Only call this for sessions that scored BELOW the flag threshold
        (i.e., high confidence benign).
        """
        if self.frozen:
            return

        self._buffer.append(float(score))

        if len(self._buffer) < 20:
            # Not enough data to adapt yet
            return

        arr = np.array(self._buffer)
        buffer_p90 = float(np.percentile(arr, 90))
        buffer_max = float(np.max(arr))

        # Candidate threshold: buffer_max + margin, or buffer_p90 + larger margin
        candidate = max(buffer_max + self.safety_margin,
                        buffer_p90 + self.safety_margin * 2)

        # EMA blend with current
        blended = (1 - self.adaptation_rate) * self._current_threshold + \
                  self.adaptation_rate * candidate

        old_thr = self._current_threshold

        if self.allow_relaxation:
            self._current_threshold = max(blended, self.base_threshold * 0.9)
        else:
            # Only allow upward (more conservative) adjustment
            self._current_threshold = max(blended, self.base_threshold)

        if abs(self._current_threshold - old_thr) > 1e-6:
            self._n_adjustments += 1
            self._history.append({
                "adjustment": self._n_adjustments,
                "old_threshold": old_thr,
                "new_threshold": self._current_threshold,
                "buffer_p90": buffer_p90,
                "buffer_max": buffer_max,
                "candidate": candidate,
                "buffer_count": len(self._buffer),
            })

    def update_batch(self, scores: np.ndarray) -> None:
        """Add multiple presumed-benign scores."""
        for s in np.asarray(scores).flat:
            self.update(float(s))

    @property
    def current_threshold(self) -> float:
        return self._current_threshold

    @property
    def state(self) -> ThresholdState:
        arr = np.array(self._buffer) if self._buffer else np.array([])
        return ThresholdState(
            base_threshold=self.base_threshold,
            current_threshold=self._current_threshold,
            buffer_size=self.buffer_size,
            buffer_count=len(self._buffer),
            buffer_p90=float(np.percentile(arr, 90)) if len(arr) > 0 else None,
            buffer_max=float(np.max(arr)) if len(arr) > 0 else None,
            n_adjustments=self._n_adjustments,
            frozen=self.frozen,
        )

    @property
    def adjustment_history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def reset(self) -> None:
        """Reset buffer and threshold to base."""
        self._buffer.clear()
        self._current_threshold = self.base_threshold
        self._n_adjustments = 0
        self._history.clear()

    def save(self, path: Path) -> None:
        data = {
            "base_threshold": self.base_threshold,
            "current_threshold": self._current_threshold,
            "buffer_size": self.buffer_size,
            "safety_margin": self.safety_margin,
            "adaptation_rate": self.adaptation_rate,
            "allow_relaxation": self.allow_relaxation,
            "frozen": self.frozen,
            "buffer": list(self._buffer),
            "n_adjustments": self._n_adjustments,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "AdaptiveThreshold":
        with open(path) as f:
            data = json.load(f)
        at = cls(
            base_threshold=data["base_threshold"],
            buffer_size=data.get("buffer_size", 200),
            safety_margin=data.get("safety_margin", 0.02),
            adaptation_rate=data.get("adaptation_rate", 0.1),
            allow_relaxation=data.get("allow_relaxation", False),
            frozen=data.get("frozen", False),
        )
        at._current_threshold = data.get("current_threshold", at.base_threshold)
        for s in data.get("buffer", []):
            at._buffer.append(s)
        at._n_adjustments = data.get("n_adjustments", 0)
        return at

