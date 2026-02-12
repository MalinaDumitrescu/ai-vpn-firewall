from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class HistSpec:
    bins: List[float]
    max_value: float | None = None
    normalize: bool = True


def _to_1d_float(x: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def fixed_hist(x: Iterable[float], spec: HistSpec) -> np.ndarray:
    """
    Fixed-bin histogram with optional clipping and optional normalization.
    Returns len(bins)-1 features.
    """
    arr = _to_1d_float(x)

    if spec.max_value is not None and arr.size > 0:
        arr = np.clip(arr, 0.0, float(spec.max_value))

    counts, _ = np.histogram(arr, bins=np.asarray(spec.bins, dtype=float))

    counts = counts.astype(float)
    if spec.normalize:
        denom = counts.sum()
        if denom > 0:
            counts = counts / denom

    return counts
