# src/deployment/normalization.py
"""
Score normalization strategies for cross-domain threshold portability.

Provides:
- passthrough: no normalization (baseline)
- rank_norm: CDF-based rank normalization per dataset (from NB33 C2)
- z_norm: z-score normalization per dataset (from NB33 C1)

All statistics are fitted from TRAIN split only — no val/test leakage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class NormMethod(str, Enum):
    PASSTHROUGH = "passthrough"
    RANK_NORM = "rank_norm"
    Z_NORM = "z_norm"


@dataclass
class _DatasetStats:
    """Per-dataset normalization statistics fitted from train."""
    mean: float = 0.0
    std: float = 1.0
    sorted_scores: Optional[np.ndarray] = None  # for rank-norm CDF


class ScoreNormalizer:
    """
    Normalize session-aggregated scores to improve cross-domain threshold
    portability.

    Usage:
        norm = ScoreNormalizer(method=NormMethod.RANK_NORM)
        norm.fit(train_scores_by_dataset)  # dict: {dataset: array}
        normalized = norm.transform(scores, dataset="iscx")
    """

    def __init__(self, method: NormMethod = NormMethod.PASSTHROUGH):
        self.method = method
        self._stats: Dict[str, _DatasetStats] = {}
        self._global_stats = _DatasetStats()
        self._fitted = False

    def fit(self, scores_by_dataset: Dict[str, np.ndarray]) -> "ScoreNormalizer":
        """
        Fit normalization statistics from train-split scores.

        Parameters
        ----------
        scores_by_dataset : dict
            {dataset_name: array of session scores from train split}
        """
        all_scores = []
        for ds_name, scores in scores_by_dataset.items():
            s = np.asarray(scores, dtype=float)
            s = s[np.isfinite(s)]
            stats = _DatasetStats(
                mean=float(np.mean(s)) if len(s) > 0 else 0.0,
                std=float(np.std(s)) if len(s) > 1 else 1.0,
                sorted_scores=np.sort(s) if len(s) > 0 else np.array([0.0]),
            )
            if stats.std < 1e-10:
                stats.std = 1.0
            self._stats[ds_name] = stats
            all_scores.extend(s.tolist())

        all_arr = np.array(all_scores)
        self._global_stats = _DatasetStats(
            mean=float(np.mean(all_arr)) if len(all_arr) > 0 else 0.0,
            std=float(np.std(all_arr)) if len(all_arr) > 1 else 1.0,
            sorted_scores=np.sort(all_arr) if len(all_arr) > 0 else np.array([0.0]),
        )
        if self._global_stats.std < 1e-10:
            self._global_stats.std = 1.0
        self._fitted = True
        return self

    def transform(
        self,
        scores: np.ndarray,
        dataset: Optional[str] = None,
    ) -> np.ndarray:
        """
        Normalize scores using fitted statistics.

        Parameters
        ----------
        scores : array
            Raw session-aggregated scores.
        dataset : str or None
            Dataset name for per-dataset normalization.
            If None or unknown, uses global (pooled) statistics.
        """
        scores = np.asarray(scores, dtype=float)
        if not self._fitted and self.method != NormMethod.PASSTHROUGH:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        if self.method == NormMethod.PASSTHROUGH:
            return scores

        stats = self._stats.get(dataset, self._global_stats) if dataset else self._global_stats

        if self.method == NormMethod.Z_NORM:
            return (scores - stats.mean) / stats.std

        if self.method == NormMethod.RANK_NORM:
            ref = stats.sorted_scores
            if ref is None or len(ref) == 0:
                return scores
            # CDF: score -> rank in [0, 1]
            ranks = np.searchsorted(ref, scores, side="right") / len(ref)
            return np.clip(ranks, 0.0, 1.0)

        return scores

    def save(self, path: Path) -> None:
        """Save normalization artifacts to JSON."""
        data: Dict[str, Any] = {
            "method": self.method.value,
            "datasets": {},
            "global": {
                "mean": self._global_stats.mean,
                "std": self._global_stats.std,
            },
        }
        for ds_name, stats in self._stats.items():
            data["datasets"][ds_name] = {
                "mean": stats.mean,
                "std": stats.std,
                "n_reference": len(stats.sorted_scores) if stats.sorted_scores is not None else 0,
                "sorted_scores": (stats.sorted_scores.tolist()
                                  if stats.sorted_scores is not None else []),
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ScoreNormalizer":
        """Load normalization artifacts from JSON."""
        with open(path) as f:
            data = json.load(f)
        norm = cls(method=NormMethod(data["method"]))
        norm._global_stats = _DatasetStats(
            mean=data["global"]["mean"],
            std=data["global"]["std"],
            sorted_scores=None,
        )
        for ds_name, ds_data in data.get("datasets", {}).items():
            norm._stats[ds_name] = _DatasetStats(
                mean=ds_data["mean"],
                std=ds_data["std"],
                sorted_scores=np.array(ds_data.get("sorted_scores", []), dtype=float),
            )
        norm._fitted = True
        return norm

