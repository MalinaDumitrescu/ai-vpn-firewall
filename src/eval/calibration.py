# src/eval/calibration.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence
import datetime

import numpy as np
import joblib

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.eval.metrics import binary_metrics, _safe_probs


CalibMethod = Literal["platt", "isotonic"]


@dataclass
class ProbabilityCalibrator:
    """
    Simple probability calibration wrapper.
    Fit on ONE split (typically val), then apply everywhere.
    """
    method: CalibMethod = "platt"
    model: Any = None  # LogisticRegression or IsotonicRegression
    metadata: Dict[str, Any] = field(default_factory=dict)

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        p_raw = _safe_probs(np.asarray(p_raw, dtype=float).reshape(-1))
        y = np.asarray(y, dtype=int).reshape(-1)

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError(
                "Calibration requires at least 2 classes, "
                f"but got only {classes.tolist()}"
            )

        if self.method == "platt":
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(p_raw.reshape(-1, 1), y)
            self.model = lr
            return self

        if self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_raw, y)
            self.model = iso
            return self

        raise ValueError(f"Unknown calibration method: {self.method}")

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Calibrator not fitted/loaded.")

        p_raw = _safe_probs(np.asarray(p_raw, dtype=float).reshape(-1))

        if self.method == "platt":
            return self.model.predict_proba(p_raw.reshape(-1, 1))[:, 1]

        if self.method == "isotonic":
            return np.asarray(self.model.predict(p_raw), dtype=float)

        raise ValueError(f"Unknown calibration method: {self.method}")

    def save(self, path: Path, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        
        meta = self.metadata.copy()
        if extra_metadata:
            meta.update(extra_metadata)
        
        # Ensure timestamp is present
        if "timestamp" not in meta:
            meta["timestamp"] = datetime.datetime.now().isoformat()

        joblib.dump({
            "method": self.method, 
            "model": self.model,
            "metadata": meta
        }, path)

    @staticmethod
    def load(path: Path) -> "ProbabilityCalibrator":
        obj = joblib.load(path)
        cal = ProbabilityCalibrator(
            method=obj.get("method", "platt"),
            metadata=obj.get("metadata", {})
        )
        cal.model = obj["model"]
        return cal


def fit_calibrator_from_df(
    df,
    *,
    prob_col: str = "p_raw",
    label_col: str = "label",
    split_col: str = "split",
    fit_split: str = "val",
    method: CalibMethod = "platt",
    fallback_splits: Optional[Sequence[str]] = ("train",),
) -> ProbabilityCalibrator:
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    needed = {prob_col, label_col, split_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    split_series = df[split_col].astype(str)

    requested = str(fit_split)
    candidates = [requested]
    if fallback_splits:
        candidates.extend([str(s) for s in fallback_splits if str(s).strip()])

    seen = set()
    ordered_candidates = []
    for s in candidates:
        if s not in seen:
            ordered_candidates.append(s)
            seen.add(s)

    def _class_counts(g_df: pd.DataFrame) -> Dict[int, int]:
        vc = g_df[label_col].astype(int).value_counts()
        out: Dict[int, int] = {}
        for k, v in vc.items():
            out[int(np.asarray(k).item())] = int(v)
        return out

    chosen_df = None
    chosen_name = None
    attempts: Dict[str, Dict[int, int]] = {}

    for s in ordered_candidates:
        g_try = df.loc[split_series == s]
        if len(g_try) == 0:
            attempts[s] = {}
            continue
        counts = _class_counts(g_try)
        attempts[s] = counts
        if len(counts) >= 2:
            chosen_df = g_try
            chosen_name = s
            break

    if chosen_df is None:
        union_splits = []
        for s in ordered_candidates:
            union_splits.append(s)
            g_try = df.loc[split_series.isin(union_splits)]
            if len(g_try) == 0:
                continue
            counts = _class_counts(g_try)
            attempts["+".join(union_splits)] = counts
            if len(counts) >= 2:
                chosen_df = g_try
                chosen_name = "+".join(union_splits)
                break

    if chosen_df is None:
        raise ValueError(
            "Calibration requires at least 2 classes in the fit data, but no candidate split/union "
            f"met that requirement. Requested='{requested}', attempts={attempts}"
        )

    cal = ProbabilityCalibrator(method=method)
    cal.fit(chosen_df[prob_col].to_numpy(), chosen_df[label_col].to_numpy())

    cal.metadata = {
        "fit_split_requested": requested,
        "fit_split_used": chosen_name,
        "fallback_used": bool(chosen_name != requested),
        "candidate_splits": ordered_candidates,
        "n_samples": len(chosen_df),
        "class_counts": _class_counts(chosen_df),
        "prob_col": prob_col,
        "method": method,
    }

    return cal


def calibration_report(
    df,
    *,
    raw_col: str = "p_raw",
    calib_col: str = "p_calib",
    label_col: str = "label",
    split_col: str = "split",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Quick before/after metrics on val/test by default (whatever is in df).
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    out: Dict[str, Any] = {"raw": {}, "calibrated": {}}
    for split in sorted(df[split_col].astype(str).unique()):
        g = df[df[split_col].astype(str) == split]
        y = g[label_col].to_numpy()

        if raw_col in g.columns:
            out["raw"][split] = binary_metrics(y, g[raw_col].to_numpy(), threshold=threshold)

        if calib_col in g.columns:
            out["calibrated"][split] = binary_metrics(y, g[calib_col].to_numpy(), threshold=threshold)

    return out
