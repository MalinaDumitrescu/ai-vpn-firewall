# src/eval/calibration.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional
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
) -> ProbabilityCalibrator:
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    needed = {prob_col, label_col, split_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    g = df[df[split_col].astype(str) == str(fit_split)]
    if len(g) == 0:
        raise ValueError(f"No rows found for fit_split='{fit_split}'.")

    cal = ProbabilityCalibrator(method=method)
    cal.fit(g[prob_col].to_numpy(), g[label_col].to_numpy())
    
    # Auto-populate some metadata
    cal.metadata = {
        "fit_split": str(fit_split),
        "n_samples": len(g),
        "prob_col": prob_col,
        "method": method
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
