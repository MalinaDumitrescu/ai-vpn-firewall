# demo_firewall/predictor.py
"""
Stage 3 — Flow-level ensemble inference with calibration.

Loads the trained balanced bagging ensemble (3 families × 3 bags = 9 models),
averages within-family probabilities, combines across families,
and applies isotonic calibration with safety checks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.pipeline.feature_pipeline import FeaturePipeline
from src.pipeline.artifacts import FeatureArtifacts
from src.eval.calibration import ProbabilityCalibrator
from src.utils.logging import setup_logger

from demo_firewall.config import (
    MODEL_FAMILIES,
    ArtifactPaths,
)
from demo_firewall.errors import (
    CalibrationError,
    ModelLoadError,
)

logger = setup_logger(name="firewall.predictor")


class EnsemblePredictor:
    """
    Flow-level ensemble predictor.

    Architecture:
    - 3 model families: XGBoost, LightGBM, CatBoost
    - 3 balanced bags per family (9 models total)
    - Within-family averaging → cross-family averaging → isotonic calibration

    Returns calibrated P(VPN) for each flow.
    """

    def __init__(
        self,
        artifact_paths: ArtifactPaths,
        family_weights: Optional[Dict[str, float]] = None,
        calibration_method: str = "isotonic",
        model_backend: str = "ensemble_all",
    ):
        """
        Parameters
        ----------
        artifact_paths : ArtifactPaths
            Resolved paths to all model artifacts.
        family_weights : dict or None
            Weights for combining family probabilities.
            Default: equal weights (1/3 each).
        calibration_method : str
            "isotonic" (default and recommended), "platt", or "none".
        model_backend : str
            Which model families to use for inference.
            "ensemble_all" (default): all 3 families (9 models).
            "xgb_only": XGBoost family only (3 models).
            "lgbm_only": LightGBM family only (3 models).
            "cat_only": CatBoost family only (3 models).
        """
        _VALID_BACKENDS = {"ensemble_all", "xgb_only", "lgbm_only", "cat_only"}
        if model_backend not in _VALID_BACKENDS:
            raise ValueError(
                f"Invalid model_backend='{model_backend}'. "
                f"Must be one of {sorted(_VALID_BACKENDS)}"
            )

        self.artifact_paths = artifact_paths
        self.calibration_method = calibration_method
        self.model_backend = model_backend

        # Determine active families from backend selection
        _BACKEND_FAMILIES = {
            "ensemble_all": list(MODEL_FAMILIES),
            "xgb_only": ["xgb"],
            "lgbm_only": ["lgbm"],
            "cat_only": ["cat"],
        }
        self._active_families = _BACKEND_FAMILIES[model_backend]

        # Default: equal family weights
        if family_weights is None:
            family_weights = {f: 1.0 / len(self._active_families) for f in self._active_families}
        total_w = sum(family_weights.values())
        self.family_weights = {k: v / total_w for k, v in family_weights.items()
                               if k in self._active_families}

        # Loaded artifacts (populated by .load())
        self._models: Dict[str, List[Any]] = {}       # {family: [model0, model1, model2]}
        self._pipeline: Optional[FeaturePipeline] = None
        self._calibrator: Optional[ProbabilityCalibrator] = None
        self._feature_names: Optional[List[str]] = None
        self._loaded = False

    def load(self) -> "EnsemblePredictor":
        """
        Load all model artifacts from disk.

        Raises
        ------
        ModelLoadError
            If any required artifact is missing or corrupted.
        """
        # Validate all paths exist
        missing = self.artifact_paths.validate()
        if missing:
            raise ModelLoadError(
                f"Missing {len(missing)} required artifact(s): {missing[:5]}"
            )

        # Load the 9 models
        model_paths = self.artifact_paths.model_paths
        for family in MODEL_FAMILIES:
            if family not in self._active_families:
                continue  # Skip inactive families
            self._models[family] = []
            for bag_path in model_paths[family]:
                try:
                    model = joblib.load(bag_path)
                    self._models[family].append(model)
                    logger.debug(f"Loaded {bag_path.name}")
                except Exception as e:
                    raise ModelLoadError(
                        f"Failed to load model {bag_path}: {e}"
                    ) from e

        # Load feature pipeline
        try:
            art = FeatureArtifacts(
                feature_columns_json=self.artifact_paths.feature_columns_json,
                scaler_pkl=self.artifact_paths.scaler_pkl,
                feature_config_hash_txt=self.artifact_paths.features_dir / "feature_config_hash.txt",
            )
            self._pipeline = FeaturePipeline.load(art)
            self._feature_names = self._pipeline.model_feature_names()
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load feature pipeline: {e}"
            ) from e

        # Load calibrator
        if self.calibration_method != "none":
            cal_path = (
                self.artifact_paths.isotonic_calibrator_path
                if self.calibration_method == "isotonic"
                else self.artifact_paths.platt_calibrator_path
            )
            if cal_path.exists():
                try:
                    self._calibrator = ProbabilityCalibrator.load(cal_path)
                    logger.info(f"Loaded {self.calibration_method} calibrator (dict format)")
                except (AttributeError, KeyError, TypeError):
                    # Fallback: the pickle may contain a raw sklearn model
                    # (e.g., IsotonicRegression) saved directly, not wrapped
                    try:
                        raw_model = joblib.load(cal_path)
                        self._calibrator = ProbabilityCalibrator(
                            method=self.calibration_method,
                            metadata={"loaded_from": str(cal_path), "format": "raw_sklearn"},
                        )
                        self._calibrator.model = raw_model
                        logger.info(
                            f"Loaded {self.calibration_method} calibrator "
                            f"(raw sklearn {type(raw_model).__name__})"
                        )
                    except Exception as e2:
                        raise CalibrationError(
                            f"Failed to load calibrator from {cal_path}: {e2}"
                        ) from e2
                except Exception as e:
                    raise CalibrationError(
                        f"Failed to load calibrator from {cal_path}: {e}"
                    ) from e
            else:
                logger.warning(
                    f"Calibrator not found at {cal_path}. "
                    "Falling back to uncalibrated probabilities."
                )
                self._calibrator = None

        n_models = sum(len(v) for v in self._models.values())
        logger.info(
            f"Ensemble loaded: {n_models} models across "
            f"{len(self._models)} families, "
            f"calibration={self.calibration_method}"
        )
        self._loaded = True
        return self

    def predict_flow(
        self,
        df_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict P(VPN) for each flow in the feature DataFrame.

        Parameters
        ----------
        df_features : pd.DataFrame
            Flow-level features from FlowTracker. Must contain all
            COMPACT_FEATURES plus metadata columns.

        Returns
        -------
        pd.DataFrame
            Original metadata + prob_raw, prob_cal, and per-family probs.
        """
        if not self._loaded:
            raise RuntimeError("Call .load() before .predict_flow()")

        # Transform features through the pipeline
        df_transformed = self._pipeline.transform(df_features, strict=False)
        X = df_transformed[self._feature_names].to_numpy(dtype=float)

        # Predict with each family
        family_probs: Dict[str, np.ndarray] = {}
        for family in MODEL_FAMILIES:
            if family not in self._active_families:
                continue  # Skip inactive families
            bag_probs = []
            for model in self._models[family]:
                try:
                    p = model.predict_proba(X)[:, 1]
                except Exception:
                    # Fallback for models that don't have predict_proba
                    p = model.predict(X).astype(float)
                bag_probs.append(p)

            # Within-family average
            family_probs[family] = np.mean(bag_probs, axis=0)

        # Cross-family weighted average
        prob_raw = np.zeros(len(X), dtype=float)
        for family, weight in self.family_weights.items():
            if family in family_probs:
                prob_raw += weight * family_probs[family]

        # Clip to [0, 1]
        prob_raw = np.clip(prob_raw, 0.0, 1.0)

        # Calibrate
        prob_cal = self._calibrate(prob_raw)

        # Build output DataFrame
        out = df_features[["flow_id", "capture_id", "label"]].copy()
        out["prob_raw"] = prob_raw
        out["prob_cal"] = prob_cal

        for family in MODEL_FAMILIES:
            out[f"prob_{family}"] = family_probs.get(family, np.zeros(len(X)))

        out["calibration_method"] = self.calibration_method
        out["confidence_margin"] = np.abs(prob_cal - 0.5) * 2.0

        return out

    def _calibrate(self, prob_raw: np.ndarray) -> np.ndarray:
        """
        Apply calibration with safety checks.

        Falls back to raw probabilities if calibration fails.
        """
        if self._calibrator is None or self.calibration_method == "none":
            return prob_raw.copy()

        try:
            prob_cal = self._calibrator.predict(prob_raw)
        except Exception as e:
            logger.warning(f"Calibration failed, using raw probabilities: {e}")
            return prob_raw.copy()

        # Safety: check output bounds
        if not np.all(np.isfinite(prob_cal)):
            raise CalibrationError(
                "Calibration produced non-finite values. "
                "The calibrator may be degenerate."
            )

        if np.min(prob_cal) < -0.01 or np.max(prob_cal) > 1.01:
            raise CalibrationError(
                f"Calibrated probabilities out of bounds: "
                f"[{np.min(prob_cal):.4f}, {np.max(prob_cal):.4f}]"
            )

        # Safety: check for degenerate (constant) calibration
        if prob_raw.size > 10 and np.std(prob_raw) > 0.01:
            if np.std(prob_cal) < 1e-8:
                raise CalibrationError(
                    "Calibration collapsed all probabilities to a constant. "
                    "This indicates a degenerate calibrator (likely single-class "
                    "calibration data)."
                )

        return np.clip(prob_cal, 0.0, 1.0)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def feature_names(self) -> List[str]:
        if self._feature_names is None:
            raise RuntimeError("Predictor not loaded.")
        return list(self._feature_names)

    def diagnostics(self) -> Dict[str, Any]:
        """Return model loading diagnostics."""
        return {
            "loaded": self._loaded,
            "model_backend": self.model_backend,
            "active_families": self._active_families,
            "n_families": len(self._models),
            "n_models_total": sum(len(v) for v in self._models.values()),
            "family_weights": self.family_weights,
            "calibration_method": self.calibration_method,
            "has_calibrator": self._calibrator is not None,
            "n_features": len(self._feature_names) if self._feature_names else 0,
            "feature_names": self._feature_names,
        }


