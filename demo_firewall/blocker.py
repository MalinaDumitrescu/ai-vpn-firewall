# demo_firewall/blocker.py
"""
Firewall orchestrator — wires together all pipeline stages.

FlowTracker → EnsemblePredictor → FirewallPolicy → Decision

This is the top-level entry point for the VPN detection firewall.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from src.utils.logging import setup_logger
from src.utils.paths import load_paths

from demo_firewall.config import (
    DeploymentMode,
    ArtifactPaths,
    default_artifact_paths,
    ThresholdConfig,
)
from demo_firewall.errors import (
    CalibrationError,
    ThresholdLeakageError,
)
from demo_firewall.flow_tracker import FlowTracker
from demo_firewall.predictor import EnsemblePredictor
from demo_firewall.policy import (
    FirewallPolicy,
    SessionDecision,
)
from demo_firewall.report import (
    compute_evaluation_metrics,
    evaluate_with_labels,
    format_report,
    save_report,
)

logger = setup_logger(name="firewall.blocker")


class FirewallBlocker:
    """
    Top-level VPN detection firewall pipeline.

    Complete inference pipeline:
        pcap/packets → flows → features → ensemble → calibration
                     → session aggregation → threshold → BLOCK/FLAG/ALLOW

    Usage
    -----
    >>> blocker = FirewallBlocker(mode=DeploymentMode.STRICT)
    >>> blocker.load()
    >>> decision = blocker.predict_pcap("capture.pcap")
    >>> print(decision.decision)  # Decision.BLOCK / FLAG / ALLOW

    For batch evaluation with labels:
    >>> results = blocker.evaluate_dataset(predictions_csv="path/to/predictions.csv")
    """

    def __init__(
        self,
        mode: DeploymentMode = DeploymentMode.STRICT,
        repo_root: Optional[Path] = None,
        artifact_paths: Optional[ArtifactPaths] = None,
        block_threshold: Optional[float] = None,
        flag_threshold: Optional[float] = None,
        drop_direction_features: bool = False,
        family_weights: Optional[Dict[str, float]] = None,
        calibration_method: str = "isotonic",
        model_backend: str = "ensemble_all",
        min_packets: int = 10,
        window_n: int = 100,
    ):
        """
        Parameters
        ----------
        mode : DeploymentMode
            STRICT (default), BALANCED, or RESEARCH.
        repo_root : Path or None
            Project root. Auto-detected if None.
        artifact_paths : ArtifactPaths or None
            Custom artifact paths. Uses defaults if None.
        block_threshold : float or None
            Override block threshold. If None, must calibrate from validation data.
        flag_threshold : float or None
            Override flag threshold.
        drop_direction_features : bool
            Remove direction_balance_bytes/packets (domain fingerprinting mitigation).
        family_weights : dict or None
            Custom weights for model families.
        calibration_method : str
            "isotonic" (recommended), "platt", or "none".
        model_backend : str
            Which model families to use: "ensemble_all", "xgb_only",
            "lgbm_only", or "cat_only".
        min_packets : int
            Minimum packets per flow.
        window_n : int
            Maximum packets per flow window.
        """
        self.mode = mode
        self.drop_direction_features = drop_direction_features
        self.min_packets = min_packets
        self.window_n = window_n

        # Resolve paths
        if repo_root is None:
            paths = load_paths()
            repo_root = paths.repo_root
        self.repo_root = repo_root

        if artifact_paths is None:
            artifact_paths = default_artifact_paths(repo_root)
        self.artifact_paths = artifact_paths

        # Initialize components
        self._predictor = EnsemblePredictor(
            artifact_paths=artifact_paths,
            family_weights=family_weights,
            calibration_method=calibration_method,
            model_backend=model_backend,
        )
        self._policy = FirewallPolicy(
            mode=mode,
            block_threshold=block_threshold,
            flag_threshold=flag_threshold,
        )
        self._loaded = False

    def load(self) -> "FirewallBlocker":
        """
        Load all model artifacts.

        Raises
        ------
        ModelLoadError
            If model files are missing or corrupted.
        CalibrationError
            If calibrator is invalid.
        """
        logger.info(f"Loading firewall pipeline (mode={self.mode.value})...")
        self._predictor.load()
        self._loaded = True
        logger.info("Firewall pipeline loaded successfully.")
        return self

    def calibrate_from_validation(
        self,
        val_predictions_path: Optional[str | Path] = None,
        val_preds_df: Optional[pd.DataFrame] = None,
        prob_col: str = "prob_iso",
    ) -> ThresholdConfig:
        """
        Calibrate thresholds from validation predictions.

        Parameters
        ----------
        val_predictions_path : str or Path
            Path to predictions CSV (from training pipeline).
        val_preds_df : pd.DataFrame
            Pre-loaded predictions. Takes priority over path.
        prob_col : str
            Probability column to use for threshold computation.

        Returns
        -------
        ThresholdConfig
        """
        if val_preds_df is None:
            if val_predictions_path is None:
                # Default: use ensemble predictions
                default_path = (
                    self.artifact_paths.ensemble_dir / "predictions.csv"
                )
                if not default_path.exists():
                    raise FileNotFoundError(
                        f"No predictions file found at {default_path}. "
                        "Provide val_predictions_path or val_preds_df."
                    )
                val_predictions_path = default_path

            logger.info(f"Loading validation predictions from {val_predictions_path}")
            val_preds_df = pd.read_csv(val_predictions_path)

        # Filter to validation split
        if "split" in val_preds_df.columns:
            val_only = val_preds_df[val_preds_df["split"] == "val"].copy()
        else:
            val_only = val_preds_df.copy()
            logger.warning("No 'split' column found — using all data for threshold calibration")

        if len(val_only) == 0:
            raise ThresholdLeakageError(
                "No validation data available for threshold calibration."
            )

        # Check that prob_col exists, fall back to alternatives
        if prob_col not in val_only.columns:
            alternatives = ["prob_cal", "prob_raw", "prob"]
            found = False
            for alt in alternatives:
                if alt in val_only.columns:
                    prob_col = alt
                    found = True
                    logger.warning(f"Using '{alt}' instead of requested prob column")
                    break
            if not found:
                raise ValueError(
                    f"Probability column '{prob_col}' not found. "
                    f"Available: {list(val_only.columns)}"
                )

        # Safety: verify both classes in validation
        unique_labels = val_only["label"].unique()
        if len(unique_labels) < 2:
            raise CalibrationError(
                f"Validation split has only {unique_labels.tolist()} label(s). "
                "Need both classes for safe threshold calibration."
            )

        # Delegate to policy engine
        threshold_config = self._policy.calibrate_thresholds(
            val_preds=val_only,
            prob_col=prob_col,
        )

        logger.info(
            f"Thresholds calibrated: block={threshold_config.block_threshold:.6f}, "
            f"flag={threshold_config.flag_threshold:.6f}"
        )

        return threshold_config

    # ─────────────────────────────────────────────
    # Inference entry points
    # ─────────────────────────────────────────────

    def predict_pcap(
        self,
        pcap_path: str | Path,
        capture_id: Optional[str] = None,
        label: int = -1,
    ) -> SessionDecision:
        """
        Full pipeline: pcap file → session decision.

        Parameters
        ----------
        pcap_path : str or Path
            Path to .pcap or .pcapng file.
        capture_id : str or None
            Session identifier. Defaults to filename stem.
        label : int
            Ground-truth label (-1 for unlabeled).

        Returns
        -------
        SessionDecision
        """
        self._check_ready()

        # Stage 1-2: Flow construction + feature extraction
        df_features = FlowTracker.from_pcap(
            pcap_path=pcap_path,
            capture_id=capture_id,
            label=label,
            drop_direction_features=self.drop_direction_features,
            min_packets=self.min_packets,
            window_n=self.window_n,
        )

        # Stage 3: Ensemble inference
        flow_preds = self._predictor.predict_flow(df_features)

        # Stage 4-5: Session aggregation + decision
        decision = self._policy.predict_session(flow_preds)

        logger.info(
            f"[{decision.capture_id}] Decision: {decision.decision.value} "
            f"(score={decision.session_score:.4f}, "
            f"threshold={decision.block_threshold:.4f}, "
            f"flows={decision.n_flows})"
        )

        return decision

    def predict_flows(
        self,
        df_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict on pre-extracted features.

        Parameters
        ----------
        df_features : pd.DataFrame
            Flow-level features from FlowTracker.

        Returns
        -------
        pd.DataFrame
            Flow-level predictions with probabilities.
        """
        self._check_ready()
        return self._predictor.predict_flow(df_features)

    def predict_session(
        self,
        flow_preds: pd.DataFrame,
    ) -> SessionDecision:
        """
        Make session decision from flow predictions.

        Parameters
        ----------
        flow_preds : pd.DataFrame
            Output from predict_flows() for one session.

        Returns
        -------
        SessionDecision
        """
        return self._policy.predict_session(flow_preds)

    def predict_sessions_batch(
        self,
        flow_preds: pd.DataFrame,
    ) -> List[SessionDecision]:
        """
        Make decisions for multiple sessions.

        Parameters
        ----------
        flow_preds : pd.DataFrame
            Output from predict_flows() for multiple sessions.

        Returns
        -------
        List[SessionDecision]
        """
        return self._policy.predict_sessions_batch(flow_preds)

    def predict_packet_stream(
        self,
        packets: Iterator[Dict[str, Any]],
        capture_id: Optional[str] = None,
        label: int = -1,
    ) -> SessionDecision:
        """
        Full pipeline from a packet iterator (live capture).

        Parameters
        ----------
        packets : Iterator[Dict]
            Packet dicts with ts, src_ip, dst_ip, src_port, dst_port, proto, size.

        Returns
        -------
        SessionDecision
        """
        self._check_ready()

        df_features = FlowTracker.from_packet_iter(
            packets=packets,
            capture_id=capture_id,
            label=label,
            drop_direction_features=self.drop_direction_features,
            min_packets=self.min_packets,
            window_n=self.window_n,
        )

        flow_preds = self._predictor.predict_flow(df_features)
        return self._policy.predict_session(flow_preds)

    def predict_capture(
        self,
        capture_id: str,
        predictions_csv: Optional[str | Path] = None,
        predictions_df: Optional[pd.DataFrame] = None,
        prob_col: str = "prob_iso",
    ) -> SessionDecision:
        """
        Look up a specific capture/session from stored predictions and return
        its firewall decision.

        Parameters
        ----------
        capture_id : str
            The capture identifier to look up.
        predictions_csv : str or Path or None
            Path to predictions CSV. Uses default ensemble predictions if None.
        predictions_df : pd.DataFrame or None
            Pre-loaded predictions. Takes priority over path.
        prob_col : str
            Probability column to use.

        Returns
        -------
        SessionDecision
        """
        self._check_ready()

        if predictions_df is None:
            if predictions_csv is None:
                predictions_csv = self.artifact_paths.ensemble_dir / "predictions.csv"
            predictions_df = pd.read_csv(predictions_csv)

        # Filter to the requested capture
        cap_df = predictions_df[
            predictions_df["capture_id"].astype(str) == str(capture_id)
        ].copy()

        if len(cap_df) == 0:
            raise ValueError(
                f"Capture '{capture_id}' not found in predictions. "
                f"Available captures: {sorted(predictions_df['capture_id'].unique()[:10])}"
            )

        # Map prob_col to prob_cal (the internal name used by policy)
        if prob_col not in cap_df.columns:
            for alt in ["prob_cal", "prob_raw", "prob"]:
                if alt in cap_df.columns:
                    prob_col = alt
                    break
        cap_df = cap_df.rename(columns={prob_col: "prob_cal"})

        return self._policy.predict_session(cap_df)

    # ─────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────

    def evaluate_dataset(
        self,
        predictions_csv: Optional[str | Path] = None,
        predictions_df: Optional[pd.DataFrame] = None,
        prob_col: str = "prob_iso",
        test_split: str = "test",
    ) -> Dict[str, Any]:
        """
        Evaluate the firewall on a labeled test set.

        Uses pre-computed predictions from the training pipeline.

        Parameters
        ----------
        predictions_csv : str or Path
            Path to predictions CSV.
        predictions_df : pd.DataFrame
            Pre-loaded predictions.
        prob_col : str
            Probability column.
        test_split : str
            Split name for test data.

        Returns
        -------
        dict
            Full evaluation metrics.
        """
        if predictions_df is None:
            if predictions_csv is None:
                predictions_csv = self.artifact_paths.ensemble_dir / "predictions.csv"
            predictions_df = pd.read_csv(predictions_csv)

        # Filter to test split
        if "split" in predictions_df.columns:
            test_df = predictions_df[predictions_df["split"] == test_split].copy()
        else:
            test_df = predictions_df.copy()

        if prob_col not in test_df.columns:
            for alt in ["prob_cal", "prob_raw", "prob"]:
                if alt in test_df.columns:
                    prob_col = alt
                    break

        # Generate flow predictions format
        test_df = test_df.rename(columns={prob_col: "prob_cal"})

        # Make session decisions
        decisions = self._policy.predict_sessions_batch(test_df)

        # Evaluate with labels
        metrics = evaluate_with_labels(
            flow_preds=test_df,
            session_decisions=decisions,
            prob_col="prob_cal",
        )

        return metrics

    # ─────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────

    def generate_report(
        self,
        session_decisions: List[SessionDecision],
        flow_preds: Optional[pd.DataFrame] = None,
        output_dir: Optional[Path] = None,
    ) -> str:
        """
        Generate and optionally save a structured report.

        Parameters
        ----------
        session_decisions : list of SessionDecision
        flow_preds : pd.DataFrame or None
            If provided with labels, full evaluation is performed.
        output_dir : Path or None
            If provided, saves JSON report.

        Returns
        -------
        str
            Formatted report text.
        """
        if flow_preds is not None:
            metrics = evaluate_with_labels(
                flow_preds=flow_preds,
                session_decisions=session_decisions,
            )
        else:
            metrics = compute_evaluation_metrics(session_decisions)

        report_text = format_report(
            metrics=metrics,
            predictor_diagnostics=self._predictor.diagnostics(),
            policy_diagnostics=self._policy.diagnostics(),
        )

        if output_dir:
            save_report(metrics, output_dir)

        return report_text

    # ─────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────

    def diagnostics(self) -> Dict[str, Any]:
        """Full system diagnostics."""
        return {
            "loaded": self._loaded,
            "mode": self.mode.value,
            "drop_direction_features": self.drop_direction_features,
            "min_packets": self.min_packets,
            "window_n": self.window_n,
            "predictor": self._predictor.diagnostics(),
            "policy": self._policy.diagnostics(),
        }

    def domain_separability_warning(self) -> Optional[str]:
        """
        Check and warn about domain fingerprinting.

        Returns a warning string if direction features are included
        (which enable dataset-identity separability with AUC ≈ 1.0).
        """
        if not self.drop_direction_features:
            return (
                "WARNING: Domain separability detected. "
                "direction_balance_bytes and direction_balance_packets "
                "enable dataset-identity classification with AUC ≈ 1.0. "
                "Set drop_direction_features=True for domain-robust inference, "
                "or acknowledge this limitation for pooled-domain deployment."
            )
        return None

    # ─────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────

    def _check_ready(self):
        """Verify pipeline is loaded and thresholds are set."""
        if not self._loaded:
            raise RuntimeError(
                "Pipeline not loaded. Call .load() first."
            )
        if not self._policy._thresholds_calibrated and self.mode != DeploymentMode.RESEARCH:
            raise RuntimeError(
                "Thresholds not calibrated. Call .calibrate_from_validation() "
                "or provide explicit thresholds at construction."
            )


