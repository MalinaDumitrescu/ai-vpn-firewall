#!/usr/bin/env python3
"""
scripts/robust9_csv_inference.py
===================================
Standalone CSV-based inference for robust9_firewall.

This script is the CORRECT inference path for robust9_firewall and serves as:
  1. A reference implementation for the FastAPI backend
  2. A debug/validation tool to reproduce notebook-level predictions
  3. An end-to-end smoke test confirming notebook ↔ backend parity

KEY DESIGN PRINCIPLES
---------------------
- Loads model binaries directly from the runtime bundle (no FeaturePipeline transform)
- Uses named pandas DataFrame columns (NOT numpy arrays) to prevent silent column mis-ordering
- Validates input columns exactly against feature_order.json before any predict_proba call
- Applies isotonic calibration using ProbabilityCalibrator.load() (dict format)
- Computes session score as p80(prob_iso) per capture_id
- Applies threshold 0.8718 → BLOCK / PASS

WHY NOT demo_firewall/predictor.py?
------------------------------------
The demo_firewall EnsemblePredictor is wired to the OLD balanced_bagging_firewall_tuned_ensemble
which uses 7 COMPACT_FEATURES:
  [sz_coef_variation, sz_p25_median_ratio, sz_p75_median_ratio, sz_iqr_norm_median,
   dispersion_symmetry, direction_balance_bytes, direction_balance_packets]

robust9_firewall was trained on 9 DIFFERENT features:
  [sz_all_mean, sz_cv, sz_all_p25, sz_all_median, sz_all_p75,
   sz_mean_max, sz_mean_min, sz_std_max, sz_std_min]

Using demo_firewall with robust9 models passes garbage input → all probabilities low → all PASS.

Usage
-----
  # Score the frontend demo CSV and show per-session decisions
  python scripts/robust9_csv_inference.py \
      --input exports/frontend_demo/frontend_demo_robust9.csv \
      --verbose

  # Score a custom CSV (must have: sz_all_mean, sz_cv, sz_all_p25, sz_all_median,
  #   sz_all_p75, sz_mean_max, sz_mean_min, sz_std_max, sz_std_min + session_id/capture_id)
  python scripts/robust9_csv_inference.py --input my_flows.csv

  # Replay notebook predictions for 5 known captures (validation check)
  python scripts/robust9_csv_inference.py --replay-notebook-check

  # Override artifact path (e.g., use runtime bundle instead of source artifact)
  python scripts/robust9_csv_inference.py \
      --artifact-dir exports/app_runtime_bundle/runtime_models/robust9_firewall \
      --input my_flows.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.calibration import ProbabilityCalibrator  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "ensemble" / "diverse_bagging_robust9"
DEFAULT_REGISTRY_DIR = ROOT / "backend" / "model_registry" / "robust9_firewall"
FAMILIES = ["xgb", "lgbm", "cat"]
BAGS = [0, 1, 2]

EXPECTED_FEATURES = [
    "sz_all_mean",
    "sz_cv",
    "sz_all_p25",
    "sz_all_median",
    "sz_all_p75",
    "sz_mean_max",
    "sz_mean_min",
    "sz_std_max",
    "sz_std_min",
]

STRICT_THRESHOLD = 0.8717948717948719
BALANCED_THRESHOLD = 0.8717948717948718
SESSION_COL_CANDIDATES = ["session_id", "capture_id"]

# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


class Robust9Inferencer:
    """
    Correct inference path for robust9_firewall.

    Loads models and calibrator from artifact_dir, validates feature schema,
    scores flows as named DataFrames, calibrates, aggregates to sessions, decides.
    """

    def __init__(
        self,
        artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
        calibration_method: str = "isotonic",
        verbose: bool = False,
    ):
        self.artifact_dir = Path(artifact_dir)
        self.calibration_method = calibration_method
        self.verbose = verbose
        self._models: dict[str, list[Any]] = {}
        self._calibrator: ProbabilityCalibrator | None = None
        self._feature_order: list[str] = EXPECTED_FEATURES
        self._loaded = False

    def load(self) -> "Robust9Inferencer":
        self._validate_artifact_dir()
        self._load_feature_order()
        self._load_models()
        self._load_calibrator()
        self._loaded = True
        if self.verbose:
            print(f"[robust9] Loaded from: {self.artifact_dir}")
            print(f"[robust9] Feature order ({len(self._feature_order)}): {self._feature_order}")
            for fam, mlist in self._models.items():
                print(f"[robust9] {fam}: {len(mlist)} bags loaded")
            print(f"[robust9] Calibrator: {self.calibration_method}, loaded={self._calibrator is not None}")
        return self

    def _validate_artifact_dir(self) -> None:
        if not self.artifact_dir.exists():
            raise FileNotFoundError(f"Artifact directory not found: {self.artifact_dir}")
        missing = []
        for fam in FAMILIES:
            for bag in BAGS:
                p = self.artifact_dir / f"model_{fam}_bag{bag}.pkl"
                if not p.exists():
                    missing.append(p.name)
        if missing:
            raise FileNotFoundError(f"Missing model files: {missing}")

    def _load_feature_order(self) -> None:
        """Load from feature_order.json if available; else fall back to hardcoded EXPECTED_FEATURES."""
        fo_path = self.artifact_dir / "feature_order.json"
        if not fo_path.exists():
            # Fall back to registry
            fo_path = DEFAULT_REGISTRY_DIR / "feature_order.json"
        if fo_path.exists():
            fo = json.loads(fo_path.read_text(encoding="utf-8"))
            self._feature_order = fo.get("feature_order", EXPECTED_FEATURES)
        else:
            self._feature_order = EXPECTED_FEATURES

        # Always cross-check against hard-coded expected list
        if set(self._feature_order) != set(EXPECTED_FEATURES):
            raise ValueError(
                f"feature_order.json contents do not match expected robust9 features.\n"
                f"  From file:     {self._feature_order}\n"
                f"  Expected:      {EXPECTED_FEATURES}"
            )

    def _load_models(self) -> None:
        for fam in FAMILIES:
            self._models[fam] = []
            for bag in BAGS:
                p = self.artifact_dir / f"model_{fam}_bag{bag}.pkl"
                m = joblib.load(p)

                # SAFETY: verify loaded model's feature names match our expected order
                model_feature_names = (
                    getattr(m, "feature_names_", None)
                    or getattr(m, "feature_name_", None)
                    or getattr(m, "feature_names_in_", None)
                )
                if model_feature_names is not None:
                    model_features = [str(x) for x in model_feature_names]
                    if model_features != self._feature_order:
                        raise ValueError(
                            f"Model {p.name} feature order mismatch!\n"
                            f"  Model  : {model_features}\n"
                            f"  Expected: {self._feature_order}"
                        )
                self._models[fam].append(m)

    def _load_calibrator(self) -> None:
        cal_filename = (
            "isotonic_calibrator.pkl"
            if self.calibration_method == "isotonic"
            else "platt_calibrator.pkl"
        )
        cal_path = self.artifact_dir / cal_filename
        if not cal_path.exists():
            print(f"[robust9] WARNING: calibrator not found at {cal_path}. Using raw probabilities.")
            return

        try:
            self._calibrator = ProbabilityCalibrator.load(cal_path)
        except (AttributeError, KeyError, TypeError):
            # Fallback: raw sklearn object (legacy format — should be re-wrapped)
            raw_model = joblib.load(cal_path)
            self._calibrator = ProbabilityCalibrator(method=self.calibration_method, metadata={"format": "raw_sklearn"})
            self._calibrator.model = raw_model
            print(f"[robust9] WARNING: calibrator loaded in legacy raw-sklearn format from {cal_path}.")
            print(f"[robust9]   Re-wrap with: scripts/debug_robust9_inference.py (fix section)")

    def predict_flows(
        self,
        df: pd.DataFrame,
        *,
        session_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Score a DataFrame of flows.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain exactly all 9 robust9 features as named columns.
            Optionally contains session_id or capture_id for grouping.
        session_col : str or None
            Column to group flows into sessions. Auto-detected if None.

        Returns
        -------
        pd.DataFrame
            Input df with added columns:
              prob_xgb, prob_lgbm, prob_cat, prob_raw, prob_iso
        """
        if not self._loaded:
            raise RuntimeError("Call .load() first")

        # Validate input features
        missing = [f for f in self._feature_order if f not in df.columns]
        if missing:
            raise ValueError(
                f"Input DataFrame missing robust9 features: {missing}\n"
                f"Required: {self._feature_order}\n"
                f"Available: {df.columns.tolist()}"
            )

        # Select features as named DataFrame (explicit column ordering, no numpy until inside predict_proba)
        X_df = df[self._feature_order].copy()
        X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        if self.verbose:
            print(f"\n[robust9] Scoring {len(X_df)} flows")
            print(f"[robust9] Feature ranges:")
            for col in self._feature_order:
                vals = X_df[col]
                print(f"  {col}: [{vals.min():.4f}, {vals.max():.4f}]  mean={vals.mean():.4f}")

        # Predict per family (pass named DataFrame to model)
        family_probs: dict[str, np.ndarray] = {}
        for fam in FAMILIES:
            bag_probs = []
            for model in self._models[fam]:
                # Use DataFrame with column names — each framework (XGB/LGBM/CatBoost)
                # will validate input names and re-order if needed
                proba = model.predict_proba(X_df)[:, 1]
                bag_probs.append(proba)
            family_probs[fam] = np.mean(bag_probs, axis=0)

        # Equal-weight cross-family average
        prob_raw = np.mean([family_probs[f] for f in FAMILIES], axis=0)
        prob_raw = np.clip(prob_raw, 0.0, 1.0)

        # Calibrate
        if self._calibrator is not None:
            prob_iso = self._calibrator.predict(prob_raw)
            prob_iso = np.clip(prob_iso, 0.0, 1.0)
        else:
            prob_iso = prob_raw.copy()
            if self.verbose:
                print("[robust9] WARNING: No calibrator. prob_iso = prob_raw")

        # Build output
        out = df.copy()
        for fam in FAMILIES:
            out[f"prob_{fam}"] = family_probs[fam]
        out["prob_raw"] = prob_raw
        out["prob_iso"] = prob_iso

        return out

    def decide_sessions(
        self,
        flow_preds: pd.DataFrame,
        session_col: str | None = None,
        strict_threshold: float = STRICT_THRESHOLD,
        prob_col: str = "prob_iso",
        aggregation: str = "p80",
    ) -> pd.DataFrame:
        """
        Aggregate flow predictions to session scores and make BLOCK/PASS decisions.

        Parameters
        ----------
        flow_preds : pd.DataFrame
            Output from predict_flows() — must have prob_iso and a session column.
        session_col : str or None
            Session grouping column. Auto-detected from SESSION_COL_CANDIDATES.
        strict_threshold : float
            Threshold for BLOCK action. Default: 0.8717948717948719
        prob_col : str
            Column to aggregate. Default: "prob_iso"
        aggregation : str
            "p80" (default) or "mean"

        Returns
        -------
        pd.DataFrame
            One row per session with: session_id, label, n_flows, session_score, action
        """
        if session_col is None:
            for col in SESSION_COL_CANDIDATES:
                if col in flow_preds.columns:
                    session_col = col
                    break
        if session_col is None:
            raise ValueError(f"No session column found. Expected one of: {SESSION_COL_CANDIDATES}")

        if prob_col not in flow_preds.columns:
            raise ValueError(f"Probability column '{prob_col}' not in flow_preds. Available: {flow_preds.columns.tolist()}")

        rows = []
        for sid, group in flow_preds.groupby(session_col):
            scores = group[prob_col].to_numpy(dtype=float)
            if aggregation == "p80":
                session_score = float(np.percentile(scores, 80))
            else:
                session_score = float(np.mean(scores))

            raw_label = group["label"].iloc[0] if "label" in group.columns else -1
            # Support both numeric (0/1) and string ("VPN"/"NONVPN") labels
            if isinstance(raw_label, str):
                label = 1 if raw_label.upper() in {"VPN", "1", "TRUE"} else (0 if raw_label.upper() in {"NONVPN", "NON_VPN", "0", "FALSE"} else -1)
            else:
                try:
                    label = int(raw_label)
                except (ValueError, TypeError):
                    label = -1
            action = "BLOCK" if session_score >= strict_threshold else "PASS"

            row: dict[str, Any] = {
                "session_id": sid,
                "label": label,
                "n_flows": len(group),
                "session_score": session_score,
                "action": action,
                "threshold": strict_threshold,
                "prob_col": prob_col,
                "aggregation": aggregation,
            }
            if "dataset" in group.columns:
                row["dataset"] = group["dataset"].iloc[0]

            # Log individual flow details if verbose
            if self.verbose:
                print(f"\n  [{sid}]  label={'VPN' if label == 1 else 'NONVPN' if label == 0 else '?'}  "
                      f"n_flows={len(group)}")
                print(f"    prob_iso: [{scores.min():.4f}, {scores.max():.4f}]  mean={scores.mean():.4f}")
                if "prob_raw" in group.columns:
                    raw = group["prob_raw"].to_numpy()
                    print(f"    prob_raw: [{raw.min():.4f}, {raw.max():.4f}]  mean={raw.mean():.4f}")
                print(f"    session_score (p80 prob_iso) = {session_score:.6f}  →  {action}")

            rows.append(row)

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Notebook replay check
# ---------------------------------------------------------------------------

def replay_notebook_check(artifact_dir: Path, verbose: bool = True) -> bool:
    """
    Re-run model scoring on the 5 known captures from predictions.csv
    and compare to stored notebook predictions.

    Returns True if all raw probabilities agree within tolerance.
    """
    KNOWN_CAPTURES = {
        "vpn_vpn_hangouts_chat1b.pcap": 1,
        "vpn_vpn_email2b.pcap": 1,
        "vpn_vpn_skype_audio1.pcap": 1,
        "vpn_vpn_bittorrent.pcap": 1,
        "nonvpn_facebook_audio2a.pcap": 0,
    }

    pred_path = artifact_dir / "predictions.csv"
    if not pred_path.exists():
        print(f"  ERROR: predictions.csv not found at {pred_path}")
        return False

    df_pred = pd.read_csv(pred_path)
    print(f"\nLoaded predictions.csv: {df_pred.shape}")
    print(f"Columns: {df_pred.columns.tolist()}")

    # Check: does predictions.csv contain feature columns, or only predictions?
    feature_cols_in_pred = [c for c in EXPECTED_FEATURES if c in df_pred.columns]
    if feature_cols_in_pred:
        print(f"\nFeature columns found in predictions.csv. Can perform live model replay.")
        print(f"Features available: {feature_cols_in_pred}")
        # Full replay: re-run model
        inferencer = Robust9Inferencer(artifact_dir=artifact_dir, verbose=False)
        inferencer.load()

        results = []
        for cap, expected_label in KNOWN_CAPTURES.items():
            subset = df_pred[df_pred["capture_id"] == cap].copy()
            if len(subset) == 0:
                print(f"  [{cap}] NOT FOUND")
                continue

            preds = inferencer.predict_flows(subset)
            session = inferencer.decide_sessions(preds, session_col="capture_id")
            sess_row = session.iloc[0]

            stored_prob_raw_mean = subset["prob_raw"].mean()
            live_prob_raw_mean = preds["prob_raw"].mean()
            diff = abs(live_prob_raw_mean - stored_prob_raw_mean)

            print(f"\n  [{cap}]")
            print(f"    label             = {expected_label}")
            print(f"    stored prob_raw   = {stored_prob_raw_mean:.6f}  (mean from predictions.csv)")
            print(f"    live   prob_raw   = {live_prob_raw_mean:.6f}  (re-run on same features)")
            print(f"    diff              = {diff:.2e}  ({'OK' if diff < 1e-4 else 'MISMATCH!'})")
            print(f"    stored prob_iso   = {subset['prob_iso'].mean():.6f}")
            print(f"    live   prob_iso   = {preds['prob_iso'].mean():.6f}")
            print(f"    session_score     = {sess_row['session_score']:.6f}")
            print(f"    action            = {sess_row['action']}")

            results.append(diff < 1e-4)

        return all(results)

    else:
        # No features in predictions.csv — can only verify calibration + session scoring
        print("\nNo feature columns in predictions.csv — performing calibration+session replay only.")
        cal_path = artifact_dir / "isotonic_calibrator.pkl"
        calibrator = ProbabilityCalibrator.load(cal_path)

        print(f"\nVerifying calibrator reproduces prob_iso from prob_raw...")
        df_pred["prob_iso_recomputed"] = calibrator.predict(df_pred["prob_raw"].values)
        max_diff = (df_pred["prob_iso_recomputed"] - df_pred["prob_iso"]).abs().max()
        print(f"Max diff: {max_diff:.2e}  ({'CONSISTENT' if max_diff < 1e-6 else 'MISMATCH!'})")

        print(f"\n--- Per-session p80(prob_iso) for known captures ---")
        all_ok = True
        for cap, expected_label in KNOWN_CAPTURES.items():
            subset = df_pred[df_pred["capture_id"] == cap]
            if len(subset) == 0:
                print(f"  [{cap}] NOT FOUND")
                continue

            p80_iso = float(np.percentile(subset["prob_iso"].values, 80))
            p80_raw = float(np.percentile(subset["prob_raw"].values, 80))
            action_iso = "BLOCK" if p80_iso >= STRICT_THRESHOLD else "PASS"
            action_raw = "BLOCK" if p80_raw >= STRICT_THRESHOLD else "PASS"
            expected_action = "BLOCK" if expected_label == 1 else "PASS"
            correct = action_iso == expected_action

            print(f"\n  [{cap}]")
            print(f"    label     = {'VPN' if expected_label == 1 else 'NONVPN'}")
            print(f"    n_flows   = {len(subset)}  split={subset['split'].iloc[0]}")
            print(f"    prob_iso  mean={subset['prob_iso'].mean():.4f}  range=[{subset['prob_iso'].min():.4f}, {subset['prob_iso'].max():.4f}]")
            print(f"    prob_raw  mean={subset['prob_raw'].mean():.4f}")
            print(f"    p80(prob_iso) = {p80_iso:.6f}  |  p80(prob_raw) = {p80_raw:.6f}")
            print(f"    ACTION (iso)  = {action_iso}  |  ACTION (raw) = {action_raw}  |  expected={expected_action}")
            print(f"    {'[OK]' if correct else '[WRONG]'} Using prob_iso: {'CORRECT' if correct else 'INCORRECT'}")

            if not correct:
                all_ok = False
                if action_iso != expected_action and action_raw == expected_action:
                    print(f"    [BUG] raw prob gives correct answer but iso does not — calibration inverts signal")
                elif action_iso != expected_action:
                    print(f"    [BUG] Both raw and iso give wrong answer — check feature preprocessing")

        return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="robust9_firewall CSV inference — reference implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input CSV with robust9 features",
    )
    parser.add_argument(
        "--artifact-dir", type=str,
        default=str(DEFAULT_ARTIFACT_DIR),
        help=f"Path to robust9 artifact directory (default: {DEFAULT_ARTIFACT_DIR})",
    )
    parser.add_argument(
        "--calibration", choices=["isotonic", "platt", "none"], default="isotonic",
        help="Calibration method (default: isotonic)",
    )
    parser.add_argument(
        "--threshold", type=float, default=STRICT_THRESHOLD,
        help=f"Block threshold (default: {STRICT_THRESHOLD})",
    )
    parser.add_argument(
        "--aggregation", choices=["p80", "mean"], default="p80",
        help="Session aggregation (default: p80)",
    )
    parser.add_argument(
        "--session-col", type=str, default=None,
        help="Session grouping column (auto-detected if not specified)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Print per-flow diagnostics",
    )
    parser.add_argument(
        "--replay-notebook-check", action="store_true", default=False,
        help="Replay notebook evaluation on known captures and compare",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save session decisions to this CSV path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    artifact_dir = Path(args.artifact_dir)

    if args.replay_notebook_check:
        print("=" * 70)
        print("NOTEBOOK REPLAY CHECK")
        print("=" * 70)
        ok = replay_notebook_check(artifact_dir, verbose=True)
        print(f"\nNotebook replay: {'PASSED' if ok else 'FAILED'}")
        if not ok:
            sys.exit(1)
        # If no input CSV given, stop here
        if args.input is None:
            return

    if args.input is None:
        print("No --input CSV specified. Use --input <path.csv> to score flows.")
        print("Use --replay-notebook-check to validate the artifact without a CSV.")
        sys.exit(0)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Load input
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        print(f"ERROR: Unsupported input format: {input_path.suffix}")
        sys.exit(1)

    print(f"Input: {input_path}  ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(f"Columns: {df.columns.tolist()}")

    # Load and score
    inferencer = Robust9Inferencer(
        artifact_dir=artifact_dir,
        calibration_method=args.calibration if args.calibration != "none" else "isotonic",
        verbose=args.verbose,
    )
    if args.calibration == "none":
        inferencer._calibrator = None

    inferencer.load()

    flow_preds = inferencer.predict_flows(df)

    print(f"\n--- Flow-level probability summary ---")
    print(f"prob_raw: min={flow_preds['prob_raw'].min():.4f}  mean={flow_preds['prob_raw'].mean():.4f}  max={flow_preds['prob_raw'].max():.4f}")
    print(f"prob_iso: min={flow_preds['prob_iso'].min():.4f}  mean={flow_preds['prob_iso'].mean():.4f}  max={flow_preds['prob_iso'].max():.4f}")

    sessions = inferencer.decide_sessions(
        flow_preds,
        session_col=args.session_col,
        strict_threshold=args.threshold,
        prob_col="prob_iso",
        aggregation=args.aggregation,
    )

    n_block = (sessions["action"] == "BLOCK").sum()
    n_pass = (sessions["action"] == "PASS").sum()
    print(f"\n--- Session decisions ---")
    print(f"Total sessions: {len(sessions)}  BLOCK: {n_block}  PASS: {n_pass}")
    print(f"Threshold: {args.threshold:.6f}  Aggregation: {args.aggregation}  Prob col: prob_iso")
    print()
    print(sessions[["session_id", "label", "n_flows", "session_score", "action"]].to_string(index=False))

    if "label" in sessions.columns:
        # Normalize labels to int for metric computation
        def _to_int_label(v: Any) -> int:
            if isinstance(v, str):
                return 1 if v.upper() in {"VPN", "1", "TRUE"} else 0
            try:
                return int(v)
            except (ValueError, TypeError):
                return -1

        labels_int = sessions["label"].map(_to_int_label)
        vpn_df = sessions[labels_int == 1]
        benign_df = sessions[labels_int == 0]
        if len(vpn_df) > 0:
            recall = (vpn_df["action"] == "BLOCK").mean()
            print(f"\n  VPN sessions BLOCKED:  {recall:.4f}  ({int(recall*len(vpn_df))}/{len(vpn_df)})")
        if len(benign_df) > 0:
            fpr = (benign_df["action"] == "BLOCK").mean()
            print(f"  Benign sessions FPR:   {fpr:.4f}  ({int(fpr*len(benign_df))}/{len(benign_df)})")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sessions.to_csv(out_path, index=False)
        print(f"\nSession decisions saved to: {out_path}")


if __name__ == "__main__":
    main()



