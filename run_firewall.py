#!/usr/bin/env python3
"""
run_firewall.py — CLI for the VPN detection firewall.

Usage examples:

  # Evaluate ensemble on test set (STRICT mode, zero-FPR)
  python run_firewall.py evaluate

  # Evaluate in BALANCED mode (≤0.1% FPR)
  python run_firewall.py evaluate --mode balanced

  # Evaluate with reduced features (no direction balance — domain-robust)
  python run_firewall.py evaluate --drop-direction

  # Compare all three modes side-by-side
  python run_firewall.py compare

  # Classify a single pcap file
  python run_firewall.py predict path/to/capture.pcap

  # Show full system diagnostics
  python run_firewall.py info

  # Save evaluation report as JSON
  python run_firewall.py evaluate --save-report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_firewall import (
    FirewallBlocker,
    DeploymentMode,
    CalibrationError,
    ThresholdLeakageError,
)
from demo_firewall.report import format_report, save_report


def cmd_evaluate(args):
    """Evaluate the ensemble on the test set."""
    mode = DeploymentMode(args.mode)

    print(f"\n{'='*70}")
    print(f"  VPN FIREWALL EVALUATION — {mode.value.upper()} MODE")
    print(f"{'='*70}\n")

    blocker = FirewallBlocker(
        mode=mode,
        drop_direction_features=args.drop_direction,
        calibration_method=args.calibration,
        model_backend=getattr(args, 'backend', 'ensemble_all'),
    )
    blocker.load()
    blocker.calibrate_from_validation(prob_col=args.prob_col)

    # Domain separability warning
    warn = blocker.domain_separability_warning()
    if warn:
        print(f"  WARNING: {warn}\n")

    # Evaluate
    metrics = blocker.evaluate_dataset(
        prob_col=args.prob_col,
        test_split=args.test_split,
    )

    # Print report
    report_text = format_report(
        metrics=metrics,
        predictor_diagnostics=blocker._predictor.diagnostics(),
        policy_diagnostics=blocker._policy.diagnostics(),
    )
    print(report_text)

    # Optionally save
    if args.save_report:
        out_dir = Path(args.output_dir)
        path = save_report(metrics, out_dir, prefix=f"firewall_{mode.value}")
        print(f"\n  Report saved to: {path}")

    return metrics


def cmd_compare(args):
    """Compare all three deployment modes side-by-side."""
    print(f"\n{'='*70}")
    print(f"  VPN FIREWALL — MODE COMPARISON")
    print(f"{'='*70}\n")

    results = {}
    for mode in DeploymentMode:
        blocker = FirewallBlocker(
            mode=mode,
            drop_direction_features=args.drop_direction,
            calibration_method=args.calibration,
            model_backend=getattr(args, 'backend', 'ensemble_all'),
        )
        blocker.load()

        if mode != DeploymentMode.RESEARCH:
            blocker.calibrate_from_validation(prob_col=args.prob_col)
        else:
            # Research mode needs thresholds for evaluation comparison
            blocker._policy._block_threshold = 0.5
            blocker._policy._flag_threshold = 0.3
            blocker._policy._thresholds_calibrated = True

        metrics = blocker.evaluate_dataset(
            prob_col=args.prob_col,
            test_split=args.test_split,
        )
        results[mode.value] = metrics

    # Print comparison table
    header = f"  {'Metric':<22s} | {'STRICT':>10s} | {'BALANCED':>10s} | {'RESEARCH':>10s}"
    print(header)
    print("  " + "-" * len(header.strip()))

    keys = [
        "session_roc_auc", "session_pr_auc",
        "block_recall", "block_fpr", "block_precision",
        "flagged_recall", "flagged_fpr",
    ]
    for key in keys:
        vals = []
        for mode_name in ["strict", "balanced", "research"]:
            v = results.get(mode_name, {}).get(key)
            vals.append(f"{v:.4f}" if v is not None else "N/A")
        print(f"  {key:<22s} | {vals[0]:>10s} | {vals[1]:>10s} | {vals[2]:>10s}")

    # Thresholds
    print()
    for mode_name in ["strict", "balanced", "research"]:
        bt = results.get(mode_name, {}).get("block_threshold", "N/A")
        ft = results.get(mode_name, {}).get("flag_threshold", "N/A")
        bt_s = f"{bt:.6f}" if isinstance(bt, float) else str(bt)
        ft_s = f"{ft:.6f}" if isinstance(ft, float) else str(ft)
        print(f"  {mode_name.upper():>10s} thresholds: block={bt_s}, flag={ft_s}")

    print()
    return results


def cmd_predict(args):
    """Classify a single pcap file."""
    pcap_path = Path(args.pcap_path)
    if not pcap_path.exists():
        print(f"ERROR: File not found: {pcap_path}")
        sys.exit(1)

    mode = DeploymentMode(args.mode)

    blocker = FirewallBlocker(
        mode=mode,
        drop_direction_features=args.drop_direction,
        calibration_method=args.calibration,
        model_backend=getattr(args, 'backend', 'ensemble_all'),
    )
    blocker.load()
    blocker.calibrate_from_validation(prob_col=args.prob_col)

    decision = blocker.predict_pcap(
        pcap_path=pcap_path,
        label=args.label,
    )

    print(f"\n{'='*70}")
    print(f"  FIREWALL DECISION — {pcap_path.name}")
    print(f"{'='*70}\n")
    print(f"  Decision:        {decision.decision.value}")
    print(f"  Session Score:   {decision.session_score:.6f}")
    print(f"  Block Threshold: {decision.block_threshold:.6f}")
    print(f"  Flag Threshold:  {decision.flag_threshold:.6f}")
    print(f"  Confidence:      {decision.confidence_margin:.6f}")
    print(f"  Flows Analyzed:  {decision.n_flows}")
    print(f"  Flows > Block:   {decision.n_flows_above_block}")
    print(f"  Flows > Flag:    {decision.n_flows_above_flag}")
    print(f"  Aggregation:     {decision.aggregation_rule}")
    print(f"  Mode:            {decision.deployment_mode}")

    if decision.flow_decisions:
        print(f"\n  Per-flow breakdown:")
        print(f"  {'Flow ID':<40s} {'Prob':>8s} {'Decision':>8s}")
        print(f"  {'-'*60}")
        for fd in decision.flow_decisions[:20]:  # Show top 20
            print(f"  {fd.flow_id:<40s} {fd.probability:>8.4f} {fd.decision.value:>8s}")
        if len(decision.flow_decisions) > 20:
            print(f"  ... and {len(decision.flow_decisions) - 20} more flows")

    print()
    return decision


def cmd_info(args):
    """Show system diagnostics."""
    blocker = FirewallBlocker(
        mode=DeploymentMode(args.mode),
        drop_direction_features=args.drop_direction,
        calibration_method=args.calibration,
        model_backend=getattr(args, 'backend', 'ensemble_all'),
    )
    blocker.load()

    diag = blocker.diagnostics()

    print(f"\n{'='*70}")
    print(f"  VPN FIREWALL — SYSTEM INFO")
    print(f"{'='*70}\n")

    print(f"  Mode:                 {diag['mode']}")
    print(f"  Drop Direction Feat:  {diag['drop_direction_features']}")
    print(f"  Min Packets:          {diag['min_packets']}")
    print(f"  Window N:             {diag['window_n']}")
    print()

    pred = diag["predictor"]
    print(f"  PREDICTOR")
    print(f"  {'─'*40}")
    print(f"  Models:       {pred['n_models_total']} ({pred['n_families']} families)")
    print(f"  Calibration:  {pred['calibration_method']}")
    print(f"  Has Calibr.:  {pred['has_calibrator']}")
    print(f"  Features:     {pred['n_features']}")
    print(f"  Feature List: {pred['feature_names']}")
    print(f"  Weights:      {pred['family_weights']}")
    print()

    pol = diag["policy"]
    print(f"  POLICY")
    print(f"  {'─'*40}")
    print(f"  Aggregation:  {pol['aggregation_rule']}")
    print(f"  Target FPR:   {pol['target_fpr']}")
    print(f"  Zero-FPR:     {pol['enforce_zero_block_fpr']}")
    print(f"  Calibrated:   {pol['thresholds_calibrated']}")
    print()

    # Artifact paths
    from demo_firewall.config import default_artifact_paths
    arts = default_artifact_paths(blocker.repo_root)
    missing = arts.validate()
    print(f"  ARTIFACTS")
    print(f"  {'─'*40}")
    print(f"  Ensemble Dir: {arts.ensemble_dir}")
    print(f"  Features Dir: {arts.features_dir}")
    print(f"  Missing:      {len(missing)}")
    if missing:
        for m in missing:
            print(f"    ✗ {m}")
    else:
        print(f"    ✓ All artifacts present")
    print()


def cmd_per_dataset(args):
    """Evaluate per-dataset (ISCX, VNAT, USBVPN) breakdown."""
    import pandas as pd
    import numpy as np

    mode = DeploymentMode(args.mode)

    blocker = FirewallBlocker(
        mode=mode,
        drop_direction_features=args.drop_direction,
        calibration_method=args.calibration,
        model_backend=getattr(args, 'backend', 'ensemble_all'),
    )
    blocker.load()
    blocker.calibrate_from_validation(prob_col=args.prob_col)

    # Load predictions
    preds = pd.read_csv(
        blocker.artifact_paths.ensemble_dir / "predictions.csv"
    )
    test_df = preds[preds["split"] == args.test_split].copy()

    prob_col = args.prob_col
    if prob_col not in test_df.columns:
        for alt in ["prob_cal", "prob_raw", "prob"]:
            if alt in test_df.columns:
                prob_col = alt
                break

    print(f"\n{'='*70}")
    print(f"  VPN FIREWALL — PER-DATASET EVALUATION ({mode.value.upper()})")
    print(f"{'='*70}\n")

    datasets = sorted(test_df["dataset"].unique())

    for ds in ["ALL"] + datasets:
        if ds == "ALL":
            subset = test_df
        else:
            subset = test_df[test_df["dataset"] == ds]

        if len(subset) == 0:
            continue

        # Rename for policy compatibility
        sub = subset.rename(columns={prob_col: "prob_cal"})

        decisions = blocker._policy.predict_sessions_batch(sub)

        if not decisions:
            continue

        from demo_firewall.report import evaluate_with_labels
        metrics = evaluate_with_labels(
            flow_preds=sub,
            session_decisions=decisions,
            prob_col="prob_cal",
        )

        n_pos = metrics.get("n_positive", 0)
        n_neg = metrics.get("n_negative", 0)
        auc = metrics.get("session_roc_auc")
        br = metrics.get("block_recall")
        bfpr = metrics.get("block_fpr")
        fr = metrics.get("flagged_recall")

        auc_s = f"{auc:.4f}" if auc is not None else "N/A"
        br_s = f"{br:.4f}" if br is not None else "N/A"
        bfpr_s = f"{bfpr:.4f}" if bfpr is not None else "N/A"
        fr_s = f"{fr:.4f}" if fr is not None else "N/A"

        label = f"  [{ds}]" if ds != "ALL" else "  [ALL DATASETS]"
        print(f"{label}")
        print(f"    Flows: {len(subset)}, Sessions: {n_pos + n_neg} (pos={n_pos}, neg={n_neg})")
        print(f"    AUC={auc_s}  Block Recall={br_s}  Block FPR={bfpr_s}  Flag Recall={fr_s}")
        print()

    return


def main():
    parser = argparse.ArgumentParser(
        description="VPN Detection Firewall — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Shared arguments
    def add_common_args(p):
        p.add_argument("--mode", choices=["strict", "balanced", "research"],
                        default="strict", help="Deployment mode (default: strict)")
        p.add_argument("--drop-direction", action="store_true",
                        help="Remove direction_balance features (domain-robust mode)")
        p.add_argument("--calibration", choices=["isotonic", "platt", "none"],
                        default="isotonic", help="Calibration method (default: isotonic)")
        p.add_argument("--backend", choices=["ensemble_all", "xgb_only", "lgbm_only", "cat_only"],
                        default="ensemble_all", help="Model backend (default: ensemble_all)")
        p.add_argument("--prob-col", default="prob_iso",
                        help="Probability column in predictions CSV (default: prob_iso)")
        p.add_argument("--test-split", default="test",
                        help="Split name for test data (default: test)")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate ensemble on test set")
    add_common_args(p_eval)
    p_eval.add_argument("--save-report", action="store_true", help="Save JSON report")
    p_eval.add_argument("--output-dir", default="artifacts/eval", help="Report output dir")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare all deployment modes")
    add_common_args(p_cmp)

    # per-dataset
    p_ds = sub.add_parser("per-dataset", help="Per-dataset breakdown")
    add_common_args(p_ds)

    # predict
    p_pred = sub.add_parser("predict", help="Classify a pcap file")
    p_pred.add_argument("pcap_path", help="Path to .pcap/.pcapng file")
    p_pred.add_argument("--label", type=int, default=-1, help="Ground truth label (-1=unknown)")
    add_common_args(p_pred)

    # info
    p_info = sub.add_parser("info", help="Show system diagnostics")
    add_common_args(p_info)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "evaluate": cmd_evaluate,
        "compare": cmd_compare,
        "predict": cmd_predict,
        "info": cmd_info,
        "per-dataset": cmd_per_dataset,
    }

    try:
        commands[args.command](args)
    except (CalibrationError, ThresholdLeakageError) as e:
        print(f"\n  SAFETY ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n  FILE ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()

