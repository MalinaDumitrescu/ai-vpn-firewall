#!/usr/bin/env python3
"""
scripts/pkt_size_parity_v2.py
=============================
A/B/C/D compare four packet-size definitions against the robust9 training
feature parquet. Picks ``ip_field`` (default), ``ip_layer``, ``frame``, or
``payload`` based on which one reproduces the training feature distribution
for the same capture.

Computes all 9 robust9 features per flow under each size mode and prints
side-by-side aggregate stats against the ground-truth parquet rows for that
capture_id.

The 9 robust9 features:
    sz_all_mean, sz_cv, sz_all_p25, sz_all_median, sz_all_p75,
    sz_mean_max, sz_mean_min, sz_std_max, sz_std_min

USAGE
-----
  # Reference distributions only
  python scripts/pkt_size_parity_v2.py

  # Verify against a known VPN ISCX pcap
  python scripts/pkt_size_parity_v2.py \\
      --pcap data/raw/iscx/vpn/vpn_aim_chat1a.pcap \\
      --capture-id vpn_vpn_aim_chat1a.pcap

  # Verify against a known NONVPN ISCX pcap
  python scripts/pkt_size_parity_v2.py \\
      --pcap data/raw/iscx/nonvpn/ICQchat2.pcapng \\
      --capture-id nonvpn_icqchat2.pcapng
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODES = ("frame", "ip_field", "ip_layer", "payload")

GROUND_TRUTH_ALIASES: dict[str, list[str]] = {
    "sz_all_mean":   ["sz_all_mean", "mean_pkt_len"],
    "sz_cv":         ["sz_cv", "sz_coef_variation", "pkt_len_cv"],
    "sz_all_p25":    ["sz_all_p25", "p25_pkt_len"],
    "sz_all_median": ["sz_all_median", "median_pkt_len"],
    "sz_all_p75":    ["sz_all_p75", "p75_pkt_len"],
    "sz_mean_max":   ["sz_mean_max"],
    "sz_mean_min":   ["sz_mean_min"],
    "sz_std_max":    ["sz_std_max"],
    "sz_std_min":    ["sz_std_min"],
}
ROBUST9_ORDER = list(GROUND_TRUTH_ALIASES.keys())


def compute_robust9_per_flow(sizes: np.ndarray, directions: np.ndarray) -> dict[str, float]:
    if len(sizes) == 0:
        return {f: 0.0 for f in ROBUST9_ORDER}
    s_all = np.asarray(np.abs(sizes), dtype=float)
    mean_all = float(s_all.mean())
    std_all = float(s_all.std(ddof=0))
    feats = {
        "sz_all_mean":   mean_all,
        "sz_cv":         std_all / max(mean_all, 1e-12),
        "sz_all_p25":    float(np.percentile(s_all, 25)),
        "sz_all_median": float(np.percentile(s_all, 50)),
        "sz_all_p75":    float(np.percentile(s_all, 75)),
    }
    fwd = s_all[directions > 0]
    bwd = s_all[directions < 0]
    if len(fwd) > 0 and len(bwd) > 0:
        fwd_mean, bwd_mean = float(fwd.mean()), float(bwd.mean())
        fwd_std, bwd_std = float(fwd.std(ddof=0)), float(bwd.std(ddof=0))
    elif len(fwd) > 0:
        fwd_mean = bwd_mean = float(fwd.mean())
        fwd_std = bwd_std = float(fwd.std(ddof=0))
    elif len(bwd) > 0:
        fwd_mean = bwd_mean = float(bwd.mean())
        fwd_std = bwd_std = float(bwd.std(ddof=0))
    else:
        fwd_mean = bwd_mean = fwd_std = bwd_std = 0.0
    feats["sz_mean_max"] = max(fwd_mean, bwd_mean)
    feats["sz_mean_min"] = min(fwd_mean, bwd_mean)
    feats["sz_std_max"] = max(fwd_std, bwd_std)
    feats["sz_std_min"] = min(fwd_std, bwd_std)
    return feats


def build_flows_from_pcap(pcap_path: Path, size_mode: str,
                          min_packets: int = 10, window_n: int = 100) -> pd.DataFrame:
    from src.datasets.pcap_reader import iter_packets

    buckets: dict[tuple, list[tuple[float, int, int]]] = defaultdict(list)
    first_pair: dict[tuple, tuple[str, int]] = {}

    for pkt in iter_packets(pcap_path, size_mode=size_mode):
        a = (pkt["src_ip"], pkt["src_port"])
        b = (pkt["dst_ip"], pkt["dst_port"])
        key = (tuple(sorted([a, b])), int(pkt["proto"]))
        if key not in first_pair:
            first_pair[key] = a
        fwd_endpoint = first_pair[key]
        direction = 1 if a == fwd_endpoint else -1
        buckets[key].append((float(pkt["ts"]), int(pkt["size"]), direction))

    rows: list[dict[str, Any]] = []
    for _key, pkts in buckets.items():
        pkts.sort(key=lambda x: x[0])
        for start in range(0, len(pkts), window_n):
            window = pkts[start : start + window_n]
            if len(window) < min_packets:
                continue
            sizes = np.array([p[1] for p in window], dtype=int)
            dirs = np.array([p[2] for p in window], dtype=int)
            feats = compute_robust9_per_flow(sizes, dirs)
            feats["n_packets"] = len(window)
            rows.append(feats)
    return pd.DataFrame(rows)


def load_ground_truth(capture_id: str) -> pd.DataFrame:
    sources = [
        ROOT / "data" / "processed" / "iscx" / "features.parquet",
        ROOT / "artifacts" / "clean_pipeline_test_iscx" / "features.parquet",
        ROOT / "data" / "processed" / "usbvpn" / "flows.parquet",
        ROOT / "data" / "processed" / "vnat" / "features_compact_eval.parquet",
    ]
    parts: list[pd.DataFrame] = []
    for p in sources:
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "capture_id" not in df.columns:
            continue
        sub = df[df["capture_id"].astype(str) == capture_id].copy()
        if len(sub) == 0:
            continue
        for canonical, aliases in GROUND_TRUTH_ALIASES.items():
            if canonical not in sub.columns:
                for alt in aliases:
                    if alt in sub.columns and alt != canonical:
                        sub[canonical] = sub[alt]
                        break
        sub["_source"] = p.relative_to(ROOT).as_posix()
        parts.append(sub)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def summarise_column(s: pd.Series) -> dict[str, float]:
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if len(s2) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan")}
    arr = s2.to_numpy(dtype=float)
    return {"n": int(len(arr)), "mean": float(arr.mean()),
            "median": float(np.median(arr)), "min": float(arr.min()),
            "max": float(arr.max())}


def log_mean_distance(mode_summary: dict[str, float], ref_summary: dict[str, float]) -> float:
    if mode_summary.get("n", 0) == 0 or ref_summary.get("n", 0) == 0:
        return float("inf")
    m_mean = mode_summary.get("mean", float("nan"))
    r_mean = ref_summary.get("mean", float("nan"))
    if np.isnan(m_mean) or np.isnan(r_mean):
        return float("inf")
    return abs(np.log1p(abs(m_mean)) - np.log1p(abs(r_mean)))


def print_side_by_side(per_mode_stats, ref_stats):
    print()
    print("-" * 110)
    print(f"{'Feature':<16} | {'Stat':<7} | "
          f"{'frame':>14} | {'ip_field':>14} | {'ip_layer':>14} | {'payload':>14} | {'REFERENCE':>14}")
    print("-" * 110)
    for feat in ROBUST9_ORDER:
        ref = ref_stats.get(feat, {"n": 0, "mean": float("nan"), "median": float("nan")})
        for stat in ("mean", "median"):
            row = [f"{feat:<16}", f"{stat:<7}"]
            for mode in MODES:
                d = per_mode_stats[mode].get(feat, {})
                v = d.get(stat, float("nan"))
                if isinstance(v, float) and np.isnan(v):
                    row.append(f"{'n/a':>14}")
                else:
                    row.append(f"{v:>14.3f}")
            v_ref = ref.get(stat, float("nan"))
            if isinstance(v_ref, float) and np.isnan(v_ref):
                row.append(f"{'n/a':>14}")
            else:
                row.append(f"{v_ref:>14.3f}")
            print(" | ".join(row))
        print("-" * 110)


def print_distance_table(per_mode_stats, ref_stats) -> tuple[dict[str, float], dict[str, float]]:
    print()
    print("Distance from each mode's flow-mean to REFERENCE flow-mean (lower = better):")
    print("-" * 92)
    print(f"{'Feature':<16} | {'frame':>14} | {'ip_field':>14} | {'ip_layer':>14} | {'payload':>14}")
    print("-" * 92)
    totals = {m: 0.0 for m in MODES}
    sz_mean_only = {m: float("inf") for m in MODES}
    for feat in ROBUST9_ORDER:
        ref = ref_stats.get(feat, {"n": 0, "mean": float("nan")})
        if ref.get("n", 0) == 0:
            continue
        row = [f"{feat:<16}"]
        for mode in MODES:
            d = per_mode_stats[mode].get(feat, {"n": 0, "mean": float("nan")})
            dist = log_mean_distance(d, ref)
            if not np.isinf(dist):
                totals[mode] += dist
            if feat == "sz_all_mean":
                sz_mean_only[mode] = dist
            row.append(f"{dist:>14.4f}" if not np.isinf(dist) else f"{'inf':>14}")
        print(" | ".join(row))
    print("-" * 92)
    row = [f"{'TOTAL':<16}"] + [f"{totals[m]:>14.4f}" for m in MODES]
    print(" | ".join(row))

    print()
    print("PRIMARY DIAGNOSTIC (sz_all_mean — single number per flow, most robust to windowing):")
    print("-" * 92)
    row = [f"{'sz_all_mean dist':<16}"] + [
        (f"{sz_mean_only[m]:>14.4f}" if not np.isinf(sz_mean_only[m]) else f"{'inf':>14}")
        for m in MODES
    ]
    print(" | ".join(row))
    return totals, sz_mean_only


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pcap", type=str, default=None)
    parser.add_argument("--capture-id", type=str, default=None)
    parser.add_argument("--min-packets", type=int, default=10)
    parser.add_argument("--window-n", type=int, default=100)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args(argv)

    print("=" * 92)
    print("REFERENCE DISTRIBUTIONS (training-time per-flow sz_all_mean)")
    print("=" * 92)
    for ref_path, feat_name in [
        (ROOT / "data" / "processed" / "usbvpn" / "flows.parquet", "sz_all_mean"),
        (ROOT / "artifacts" / "clean_pipeline_test_iscx" / "features.parquet", "mean_pkt_len"),
    ]:
        if not ref_path.exists():
            continue
        df = pd.read_parquet(ref_path, columns=[feat_name])
        s = pd.to_numeric(df[feat_name], errors="coerce").dropna().to_numpy(dtype=float)
        if len(s) == 0:
            continue
        verdict = "IP-layer training" if float(s.min()) <= 40 else "L2/frame"
        print(f"  {ref_path.relative_to(ROOT).as_posix()}: n={len(s)}  "
              f"min={float(s.min()):.1f}  mean={float(s.mean()):.1f}  "
              f"max={float(s.max()):.1f}  -> {verdict}")

    if args.reference_only or args.pcap is None:
        if args.pcap is None and not args.reference_only:
            print("\nNo --pcap supplied. Use --pcap path/to/capture.pcap to run mode comparison.")
        return 0

    pcap = Path(args.pcap)
    if not pcap.exists():
        print(f"\nERROR: pcap not found: {pcap}")
        return 1

    capture_id = args.capture_id
    if not capture_id:
        parent = pcap.parent.name.lower()
        prefix = "vpn_" if "vpn" in parent and "non" not in parent else "nonvpn_"
        capture_id = f"{prefix}{pcap.name.lower()}"
        print(f"\nInferred capture_id='{capture_id}' (override with --capture-id)")

    print("\n" + "=" * 92)
    print(f"Building flows from: {pcap.name}")
    print(f"  capture_id   = {capture_id}")
    print(f"  min_packets  = {args.min_packets}")
    print(f"  window_n     = {args.window_n}")
    print("=" * 92)

    per_mode_stats: dict[str, dict[str, dict[str, float]]] = {}
    for mode in MODES:
        try:
            df_flows = build_flows_from_pcap(pcap, size_mode=mode,
                                             min_packets=args.min_packets,
                                             window_n=args.window_n)
        except Exception as exc:
            print(f"  mode={mode}: ERROR {exc}")
            df_flows = pd.DataFrame()

        per_mode_stats[mode] = {
            f: summarise_column(df_flows[f]) if f in df_flows.columns
            else {"n": 0, "mean": float("nan"), "median": float("nan")}
            for f in ROBUST9_ORDER
        }
        n_flows = len(df_flows)
        n_pkts_total = int(df_flows["n_packets"].sum()) if "n_packets" in df_flows.columns else 0
        print(f"  mode={mode:<9s} -> {n_flows} flows ({n_pkts_total} packets retained)")

    print("\n" + "=" * 92)
    print(f"Loading ground-truth features for capture_id={capture_id!r}")
    print("=" * 92)
    gt = load_ground_truth(capture_id)
    if len(gt) == 0:
        print(f"  WARNING: No ground-truth rows found for capture_id={capture_id}.")
        return 2

    print(f"  Found {len(gt)} ground-truth flow rows from:")
    for src in gt["_source"].unique():
        n = int((gt["_source"] == src).sum())
        print(f"    {src}  ({n} rows)")

    ref_stats: dict[str, dict[str, float]] = {}
    for feat in ROBUST9_ORDER:
        if feat in gt.columns:
            ref_stats[feat] = summarise_column(gt[feat])
        else:
            ref_stats[feat] = {"n": 0, "mean": float("nan"), "median": float("nan")}

    print("\nSIDE-BY-SIDE FLOW-LEVEL STATISTICS (mean/median across this capture's flows)")
    print_side_by_side(per_mode_stats, ref_stats)
    totals, sz_mean_only = print_distance_table(per_mode_stats, ref_stats)

    # Final decision: prefer the sz_all_mean-only verdict (least noisy).
    # Total log-distance is reported for transparency, but features like
    # std_max/std_min and percentiles depend on flow-construction details
    # (window boundaries, TCP FIN/RST splitting) that differ between this
    # script's simple 5-tuple+window grouping and the training-time FlowBuilder.
    finite_sz = {m: t for m, t in sz_mean_only.items() if not np.isinf(t)}
    finite_total = {m: t for m, t in totals.items() if not np.isinf(t) and t > 0}

    if not finite_sz:
        print("\nVERDICT: sz_all_mean distance could not be computed. Check ground-truth availability.")
        return 3

    best_sz = min(finite_sz, key=lambda m: finite_sz[m])
    best_total = min(finite_total, key=lambda m: finite_total[m]) if finite_total else best_sz

    # Tie-break: ip_field preferred over ip_layer when essentially identical
    if "ip_field" in finite_sz and "ip_layer" in finite_sz:
        if abs(finite_sz["ip_field"] - finite_sz["ip_layer"]) < 0.01:
            best_sz = "ip_field"

    print()
    print("=" * 92)
    print(f"VERDICT")
    print(f"  by sz_all_mean (PRIMARY): best = '{best_sz}'  "
          f"(distance = {sz_mean_only[best_sz]:.4f})")
    print(f"  by total 9-feature dist : best = '{best_total}'  "
          f"(distance = {totals[best_total]:.4f})")
    print()
    print("  Why prefer sz_all_mean? It is a single number per flow that averages")
    print("  over all packets, so it is robust to the flow-construction differences")
    print("  between this script and the training-time FlowBuilder. Per-direction")
    print("  std/percentile features depend on window boundaries and direction")
    print("  assignment, which add noise unrelated to packet size.")
    print()
    for mode in MODES:
        marker = "  <-- sz_all_mean winner" if mode == best_sz else ""
        d_sz = sz_mean_only[mode]
        d_sz_s = f"{d_sz:.4f}" if not np.isinf(d_sz) else "inf"
        print(f"  {mode:<9s}  sz_all_mean_dist = {d_sz_s:<10s}  total = {totals[mode]:.4f}{marker}")

    if best_sz == "ip_field":
        print("\n  -> src/datasets/pcap_reader.py default is already 'ip_field'. No change needed.")
        print("     This matches the IP-layer convention learned from VPN-class training data")
        print("     (USBVPN flows min=28, ISCX clean_pipeline min=30 -> both impossible at L2).")
    elif best_sz == "ip_layer":
        print("\n  -> 'ip_field' and 'ip_layer' are essentially identical for non-truncated pcaps.")
        print("     'ip_field' is the safer default (robust to snaplen-truncated captures).")
    else:
        print(f"\n  -> Best sz_all_mean match is '{best_sz}'. Note this may indicate the specific")
        print(f"     ground-truth capture was extracted with that convention. For LIVE deployment")
        print(f"     targeting VPN detection, prefer 'ip_field' regardless (matches VPN-class training).")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

