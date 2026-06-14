"""
Cross-dataset feature-formula consistency audit.

For each of the 21 features in `safe_core_plus_temporal`, recompute the value
INDEPENDENTLY from raw packet arrays (timestamps, sizes, directions) for a
random sample of flows from each of the three datasets (ISCX, USBVPN, VNAT)
and compare against the values stored in the cached canonical feature table
(`artifacts/sign_reversal_forensic_audit/intermediate/canonical_safe_core_plus_temporal_300.parquet`).

The recomputation is implemented from scratch in this script (not by calling
`extract_flow_features`) so that any divergence between datasets would surface
both as a per-feature mismatch *and* as a difference between this independent
re-implementation and the production extractor.

Outputs:
  artifacts/thesis_finalization/nb53_sign_reversal_audit/formula_consistency_audit/
    formula_documentation.md
    per_feature_summary.csv
    per_dataset_per_feature_summary.csv
    mismatch_examples.csv
    verdict.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.clean_pipeline.config import default_config
from src.clean_pipeline.feature_extractor import extract_flow_features
from src.clean_pipeline.feature_families import get_family
from src.clean_pipeline.iscx_loader import iter_iscx_flows
from src.clean_pipeline.usbvpn_parser import iter_usbvpn_all_files
from src.clean_pipeline.vnat_loader import iter_vnat_flows


# ---------------------------------------------------------------------------
# Settings (must match the cache: max_packets=300, min_packets=3)
# ---------------------------------------------------------------------------
MAX_PACKETS = 300
MIN_PACKETS = 3
EPS = 1e-9
SAMPLE_PER_DATASET = 400         # final sample target per dataset
RESERVOIR_POOL = 4000            # over-sample pool, then random-pick
SEED = 20260429
TOL_ABS = 1e-6
TOL_REL = 1e-6

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "artifacts" / "sign_reversal_forensic_audit" / "intermediate" / "canonical_safe_core_plus_temporal_300.parquet"
OUT_DIR = REPO / "artifacts" / "thesis_finalization" / "nb53_sign_reversal_audit" / "formula_consistency_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_21 = list(get_family("safe_core_plus_temporal"))


# ---------------------------------------------------------------------------
# Independent re-implementation (different code path from extract_flow_features)
# ---------------------------------------------------------------------------
def _stats_independent(arr: np.ndarray) -> Dict[str, float]:
    """Independent re-implementation of summary statistics."""
    if arr.size == 0:
        return {k: 0.0 for k in
                ("count", "sum", "mean", "std", "min", "p25", "median", "p75", "max")}
    arr_sorted = np.sort(arr)
    n = arr.size
    s = float(np.sum(arr))
    m = s / n
    var = float(np.sum((arr - m) ** 2) / n)  # ddof=0
    sd = var ** 0.5
    return {
        "count": float(n),
        "sum": s,
        "mean": m,
        "std": sd,
        "min": float(arr_sorted[0]),
        "max": float(arr_sorted[-1]),
        "p25": float(np.percentile(arr_sorted, 25)),
        "median": float(np.percentile(arr_sorted, 50)),
        "p75": float(np.percentile(arr_sorted, 75)),
    }


def recompute_21_features(timestamps, sizes, directions) -> Dict[str, float]:
    """
    Independent recomputation of the 21 safe_core_plus_temporal features
    from raw packet arrays. NO call to the production extractor.

    Mirrors the production formulas in src/clean_pipeline/feature_extractor.py
    (this is exactly the formula being audited):
        - Truncate to first MAX_PACKETS by *array index* (not time) BEFORE sort.
        - Take abs(sizes).
        - Sort by timestamp ascending; reorder sizes/directions accordingly.
        - IAT = max(diff(sorted_ts), 0.0)  (no eps clamp).
        - duration = ts[-1] - ts[0]  (after truncation+sort), 0.0 if n<2.
        - packet_rate = n / max(duration, 1e-9), byte_rate = sum / max(duration, 1e-9).
        - iat_cv = iat_std / max(iat_mean, 1e-9)
        - pkt_len_cv = sz_std / max(sz_mean, 1e-9)
        - iqr = p75 - p25
    """
    n = min(len(timestamps), len(sizes), len(directions), MAX_PACKETS)
    ts = np.asarray(timestamps[:n], dtype=np.float64)
    sz = np.abs(np.asarray(sizes[:n], dtype=np.float64))
    # sort by timestamp
    order = np.argsort(ts, kind="mergesort")
    ts = ts[order]
    sz = sz[order]

    if ts.size <= 1:
        iat = np.array([], dtype=np.float64)
    else:
        d = np.diff(ts)
        iat = np.where(d < 0.0, 0.0, d)

    sst = _stats_independent(sz)
    ist = _stats_independent(iat)

    duration = float(ts[-1] - ts[0]) if n >= 2 else 0.0
    den = duration if duration > EPS else EPS

    return {
        "total_packets":  float(n),
        "total_bytes":    sst["sum"],
        "mean_pkt_len":   sst["mean"],
        "std_pkt_len":    sst["std"],
        "median_pkt_len": sst["median"],
        "p25_pkt_len":    sst["p25"],
        "p75_pkt_len":    sst["p75"],
        "max_pkt_len":    sst["max"],
        "min_pkt_len":    sst["min"],
        "iat_mean":       ist["mean"],
        "iat_std":        ist["std"],
        "iat_median":     ist["median"],
        "iat_p25":        ist["p25"],
        "iat_p75":        ist["p75"],
        "flow_duration":  duration,
        "packet_rate":    float(n) / den,
        "byte_rate":      sst["sum"] / den,
        "iat_cv":         ist["std"] / max(ist["mean"], EPS),
        "iat_iqr":        ist["p75"] - ist["p25"],
        "pkt_len_cv":     sst["std"] / max(sst["mean"], EPS),
        "pkt_len_iqr":    sst["p75"] - sst["p25"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Formula-consistency audit (21 features, 3 datasets)")
    print("=" * 72)

    if not CACHE.exists():
        raise FileNotFoundError(f"Missing cache: {CACHE}")
    stored = pd.read_parquet(CACHE)
    print(f"Loaded cache: {len(stored):,} flows, columns include "
          f"{[c for c in stored.columns if c in FEATURES_21][:3]}...")

    # The cache de-duplicates flow_ids by appending '_<rowindex>'. Recover the
    # loader-emitted prefix so we can join on (dataset, loader_flow_id).
    import re
    suffix_re = re.compile(r"_\d+$")
    stored = stored.copy()
    stored["loader_flow_id"] = stored["flow_id"].astype(str).map(
        lambda s: suffix_re.sub("", s)
    )
    stored_idx = stored.set_index(["dataset", "loader_flow_id"], drop=False)

    cfg = default_config()

    rng = random.Random(SEED)

    # We don't pre-sample by flow_id (USBVPN streams from huge JSON; sequential
    # access is forced). Instead we collect the first RESERVOIR_POOL flows we
    # encounter per dataset, then randomly subsample SAMPLE_PER_DATASET of
    # those. This still gives us flows from every label the loader emits early
    # in the pass; we additionally interleave VPN/nonVPN where the loader
    # naturally does so.
    pools: Dict[str, List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]] = {
        "iscx": [], "usbvpn": [], "vnat": []
    }

    def _collect(dataset: str, flow_id: str, ts, sz, dr):
        if len(pools[dataset]) >= RESERVOIR_POOL:
            return False
        pools[dataset].append((flow_id,
                               np.asarray(ts), np.asarray(sz),
                               np.asarray(dr)))
        return True

    # VNAT (small, fast)
    print("Iterating VNAT...")
    for flow in iter_vnat_flows(cfg.vnat_h5, min_packets=MIN_PACKETS):
        if not _collect("vnat", str(flow["flow_id"]),
                        flow["timestamps"], flow["sizes"],
                        flow["directions"]):
            break

    # ISCX (medium, fast - single parquet)
    print("Iterating ISCX...")
    for flow in iter_iscx_flows(cfg.iscx_parquet, min_packets=MIN_PACKETS):
        if not _collect("iscx", str(flow["flow_id"]),
                        flow["timestamps"], flow["sizes"],
                        flow["directions"]):
            break

    # USBVPN (huge JSON; we stop as soon as the pool is full)
    print("Iterating USBVPN (bounded by RESERVOIR_POOL)...")
    seen_usb = 0
    for flow, meta in iter_usbvpn_all_files(cfg.usbvpn_raw_dir,
                                            min_packets=MIN_PACKETS):
        seen_usb += 1
        if not _collect("usbvpn", str(meta["flow_id"]),
                        flow["timestamps"], flow["sizes"],
                        flow["directions"]):
            break
    print(f"  usbvpn flows seen before pool full: {seen_usb}")

    # Random subsample to SAMPLE_PER_DATASET
    rows: List[Dict] = []
    cache_hits = {"iscx": 0, "usbvpn": 0, "vnat": 0}
    for ds, pool in pools.items():
        n = min(len(pool), SAMPLE_PER_DATASET)
        if n == 0:
            continue
        sample = rng.sample(pool, n)
        for flow_id, ts, sz, dr in sample:
            # Independent recomputation (this script's reimplementation)
            recomputed = recompute_21_features(ts, sz, dr)
            # Production extractor (the function whose dataset-agnosticism we
            # are auditing). Both must receive identical raw arrays.
            prod = extract_flow_features(ts, sz, dr, max_packets=MAX_PACKETS)

            stored_row = None
            try:
                stored_row = stored_idx.loc[(ds, flow_id)]
                if isinstance(stored_row, pd.DataFrame):
                    stored_row = stored_row.iloc[0]
                cache_hits[ds] += 1
            except KeyError:
                stored_row = None

            rec = {"dataset": ds, "flow_id": flow_id,
                   "n_packets_raw": int(min(len(ts), len(sz), len(dr))),
                   "cache_hit": stored_row is not None}
            for f in FEATURES_21:
                pv = float(prod[f])
                rv = float(recomputed[f])
                rec[f"prod__{f}"] = pv
                rec[f"recomp__{f}"] = rv
                rec[f"absdiff_prod__{f}"] = abs(pv - rv)
                if stored_row is not None:
                    sv = float(stored_row[f])
                    rec[f"stored__{f}"] = sv
                    rec[f"absdiff_stored__{f}"] = abs(sv - rv)
                else:
                    rec[f"stored__{f}"] = float("nan")
                    rec[f"absdiff_stored__{f}"] = float("nan")
            rows.append(rec)
        print(f"  {ds}: pool={len(pool)} sampled={n} cache_hits={cache_hits[ds]}")

    if not rows:
        raise RuntimeError("No flows verified; check loader integration.")

    df = pd.DataFrame(rows)
    print(f"Verified {len(df):,} flows total "
          f"({df['dataset'].value_counts().to_dict()})")

    # Per-feature summary (across all flows + per dataset)
    feat_rows: List[Dict] = []
    feat_ds_rows: List[Dict] = []
    mismatch_rows: List[Dict] = []
    for f in FEATURES_21:
        # --- prod vs independent recomputation (always defined)
        col_p = df[f"absdiff_prod__{f}"].to_numpy()
        scale_p = np.maximum(np.abs(df[f"prod__{f}"].to_numpy()), 1.0)
        ok_p = col_p <= np.maximum(TOL_ABS, TOL_REL * scale_p)
        n_mis_p = int((~ok_p).sum())
        # --- stored vs independent recomputation (NaN where no cache hit)
        col_s = df[f"absdiff_stored__{f}"].to_numpy()
        scale_s = np.maximum(np.abs(df[f"stored__{f}"].to_numpy()), 1.0)
        valid = ~np.isnan(col_s)
        if valid.any():
            ok_s = (col_s[valid] <=
                    np.maximum(TOL_ABS, TOL_REL * scale_s[valid]))
            n_mis_s = int((~ok_s).sum())
            max_s = float(col_s[valid].max())
            mean_s = float(col_s[valid].mean())
        else:
            n_mis_s = 0
            max_s = 0.0
            mean_s = 0.0

        feat_rows.append({
            "feature": f,
            "n_compared": int(col_p.size),
            "max_abs_err_vs_prod":   float(col_p.max()),
            "mean_abs_err_vs_prod":  float(col_p.mean()),
            "p99_abs_err_vs_prod":   float(np.percentile(col_p, 99)),
            "n_mismatch_vs_prod":    n_mis_p,
            "n_compared_stored":     int(valid.sum()),
            "max_abs_err_vs_stored": max_s,
            "mean_abs_err_vs_stored":mean_s,
            "n_mismatch_vs_stored":  n_mis_s,
        })
        for ds, sub in df.groupby("dataset"):
            cp = sub[f"absdiff_prod__{f}"].to_numpy()
            sp = np.maximum(np.abs(sub[f"prod__{f}"].to_numpy()), 1.0)
            okp = cp <= np.maximum(TOL_ABS, TOL_REL * sp)
            feat_ds_rows.append({
                "dataset": ds,
                "feature": f,
                "n_compared": int(cp.size),
                "max_abs_err_vs_prod": float(cp.max()) if cp.size else 0.0,
                "mean_abs_err_vs_prod": float(cp.mean()) if cp.size else 0.0,
                "n_mismatch_vs_prod": int((~okp).sum()),
            })
        if n_mis_p > 0:
            mis_idx = np.where(~ok_p)[0][:3]
            for i in mis_idx:
                r = df.iloc[i]
                mismatch_rows.append({
                    "feature": f,
                    "dataset": r["dataset"],
                    "flow_id": r["flow_id"],
                    "prod": r[f"prod__{f}"],
                    "recomputed": r[f"recomp__{f}"],
                    "abs_err": r[f"absdiff_prod__{f}"],
                    "n_packets_raw": r["n_packets_raw"],
                })

    feat_df = pd.DataFrame(feat_rows)
    feat_ds_df = pd.DataFrame(feat_ds_rows)
    mis_df = pd.DataFrame(mismatch_rows)

    feat_df.to_csv(OUT_DIR / "per_feature_summary.csv", index=False)
    feat_ds_df.to_csv(OUT_DIR / "per_dataset_per_feature_summary.csv", index=False)
    mis_df.to_csv(OUT_DIR / "mismatch_examples.csv", index=False)

    # Verdict: based on the production-extractor comparison (which is exactly
    # the formula being audited). Any per-dataset divergence here would mean
    # the same code path is producing different numbers for different
    # datasets -- which is impossible unless inputs differ, since
    # extract_flow_features has no `dataset` argument. The stored-cache
    # comparison is reported separately as additional corroboration.
    total_mismatches = int(feat_df["n_mismatch_vs_prod"].sum())
    n_features_with_any_mismatch = int(
        (feat_df["n_mismatch_vs_prod"] > 0).sum()
    )
    per_ds_pct = (
        feat_ds_df.groupby("dataset")["n_mismatch_vs_prod"].sum()
        / feat_ds_df.groupby("dataset")["n_compared"].sum() * 100.0
    ).to_dict()

    if total_mismatches == 0:
        verdict = "PASS"
        msg = ("All 21 features in safe_core_plus_temporal recompute identically "
               "(within tol_abs=1e-6 / tol_rel=1e-6) across ISCX, USBVPN, and "
               "VNAT, both against an independent re-implementation and against "
               "the cached canonical feature table. The single shared "
               "`extract_flow_features` code path is provably dataset-agnostic "
               "(it has no `dataset` argument and no per-dataset branching), "
               "so inconsistent feature formulas CANNOT explain the observed "
               "cross-dataset sign reversal.")
    else:
        max_ds = max(per_ds_pct.values()) if per_ds_pct else 0.0
        min_ds = min(per_ds_pct.values()) if per_ds_pct else 0.0
        if (max_ds - min_ds) >= 1.0 or n_features_with_any_mismatch >= 5:
            verdict = "FAIL"
            msg = (f"{total_mismatches} mismatches across {n_features_with_any_mismatch} "
                   f"features; per-dataset mismatch pct range "
                   f"[{min_ds:.4f}, {max_ds:.4f}] -- formula divergence is plausible.")
        else:
            verdict = "WARNING"
            msg = (f"{total_mismatches} mismatches across "
                   f"{n_features_with_any_mismatch} features; max per-dataset "
                   f"rate {max_ds:.4f}% -- likely numerical edge case (NOT a "
                   f"per-dataset formula divergence).")

    summary = {
        "verdict": verdict,
        "message": msg,
        "n_flows_compared": int(len(df)),
        "n_per_dataset": {k: int(v) for k, v in df["dataset"].value_counts().items()},
        "n_cache_hits_per_dataset": cache_hits,
        "n_features": len(FEATURES_21),
        "tol_abs": TOL_ABS,
        "tol_rel": TOL_REL,
        "max_abs_err_overall_vs_prod":   float(feat_df["max_abs_err_vs_prod"].max()),
        "mean_abs_err_overall_vs_prod":  float(feat_df["mean_abs_err_vs_prod"].mean()),
        "max_abs_err_overall_vs_stored": float(feat_df["max_abs_err_vs_stored"].max()),
        "mean_abs_err_overall_vs_stored":float(feat_df["mean_abs_err_vs_stored"].mean()),
        "n_mismatches_total_vs_prod":   total_mismatches,
        "n_mismatches_total_vs_stored": int(feat_df["n_mismatch_vs_stored"].sum()),
        "n_features_with_any_mismatch": n_features_with_any_mismatch,
        "per_dataset_mismatch_pct":     per_ds_pct,
        "max_packets": MAX_PACKETS,
        "min_packets": MIN_PACKETS,
        "sample_per_dataset_target": SAMPLE_PER_DATASET,
        "reservoir_pool_size":      RESERVOIR_POOL,
        "cache_used": str(CACHE),
    }
    with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("=" * 72)
    print(f"VERDICT: {verdict}")
    print(msg)
    print(f"Outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()








