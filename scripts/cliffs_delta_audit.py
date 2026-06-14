"""
Cliff's-delta nonparametric direction audit for the 21 safe_core_plus_temporal
features across {vnat, iscx, usbvpn}, with capture-level bootstrap CI.

For each (feature, dataset):
  delta = (2*U / (n_v*n_n)) - 1   with U = Mann-Whitney rank sum of VPN
        = P(X_VPN > Y_nonVPN) - P(X_VPN < Y_nonVPN)
  CI    = 2.5/97.5 percentile of delta over B bootstrap replicates,
          where each replicate resamples captures (with replacement, within class).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(r"C:\Users\scoti\PycharmProjects\ai-vpn-firewall")
OUT = ROOT / "artifacts/thesis_finalization/nb53_sign_reversal_audit/raw_sign"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["vnat", "iscx", "usbvpn"]
FEATURES = [
    "total_packets","total_bytes","mean_pkt_len","std_pkt_len","median_pkt_len",
    "p25_pkt_len","p75_pkt_len","iat_mean","iat_std","iat_median",
    "flow_duration","packet_rate","byte_rate","max_pkt_len","min_pkt_len",
    "iat_cv","iat_p25","iat_p75","iat_iqr","pkt_len_cv","pkt_len_iqr",
]
B = 400
RNG = np.random.default_rng(0)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    pooled = np.concatenate([x, y])
    ranks = rankdata(pooled)
    U = ranks[:nx].sum() - nx * (nx + 1) / 2.0     # Mann-Whitney U for x
    return float(2.0 * U / (nx * ny) - 1.0)


def magnitude(d: float) -> str:
    a = abs(d)
    if a < 0.147: return "negligible"
    if a < 0.33:  return "small"
    if a < 0.474: return "medium"
    return "large"


def main():
    df = pd.read_parquet(ROOT / "artifacts/clean_pipeline/features.parquet",
                         columns=["dataset", "capture_id", "label", *FEATURES])
    df = df.dropna(subset=FEATURES)

    # raw mean/median signs precomputed for agreement check
    raw_signs = pd.read_csv(OUT / "raw_mean_median_signs.csv")

    rows = []
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        # group flow rows by (class, capture_id) for fast bootstrap resampling
        cap_to_idx_v: dict = {c: g.index.values
                              for c, g in sub[sub.label == 1].groupby("capture_id")}
        cap_to_idx_n: dict = {c: g.index.values
                              for c, g in sub[sub.label == 0].groupby("capture_id")}
        caps_v = np.array(list(cap_to_idx_v.keys()))
        caps_n = np.array(list(cap_to_idx_n.keys()))
        # cache feature values
        for f in FEATURES:
            xv = sub.loc[sub.label == 1, f].values
            xn = sub.loc[sub.label == 0, f].values
            d_point = cliffs_delta(xv, xn)
            # bootstrap
            d_boot = np.empty(B)
            for b in range(B):
                rv = RNG.choice(caps_v, size=len(caps_v), replace=True)
                rn = RNG.choice(caps_n, size=len(caps_n), replace=True)
                idx_v = np.concatenate([cap_to_idx_v[c] for c in rv])
                idx_n = np.concatenate([cap_to_idx_n[c] for c in rn])
                xv_b = sub.loc[idx_v, f].values
                xn_b = sub.loc[idx_n, f].values
                d_boot[b] = cliffs_delta(xv_b, xn_b)
            lo, hi = np.percentile(d_boot, [2.5, 97.5])
            # raw mean/median agreement
            row_raw = raw_signs[(raw_signs.feature == f) & (raw_signs.dataset == ds)].iloc[0]
            sign_mean   = int(row_raw["sign_mean"])
            sign_median = int(row_raw["sign_median"])
            sign_delta  = int(np.sign(d_point)) if np.isfinite(d_point) else 0
            rows.append({
                "feature": f, "dataset": ds,
                "cliffs_delta": d_point,
                "ci_lower": float(lo), "ci_upper": float(hi),
                "magnitude": magnitude(d_point),
                "direction": sign_delta,
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "agrees_with_raw_mean":   bool(sign_delta == sign_mean   and sign_delta != 0),
                "agrees_with_raw_median": bool(sign_delta == sign_median and sign_delta != 0),
                "n_vpn": int(len(xv)), "n_nonvpn": int(len(xn)),
                "n_captures_vpn": int(len(caps_v)),
                "n_captures_nonvpn": int(len(caps_n)),
            })
    long_df = pd.DataFrame(rows)
    long_df.to_csv(OUT / "cliffs_delta_per_feature_per_dataset.csv", index=False)

    # pivot: per-feature direction & CI across datasets
    pivot_dir   = long_df.pivot(index="feature", columns="dataset", values="direction").reindex(FEATURES)[DATASETS]
    pivot_ci_ok = long_df.pivot(index="feature", columns="dataset", values="ci_excludes_zero").reindex(FEATURES)[DATASETS]
    pivot_delta = long_df.pivot(index="feature", columns="dataset", values="cliffs_delta").reindex(FEATURES)[DATASETS]

    # final classification
    summary = []
    for f in FEATURES:
        dirs = pivot_dir.loc[f].astype(int).values
        cis  = pivot_ci_ok.loc[f].astype(bool).values
        dvals = pivot_delta.loc[f].values
        signs = set(int(d) for d in dirs if d != 0)
        sign_changes = (1 in signs) and (-1 in signs)
        n_ci_ok = int(cis.sum())
        # classification
        if sign_changes and n_ci_ok == 3:
            cat = "verified_reversal"
        elif sign_changes and n_ci_ok == 2:
            cat = "likely_reversal"
        elif sign_changes and n_ci_ok <= 1:
            cat = "ambiguous"
        elif not sign_changes:
            cat = "no_reversal"
        else:
            cat = "ambiguous"
        summary.append({
            "feature": f,
            "dir_vnat": int(dirs[0]),
            "dir_iscx": int(dirs[1]),
            "dir_usbvpn": int(dirs[2]),
            "delta_vnat": float(dvals[0]),
            "delta_iscx": float(dvals[1]),
            "delta_usbvpn": float(dvals[2]),
            "ci_ok_vnat": bool(cis[0]),
            "ci_ok_iscx": bool(cis[1]),
            "ci_ok_usbvpn": bool(cis[2]),
            "n_ci_excludes_zero": n_ci_ok,
            "sign_changes_across_datasets": bool(sign_changes),
            "category": cat,
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "cliffs_delta_summary.csv", index=False)
    counts = summary_df["category"].value_counts().to_dict()
    (OUT / "cliffs_delta_category_counts.json").write_text(json.dumps(counts, indent=2))

    # also: per-feature agreement with raw mean / raw median
    long_df["full_row"] = long_df.apply(
        lambda r: f"{r['feature']:>15} | {r['dataset']:>6} | "
                  f"d={r['cliffs_delta']:+.3f} CI[{r['ci_lower']:+.3f},{r['ci_upper']:+.3f}] "
                  f"dir={r['direction']:+d} mag={r['magnitude']:>10} "
                  f"agree_mean={r['agrees_with_raw_mean']} agree_med={r['agrees_with_raw_median']}",
        axis=1)
    print("\n=== PER-(feature, dataset) ===")
    print("\n".join(long_df["full_row"].tolist()))
    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))
    print("\nCategory counts:", counts)


if __name__ == "__main__":
    main()

