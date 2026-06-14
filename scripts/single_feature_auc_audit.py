"""
Single-feature ROC-AUC direction audit per (feature, dataset), with
capture-level bootstrap 95% CI. Compares AUC direction with previously
computed mean/median/Cliff's-delta directions.

AUC > 0.5  -> larger feature value indicates VPN
AUC < 0.5  -> larger feature value indicates nonVPN
Reversal across datasets <=> at least one AUC < 0.5 and at least one > 0.5.

Note: AUC = (Cliff's delta + 1)/2 (ignoring ties), so a CI on AUC that
excludes 0.5 is exactly equivalent to a Cliff's-delta CI that excludes 0.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(r"C:\Users\scoti\PycharmProjects\ai-vpn-firewall")
OUT  = ROOT / "artifacts/thesis_finalization/nb53_sign_reversal_audit/raw_sign"
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


def auc_score(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    """AUC of single feature distinguishing pos (VPN) from neg (nonVPN)."""
    n_p, n_n = len(x_pos), len(x_neg)
    if n_p == 0 or n_n == 0:
        return np.nan
    pooled = np.concatenate([x_pos, x_neg])
    ranks = rankdata(pooled)
    U = ranks[:n_p].sum() - n_p * (n_p + 1) / 2.0  # Mann-Whitney U for positives
    return float(U / (n_p * n_n))


def main():
    df = pd.read_parquet(ROOT / "artifacts/clean_pipeline/features.parquet",
                         columns=["dataset", "capture_id", "label", *FEATURES])
    df = df.dropna(subset=FEATURES)

    # cross-check tables
    raw_signs = pd.read_csv(OUT / "raw_mean_median_signs.csv")
    cliffs    = pd.read_csv(OUT / "cliffs_delta_per_feature_per_dataset.csv")

    rows = []
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        cap_v = {c: g.index.values for c, g in sub[sub.label == 1].groupby("capture_id")}
        cap_n = {c: g.index.values for c, g in sub[sub.label == 0].groupby("capture_id")}
        caps_v = np.array(list(cap_v.keys())); caps_n = np.array(list(cap_n.keys()))
        for f in FEATURES:
            xv = sub.loc[sub.label == 1, f].values
            xn = sub.loc[sub.label == 0, f].values
            auc = auc_score(xv, xn)
            # bootstrap
            boot = np.empty(B)
            for b in range(B):
                rv = RNG.choice(caps_v, size=len(caps_v), replace=True)
                rn = RNG.choice(caps_n, size=len(caps_n), replace=True)
                idx_v = np.concatenate([cap_v[c] for c in rv])
                idx_n = np.concatenate([cap_n[c] for c in rn])
                boot[b] = auc_score(sub.loc[idx_v, f].values, sub.loc[idx_n, f].values)
            lo, hi = np.percentile(boot, [2.5, 97.5])

            # cross-checks
            row_raw = raw_signs[(raw_signs.feature == f) & (raw_signs.dataset == ds)].iloc[0]
            row_cd  = cliffs[(cliffs.feature == f) & (cliffs.dataset == ds)].iloc[0]
            sign_mean   = int(row_raw["sign_mean"])
            sign_median = int(row_raw["sign_median"])
            sign_delta  = int(row_cd["direction"])
            auc_dir = +1 if auc > 0.5 else (-1 if auc < 0.5 else 0)

            rows.append({
                "feature": f, "dataset": ds,
                "auc": auc, "auc_reversed": 1.0 - auc,
                "ci_lower": float(lo), "ci_upper": float(hi),
                "ci_excludes_0_5": bool(lo > 0.5 or hi < 0.5),
                "direction": auc_dir,
                "interpretation": "larger=>VPN" if auc_dir > 0 else "larger=>nonVPN",
                "agrees_with_mean":   bool(auc_dir == sign_mean),
                "agrees_with_median": bool(auc_dir == sign_median),
                "agrees_with_cliffs_delta": bool(auc_dir == sign_delta),
            })
    long_df = pd.DataFrame(rows)
    long_df.to_csv(OUT / "single_feature_auc.csv", index=False)

    # pivot
    auc_pivot = long_df.pivot(index="feature", columns="dataset", values="auc").reindex(FEATURES)[DATASETS]
    dir_pivot = long_df.pivot(index="feature", columns="dataset", values="direction").reindex(FEATURES)[DATASETS]
    ci_pivot  = long_df.pivot(index="feature", columns="dataset", values="ci_excludes_0_5").reindex(FEATURES)[DATASETS]

    # final classification
    summary = []
    for f in FEATURES:
        a = auc_pivot.loc[f].values
        d = dir_pivot.loc[f].astype(int).values
        c = ci_pivot.loc[f].astype(bool).values
        signs = set(int(x) for x in d if x != 0)
        flips = (1 in signs) and (-1 in signs)
        n_ci = int(c.sum())
        if flips and n_ci == 3:
            cat = "verified_reversal"
        elif flips and n_ci == 2:
            cat = "likely_reversal"
        elif flips and n_ci <= 1:
            cat = "ambiguous"
        elif not flips:
            cat = "no_reversal"
        else:
            cat = "ambiguous"
        # human-readable direction pattern
        def s(x): return "+" if x > 0 else ("-" if x < 0 else "0")
        pattern = f"vnat:{s(d[0])} iscx:{s(d[1])} usbvpn:{s(d[2])}"
        summary.append({
            "feature": f,
            "AUC_vnat":   float(a[0]),
            "AUC_iscx":   float(a[1]),
            "AUC_usbvpn": float(a[2]),
            "ci_ok_vnat":   bool(c[0]),
            "ci_ok_iscx":   bool(c[1]),
            "ci_ok_usbvpn": bool(c[2]),
            "direction_pattern": pattern,
            "n_ci_excludes_0_5": n_ci,
            "flips_across_datasets": bool(flips),
            "category": cat,
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "single_feature_auc_summary.csv", index=False)
    counts = summary_df["category"].value_counts().to_dict()
    (OUT / "single_feature_auc_counts.json").write_text(json.dumps(counts, indent=2))

    # cross-check agreement counts
    agree_counts = {
        "auc_vs_mean":    int(long_df["agrees_with_mean"].sum()),
        "auc_vs_median":  int(long_df["agrees_with_median"].sum()),
        "auc_vs_cliffs":  int(long_df["agrees_with_cliffs_delta"].sum()),
        "total_cells":    int(len(long_df)),
    }
    (OUT / "single_feature_auc_agreement.json").write_text(json.dumps(agree_counts, indent=2))

    print("=== AUC pivot ==="); print(auc_pivot.round(3).to_string()); print()
    print("=== summary ==="); print(summary_df.to_string(index=False)); print()
    print("Category counts:", counts)
    print("Agreement vs (mean, median, cliffs-delta):", agree_counts)


if __name__ == "__main__":
    main()

