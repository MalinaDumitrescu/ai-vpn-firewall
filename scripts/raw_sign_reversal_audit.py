"""
Raw-space sign-reversal verification for the 21 safe_core_plus_temporal features.

For each (feature, dataset) computes:
  - mean(VPN), mean(nonVPN), mean_diff = mean(VPN) - mean(nonVPN)
  - median(VPN), median(nonVPN), median_diff
  - capture-balanced versions (each capture contributes equally)
  - capture-level bootstrap 95% CI for both mean_diff and median_diff

Identifies features whose sign disagrees across the 3 datasets in each table.

NO scaling / normalization / PCA / whitening / log transform.
Outputs CSVs to artifacts/thesis_finalization/nb53_sign_reversal_audit/raw_sign/.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

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
B = 1000  # bootstrap replicates
RNG = np.random.default_rng(0)


def sgn(x):
    if not np.isfinite(x):
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def per_capture_means(df_sub: pd.DataFrame, feat: str) -> pd.Series:
    """Mean of `feat` per capture. Returns Series indexed by capture_id."""
    return df_sub.groupby("capture_id")[feat].mean()


def per_capture_medians(df_sub: pd.DataFrame, feat: str) -> pd.Series:
    return df_sub.groupby("capture_id")[feat].median()


def main():
    df = pd.read_parquet(ROOT / "artifacts/clean_pipeline/features.parquet",
                         columns=["dataset", "capture_id", "label", *FEATURES])
    df = df.dropna(subset=FEATURES)

    raw_rows = []
    bal_rows = []
    boot_rows = []

    for ds in DATASETS:
        sub = df[df.dataset == ds]
        sub_v = sub[sub.label == 1]
        sub_n = sub[sub.label == 0]
        # capture lists per class (for bootstrap and capture-balancing)
        caps_v = sub_v["capture_id"].unique()
        caps_n = sub_n["capture_id"].unique()
        for f in FEATURES:
            # ---------- 1. raw flow-level ----------
            mv  = float(sub_v[f].mean());   mn  = float(sub_n[f].mean())
            mdv = float(sub_v[f].median()); mdn = float(sub_n[f].median())
            mean_diff = mv - mn
            med_diff = mdv - mdn
            raw_rows.append({
                "feature": f, "dataset": ds,
                "mean_vpn": mv, "mean_nonvpn": mn, "mean_diff": mean_diff,
                "sign_mean": sgn(mean_diff),
                "median_vpn": mdv, "median_nonvpn": mdn, "median_diff": med_diff,
                "sign_median": sgn(med_diff),
                "n_vpn": int((sub_v[f].notna()).sum()),
                "n_nonvpn": int((sub_n[f].notna()).sum()),
            })

            # ---------- 2. capture-balanced (each capture = 1 unit) ----------
            cap_means_v  = per_capture_means(sub_v, f)
            cap_means_n  = per_capture_means(sub_n, f)
            cap_med_v    = per_capture_medians(sub_v, f)
            cap_med_n    = per_capture_medians(sub_n, f)
            bal_mean_diff = float(cap_means_v.mean()) - float(cap_means_n.mean())
            bal_med_diff  = float(cap_med_v.mean())   - float(cap_med_n.mean())
            bal_rows.append({
                "feature": f, "dataset": ds,
                "balanced_mean_vpn":   float(cap_means_v.mean()),
                "balanced_mean_nonvpn": float(cap_means_n.mean()),
                "balanced_mean_diff":   bal_mean_diff,
                "sign_balanced_mean":   sgn(bal_mean_diff),
                "balanced_median_vpn":   float(cap_med_v.mean()),
                "balanced_median_nonvpn": float(cap_med_n.mean()),
                "balanced_median_diff":   bal_med_diff,
                "sign_balanced_median":   sgn(bal_med_diff),
                "n_captures_vpn": int(len(cap_means_v)),
                "n_captures_nonvpn": int(len(cap_means_n)),
            })

            # ---------- 3. capture-level bootstrap CIs ----------
            cmv = cap_means_v.values; cmn = cap_means_n.values
            cdv = cap_med_v.values;   cdn = cap_med_n.values
            if len(cmv) == 0 or len(cmn) == 0:
                boot_rows.append({"feature": f, "dataset": ds,
                                  "mean_diff_lo": np.nan, "mean_diff_hi": np.nan,
                                  "median_diff_lo": np.nan, "median_diff_hi": np.nan,
                                  "ci_excludes_zero_mean": False,
                                  "ci_excludes_zero_median": False})
                continue
            md_boot = np.empty(B); xd_boot = np.empty(B)
            for b in range(B):
                rv = RNG.choice(cmv, size=len(cmv), replace=True)
                rn = RNG.choice(cmn, size=len(cmn), replace=True)
                rv2 = RNG.choice(cdv, size=len(cdv), replace=True)
                rn2 = RNG.choice(cdn, size=len(cdn), replace=True)
                md_boot[b]  = rv.mean() - rn.mean()
                xd_boot[b]  = rv2.mean() - rn2.mean()
            mlo, mhi = np.percentile(md_boot, [2.5, 97.5])
            xlo, xhi = np.percentile(xd_boot, [2.5, 97.5])
            boot_rows.append({
                "feature": f, "dataset": ds,
                "mean_diff_point": bal_mean_diff,
                "mean_diff_lo": float(mlo), "mean_diff_hi": float(mhi),
                "ci_excludes_zero_mean": bool(mlo > 0 or mhi < 0),
                "median_diff_point": bal_med_diff,
                "median_diff_lo": float(xlo), "median_diff_hi": float(xhi),
                "ci_excludes_zero_median": bool(xlo > 0 or xhi < 0),
            })

    raw_df  = pd.DataFrame(raw_rows)
    bal_df  = pd.DataFrame(bal_rows)
    boot_df = pd.DataFrame(boot_rows)

    raw_df.to_csv(OUT / "raw_mean_median_signs.csv", index=False)
    bal_df.to_csv(OUT / "capture_balanced_signs.csv", index=False)
    boot_df.to_csv(OUT / "capture_bootstrap_ci.csv", index=False)

    # ---------- pivots: sign across datasets (one row per feature) ----------
    def pivot_sign(df_long, sign_col, value_col):
        p = df_long.pivot(index="feature", columns="dataset", values=sign_col).reindex(FEATURES)
        v = df_long.pivot(index="feature", columns="dataset", values=value_col).reindex(FEATURES)
        p = p[DATASETS]; v = v[DATASETS]
        out = p.copy()
        out.columns = [f"sign_{c}" for c in p.columns]
        for c in DATASETS:
            out[f"diff_{c}"] = v[c]
        out["unique_signs"] = p.apply(lambda r: tuple(sorted(set(int(x) for x in r if pd.notna(x)))), axis=1)
        out["reversal"]     = p.apply(lambda r: (1 in r.values) and (-1 in r.values), axis=1)
        return out

    raw_mean_pivot   = pivot_sign(raw_df, "sign_mean",          "mean_diff")
    raw_med_pivot    = pivot_sign(raw_df, "sign_median",        "median_diff")
    bal_mean_pivot   = pivot_sign(bal_df, "sign_balanced_mean", "balanced_mean_diff")
    bal_med_pivot    = pivot_sign(bal_df, "sign_balanced_median","balanced_median_diff")

    raw_mean_pivot.to_csv(OUT / "pivot_raw_mean_signs.csv")
    raw_med_pivot.to_csv(OUT / "pivot_raw_median_signs.csv")
    bal_mean_pivot.to_csv(OUT / "pivot_balanced_mean_signs.csv")
    bal_med_pivot.to_csv(OUT / "pivot_balanced_median_signs.csv")

    # ---------- final classification per feature ----------
    # Robust raw-space reversal: reversal in BOTH raw mean AND raw median AND capture-balanced AND every CI excludes 0
    summary_rows = []
    boot_pivot_mean = boot_df.pivot(index="feature", columns="dataset",
                                    values="ci_excludes_zero_mean").reindex(FEATURES)[DATASETS]
    boot_pivot_med  = boot_df.pivot(index="feature", columns="dataset",
                                    values="ci_excludes_zero_median").reindex(FEATURES)[DATASETS]
    for f in FEATURES:
        rev_raw_mean = bool(raw_mean_pivot.loc[f, "reversal"])
        rev_raw_med  = bool(raw_med_pivot.loc[f, "reversal"])
        rev_bal_mean = bool(bal_mean_pivot.loc[f, "reversal"])
        rev_bal_med  = bool(bal_med_pivot.loc[f, "reversal"])
        all_ci_signif_mean = bool(boot_pivot_mean.loc[f].all())
        all_ci_signif_med  = bool(boot_pivot_med.loc[f].all())
        # classification
        if rev_raw_mean and rev_raw_med and rev_bal_mean and rev_bal_med and all_ci_signif_mean and all_ci_signif_med:
            cat = "ROBUST_REVERSAL"
        elif (rev_raw_mean or rev_raw_med) and not (rev_bal_mean or rev_bal_med):
            cat = "REVERSAL_ONLY_UNDER_IMBALANCE"
        elif (rev_raw_mean or rev_raw_med) and (rev_bal_mean or rev_bal_med) and not (all_ci_signif_mean and all_ci_signif_med):
            cat = "WEAK_OR_UNCERTAIN"
        elif rev_raw_mean or rev_raw_med or rev_bal_mean or rev_bal_med:
            cat = "WEAK_OR_UNCERTAIN"
        else:
            cat = "NO_REVERSAL"
        summary_rows.append({
            "feature": f,
            "rev_raw_mean": rev_raw_mean,
            "rev_raw_median": rev_raw_med,
            "rev_balanced_mean": rev_bal_mean,
            "rev_balanced_median": rev_bal_med,
            "all_ci_excl0_mean": all_ci_signif_mean,
            "all_ci_excl0_median": all_ci_signif_med,
            "category": cat,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary_per_feature.csv", index=False)

    print("=== RAW MEAN SIGN PIVOT ===");        print(raw_mean_pivot.to_string()); print()
    print("=== RAW MEDIAN SIGN PIVOT ===");      print(raw_med_pivot.to_string()); print()
    print("=== CAPTURE-BALANCED MEAN PIVOT ==="); print(bal_mean_pivot.to_string()); print()
    print("=== CAPTURE-BALANCED MEDIAN PIVOT ==="); print(bal_med_pivot.to_string()); print()
    print("=== BOOTSTRAP CI ==="); print(boot_df.to_string(index=False)); print()
    print("=== SUMMARY ==="); print(summary.to_string(index=False)); print()

    # category counts
    counts = summary["category"].value_counts().to_dict()
    (OUT / "category_counts.json").write_text(json.dumps(counts, indent=2))
    print("Category counts:", counts)


if __name__ == "__main__":
    main()

