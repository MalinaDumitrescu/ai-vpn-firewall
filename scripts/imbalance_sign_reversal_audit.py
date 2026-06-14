"""
Class-imbalance / large-capture-dominance audit for sign reversal.

For each of 4 sampling strategies, computes single-feature ROC-AUC direction
per (feature, dataset), determines whether the sign flips across the 3
datasets, and reports the *frequency* of reversal across N_SEEDS seeds:

  (a) all flows                       -- single deterministic run
  (b) class-balanced within dataset   -- downsample majority class per (dataset)
                                          to minority size; bootstrap 100 seeds
  (c) capture-balanced                -- 1 flow per capture per class
                                          (random pick per seed); bootstrap 100
                                          seeds
  (d) class+capture-balanced          -- 1 flow per capture, then equalise
                                          captures per class to min count;
                                          bootstrap 100 seeds

Reports per-feature reversal frequency over seeds and final classification.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(r"C:\Users\scoti\PycharmProjects\ai-vpn-firewall")
OUT  = ROOT / "artifacts/thesis_finalization/nb53_sign_reversal_audit/imbalance_audit"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["vnat", "iscx", "usbvpn"]
FEATURES = [
    "total_packets","total_bytes","mean_pkt_len","std_pkt_len","median_pkt_len",
    "p25_pkt_len","p75_pkt_len","iat_mean","iat_std","iat_median",
    "flow_duration","packet_rate","byte_rate","max_pkt_len","min_pkt_len",
    "iat_cv","iat_p25","iat_p75","iat_iqr","pkt_len_cv","pkt_len_iqr",
]
N_SEEDS = 100


def auc(x_pos, x_neg):
    n_p, n_n = len(x_pos), len(x_neg)
    if n_p == 0 or n_n == 0:
        return np.nan
    pooled = np.concatenate([x_pos, x_neg])
    r = rankdata(pooled)
    U = r[:n_p].sum() - n_p * (n_p + 1) / 2.0
    return float(U / (n_p * n_n))


def directions_for(df: pd.DataFrame) -> pd.DataFrame:
    """For each feature, sign of AUC vs 0.5 in each dataset."""
    rows = []
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        for f in FEATURES:
            xv = sub.loc[sub.label == 1, f].values
            xn = sub.loc[sub.label == 0, f].values
            a = auc(xv, xn)
            d = +1 if a > 0.5 else (-1 if a < 0.5 else 0)
            rows.append({"feature": f, "dataset": ds, "auc": a, "direction": d})
    return pd.DataFrame(rows)


def reverses_per_feature(dirs: pd.DataFrame) -> pd.Series:
    """Returns Series[feature -> bool: direction flips across the 3 datasets]."""
    p = dirs.pivot(index="feature", columns="dataset", values="direction")[DATASETS]
    return p.apply(lambda r: (1 in r.values) and (-1 in r.values), axis=1)


# ---------------- samplers ----------------
def sample_class_balanced(df, rng):
    out = []
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        sub_v = sub[sub.label == 1]; sub_n = sub[sub.label == 0]
        m = min(len(sub_v), len(sub_n))
        out.append(sub_v.sample(n=m, random_state=int(rng.integers(0, 2**31-1))))
        out.append(sub_n.sample(n=m, random_state=int(rng.integers(0, 2**31-1))))
    return pd.concat(out, ignore_index=True)


def sample_capture_balanced(df, rng):
    """1 flow per capture per class (per dataset)."""
    out = []
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        for lab in [0, 1]:
            for cap, grp in sub[sub.label == lab].groupby("capture_id"):
                out.append(grp.sample(n=1, random_state=int(rng.integers(0, 2**31-1))))
    return pd.concat(out, ignore_index=True)


def sample_both_balanced(df, rng):
    """1 flow per capture per class, then equalise n_captures per class within
    each dataset to min."""
    parts = []
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        per_cap = []
        for lab in [0, 1]:
            for cap, grp in sub[sub.label == lab].groupby("capture_id"):
                per_cap.append(grp.sample(n=1, random_state=int(rng.integers(0, 2**31-1))))
        per_cap = pd.concat(per_cap, ignore_index=True)
        # now per_cap has 1 flow per capture; balance classes
        v = per_cap[per_cap.label == 1]; n = per_cap[per_cap.label == 0]
        m = min(len(v), len(n))
        parts.append(v.sample(n=m, random_state=int(rng.integers(0, 2**31-1))))
        parts.append(n.sample(n=m, random_state=int(rng.integers(0, 2**31-1))))
    return pd.concat(parts, ignore_index=True)


# ---------------- main ----------------
def main():
    df = pd.read_parquet(ROOT / "artifacts/clean_pipeline/features.parquet",
                         columns=["dataset","capture_id","label", *FEATURES])
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    # ---- 1. class counts per dataset and per-capture flow counts ----
    cls_counts = df.groupby(["dataset","label"]).size().unstack(fill_value=0)
    cls_counts.columns = ["nonVPN","VPN"]
    cap_sizes = df.groupby(["dataset","capture_id"]).size().reset_index(name="n_flows")
    cap_sizes_summary = cap_sizes.groupby("dataset")["n_flows"].describe(
        percentiles=[0.1, 0.5, 0.9]).reset_index()
    cls_counts.to_csv(OUT / "class_counts_per_dataset.csv")
    cap_sizes_summary.to_csv(OUT / "flows_per_capture_summary.csv", index=False)

    # ---- (a) all flows ----
    rev_a = reverses_per_feature(directions_for(df))

    # ---- (b),(c),(d): seed loops ----
    rng = np.random.default_rng(0)
    results = {"b": np.zeros(len(FEATURES), dtype=int),
               "c": np.zeros(len(FEATURES), dtype=int),
               "d": np.zeros(len(FEATURES), dtype=int)}
    feat_index = {f: i for i, f in enumerate(FEATURES)}

    for seed in range(N_SEEDS):
        for tag, sampler in [("b", sample_class_balanced),
                             ("c", sample_capture_balanced),
                             ("d", sample_both_balanced)]:
            df_s = sampler(df, rng)
            rev = reverses_per_feature(directions_for(df_s))
            for f, r in rev.items():
                if bool(r):
                    results[tag][feat_index[f]] += 1
        if (seed + 1) % 10 == 0:
            print(f"  seed {seed+1}/{N_SEEDS}")

    summary = pd.DataFrame({
        "feature": FEATURES,
        "all_flows_reversal":     [bool(rev_a[f]) for f in FEATURES],
        "class_balanced_freq":    results["b"] / N_SEEDS,
        "capture_balanced_freq":  results["c"] / N_SEEDS,
        "both_balanced_freq":     results["d"] / N_SEEDS,
    })

    def verdict(row):
        # all four >= 0.95 -> robust
        if (row["all_flows_reversal"]
            and row["class_balanced_freq"]   >= 0.95
            and row["capture_balanced_freq"] >= 0.95
            and row["both_balanced_freq"]    >= 0.95):
            return "robust_to_imbalance"
        if (row["all_flows_reversal"]
            and row["class_balanced_freq"]   >= 0.50
            and row["capture_balanced_freq"] >= 0.50
            and row["both_balanced_freq"]    >= 0.50):
            return "mostly_robust"
        if row["all_flows_reversal"] and (
            row["class_balanced_freq"]   < 0.20
            or row["capture_balanced_freq"] < 0.20
            or row["both_balanced_freq"]    < 0.20):
            return "imbalance_artifact"
        return "uncertain"
    summary["verdict"] = summary.apply(verdict, axis=1)
    summary.to_csv(OUT / "imbalance_audit_summary.csv", index=False)

    counts = summary["verdict"].value_counts().to_dict()
    (OUT / "imbalance_audit_counts.json").write_text(json.dumps(counts, indent=2))

    print("\n=== Class counts per dataset ===")
    print(cls_counts.to_string())
    print("\n=== Flows per capture (summary) ===")
    print(cap_sizes_summary.to_string(index=False))
    print("\n=== Per-feature reversal frequencies (over", N_SEEDS, "seeds) ===")
    print(summary.to_string(index=False))
    print("\n=== Verdict counts ===")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()

