"""
Leakage and scaling-artifact audit for sign-reversal detection.

Re-runs single-feature AUC sign-reversal under four scaling regimes and
compares reversal counts:
  (a) raw                                  -- baseline
  (b) source-fitted StandardScaler         -- LODO-clean: scaler fitted only on
                                              the two source datasets, then
                                              applied to all three.
  (c) pooled-fit StandardScaler            -- illegally fitted on all three
                                              datasets pooled.  Diagnostic only.
  (d) per-capture z-score                  -- legacy FeaturePipeline behaviour
                                              for COMPACT_FEATURES.  Diagnostic
                                              only -- this transform is NOT
                                              monotone across captures and CAN
                                              change ranks.

Direction of single-feature AUC (>0.5 vs <0.5) is recorded per (regime,
feature, dataset) and the count of features whose direction flips across the
three datasets is reported per regime.

Per-column StandardScaler is monotone -> regimes (a),(b),(c) MUST give
identical reversal counts; that is the leakage-falsification test.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"C:\Users\scoti\PycharmProjects\ai-vpn-firewall")
OUT  = ROOT / "artifacts/thesis_finalization/nb53_sign_reversal_audit/leakage_audit"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["vnat", "iscx", "usbvpn"]
FEATURES = [
    "total_packets","total_bytes","mean_pkt_len","std_pkt_len","median_pkt_len",
    "p25_pkt_len","p75_pkt_len","iat_mean","iat_std","iat_median",
    "flow_duration","packet_rate","byte_rate","max_pkt_len","min_pkt_len",
    "iat_cv","iat_p25","iat_p75","iat_iqr","pkt_len_cv","pkt_len_iqr",
]


def auc_score(x_pos, x_neg):
    n_p, n_n = len(x_pos), len(x_neg)
    if n_p == 0 or n_n == 0:
        return np.nan
    pooled = np.concatenate([x_pos, x_neg])
    ranks = rankdata(pooled)
    U = ranks[:n_p].sum() - n_p * (n_p + 1) / 2.0
    return float(U / (n_p * n_n))


def per_feature_aucs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        for f in FEATURES:
            xv = sub.loc[sub.label == 1, f].values
            xn = sub.loc[sub.label == 0, f].values
            rows.append({"feature": f, "dataset": ds,
                         "auc": auc_score(xv, xn),
                         "direction": +1 if auc_score(xv, xn) > 0.5 else
                                      (-1 if auc_score(xv, xn) < 0.5 else 0)})
    return pd.DataFrame(rows)


def reversal_count(auc_df: pd.DataFrame) -> int:
    pivot = auc_df.pivot(index="feature", columns="dataset", values="direction")[DATASETS]
    flips = pivot.apply(lambda r: (1 in r.values) and (-1 in r.values), axis=1)
    return int(flips.sum())


def per_capture_zscore(df: pd.DataFrame, feats) -> pd.DataFrame:
    out = df.copy()
    for c in feats:
        means = out.groupby("capture_id")[c].transform("mean")
        stds  = out.groupby("capture_id")[c].transform("std").replace(0, np.nan).fillna(1.0)
        out[c] = (out[c] - means) / stds
    return out


def main():
    df = pd.read_parquet(ROOT / "artifacts/clean_pipeline/features.parquet",
                         columns=["dataset","capture_id","label","split", *FEATURES])
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    summary = {}
    pivots = {}

    # ---------- (a) raw ----------
    auc_a = per_feature_aucs(df)
    auc_a.to_csv(OUT / "auc_raw.csv", index=False)
    summary["a_raw"] = {
        "reversals": reversal_count(auc_a),
        "description": "Raw features, no scaling.",
    }
    pivots["a_raw"] = auc_a.pivot(index="feature", columns="dataset", values="auc")[DATASETS]

    # ---------- (b) LODO source-fitted StandardScaler ----------
    # for each held-out dataset, fit on source-train rows of the other two,
    # transform all three.  Then check direction per dataset.
    # (We fit a fresh scaler per LODO fold; the resulting AUCs may differ across
    #  folds -- we record the held-out direction for each fold and the source
    #  directions averaged across folds.)
    rev_per_fold = {}
    for held in DATASETS:
        sources = [d for d in DATASETS if d != held]
        train_mask = df.dataset.isin(sources) & (df.split == "train")
        scaler = StandardScaler().fit(df.loc[train_mask, FEATURES].values)
        df_b = df.copy()
        df_b[FEATURES] = scaler.transform(df_b[FEATURES].values)
        auc_b = per_feature_aucs(df_b)
        rev_per_fold[held] = reversal_count(auc_b)
        auc_b.to_csv(OUT / f"auc_b_lodo_held_{held}.csv", index=False)
    summary["b_source_fitted_lodo"] = {
        "reversals_per_fold": rev_per_fold,
        "description": ("Per-fold StandardScaler fit only on source-train rows of "
                        "the two non-held datasets, applied to all rows. Identical "
                        "to (a) up to a per-column affine; cannot change AUC."),
    }

    # ---------- (c) pooled-fit StandardScaler ----------
    scaler_c = StandardScaler().fit(df[FEATURES].values)
    df_c = df.copy()
    df_c[FEATURES] = scaler_c.transform(df_c[FEATURES].values)
    auc_c = per_feature_aucs(df_c)
    auc_c.to_csv(OUT / "auc_c_pooled.csv", index=False)
    summary["c_pooled_fit"] = {
        "reversals": reversal_count(auc_c),
        "description": ("StandardScaler fit on ALL flows pooled across datasets "
                        "(leakage). Per-column affine -> still rank-preserving."),
    }
    pivots["c_pooled"] = auc_c.pivot(index="feature", columns="dataset", values="auc")[DATASETS]

    # ---------- (d) per-capture z-score (DIAGNOSTIC ONLY) ----------
    df_d = per_capture_zscore(df, FEATURES)
    auc_d = per_feature_aucs(df_d)
    auc_d.to_csv(OUT / "auc_d_per_capture_zscore.csv", index=False)
    summary["d_per_capture_zscore"] = {
        "reversals": reversal_count(auc_d),
        "description": ("Per-capture z-score (legacy FeaturePipeline behaviour for "
                        "COMPACT_FEATURES). NOT monotone across captures: collapses "
                        "every capture to mean 0 / sd 1, so can flip ranks. "
                        "Diagnostic only -- not used in the 21-feature audit."),
    }
    pivots["d_per_capture_z"] = auc_d.pivot(index="feature", columns="dataset", values="auc")[DATASETS]

    # ---------- equivalence check (a) vs (b)/(c) at the rank level ----------
    # Per-column StandardScaler must leave AUC bit-exactly unchanged.
    same_b = []
    for held in DATASETS:
        auc_b = pd.read_csv(OUT / f"auc_b_lodo_held_{held}.csv")
        same_b.append(np.allclose(
            sorted(auc_a.auc.values), sorted(auc_b.auc.values), atol=1e-9))
    same_c = np.allclose(sorted(auc_a.auc.values), sorted(auc_c.auc.values), atol=1e-9)
    summary["equivalence_check"] = {
        "a_equals_b_per_fold": same_b,
        "a_equals_c": bool(same_c),
        "interpretation": ("a == b == c in AUC values confirms that per-column "
                           "monotone scaling cannot create or destroy sign reversals."),
    }

    (OUT / "leakage_audit_summary.json").write_text(json.dumps(summary, indent=2))

    # combined pivot for printing
    print("=== Reversal counts per regime ===")
    print(json.dumps({k: v.get("reversals", v.get("reversals_per_fold"))
                      for k, v in summary.items()
                      if k != "equivalence_check"}, indent=2))
    print("\n=== Equivalence check (a) vs (b) and (c) ===")
    print(json.dumps(summary["equivalence_check"], indent=2))

    # also print per-feature AUC under (a) and (d) side by side
    join = pd.merge(
        pivots["a_raw"].add_prefix("a_"),
        pivots["d_per_capture_z"].add_prefix("d_"),
        left_index=True, right_index=True,
    )
    join.to_csv(OUT / "auc_a_vs_d_pivot.csv")


if __name__ == "__main__":
    main()

