"""
Flow-construction-mismatch audit: are the cross-dataset sign reversals
explainable by incompatible flow definition / segmentation / window rules
across VNAT, ISCX, USBVPN, rather than by a coding bug?

Outputs (under artifacts/thesis_finalization/nb53_sign_reversal_audit/flow_construction/):
  - per_dataset_descriptor_summary.csv
  - pairwise_ks_jsd.csv
  - domain_classifier_groupkfold.json
  - confusion_matrix_<pair>.csv
  - histograms.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix
)
from sklearn.model_selection import GroupKFold

ROOT = Path(r"C:\Users\scoti\PycharmProjects\ai-vpn-firewall")
OUT  = ROOT / "artifacts/thesis_finalization/nb53_sign_reversal_audit/flow_construction"
OUT.mkdir(parents=True, exist_ok=True)

DESC = ["flow_duration", "total_packets", "total_bytes", "packet_rate", "byte_rate"]
DATASETS = ["vnat", "iscx", "usbvpn"]


def jsd(p, q, bins=60, lo=None, hi=None):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p[np.isfinite(p)]; q = q[np.isfinite(q)]
    if len(p) == 0 or len(q) == 0:
        return np.nan
    if lo is None:
        lo = float(min(p.min(), q.min()))
        hi = float(max(p.max(), q.max()))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    hp, _ = np.histogram(p, bins=edges); hq, _ = np.histogram(q, bins=edges)
    hp = hp / max(hp.sum(), 1); hq = hq / max(hq.sum(), 1)
    m = 0.5 * (hp + hq)
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * kl(hp, m) + 0.5 * kl(hq, m)


def main():
    df = pd.read_parquet(ROOT / "artifacts/clean_pipeline/features.parquet")
    df = df[["dataset", "capture_id", *DESC]].copy()

    # ---------- 1. per-dataset descriptors ----------
    summary_rows = []
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        for c in DESC:
            x = sub[c].values
            x = x[np.isfinite(x)]
            summary_rows.append({
                "dataset": ds,
                "descriptor": c,
                "n_flows": len(x),
                "median": float(np.median(x)),
                "p10": float(np.percentile(x, 10)),
                "p90": float(np.percentile(x, 90)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
            })
    # capture-size descriptor (#flows per capture)
    cap_sizes = df.groupby(["dataset", "capture_id"]).size().reset_index(name="flows_per_capture")
    for ds in DATASETS:
        x = cap_sizes[cap_sizes.dataset == ds]["flows_per_capture"].values
        summary_rows.append({
            "dataset": ds, "descriptor": "flows_per_capture",
            "n_flows": int(len(x)),
            "median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)),
            "mean": float(np.mean(x)), "std": float(np.std(x)),
        })
    pd.DataFrame(summary_rows).to_csv(OUT / "per_dataset_descriptor_summary.csv", index=False)

    # ---------- 2. pairwise KS + JSD on log-scaled descriptors ----------
    pair_rows = []
    pairs = [("vnat","iscx"), ("vnat","usbvpn"), ("iscx","usbvpn")]
    for a, b in pairs:
        for c in DESC:
            xa = df.loc[df.dataset == a, c].values
            xb = df.loc[df.dataset == b, c].values
            xa = xa[np.isfinite(xa)]; xb = xb[np.isfinite(xb)]
            # KS on raw
            ks_stat, ks_p = stats.ks_2samp(xa, xb)
            # JSD on log1p (descriptors span many orders of magnitude)
            la = np.log1p(np.maximum(xa, 0)); lb = np.log1p(np.maximum(xb, 0))
            j = jsd(la, lb)
            pair_rows.append({
                "pair": f"{a}_vs_{b}", "descriptor": c,
                "ks_stat": float(ks_stat), "ks_pvalue": float(ks_p),
                "jsd_log1p": float(j),
            })
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(OUT / "pairwise_ks_jsd.csv", index=False)

    # ---------- 3. domain classifier with capture-grouped CV ----------
    # log1p the descriptors so heavy tails don't dominate splits
    X_full = np.log1p(np.maximum(df[DESC].values, 0))
    groups = df["capture_id"].values
    out_clf = {}
    for a, b in pairs:
        m = df.dataset.isin([a, b])
        Xs = X_full[m]
        ys = (df.dataset[m].values == b).astype(int)  # b = positive class
        gs = groups[m]
        # Need #unique groups >= n_splits; use min(5, ngroups)
        ngrp = len(np.unique(gs))
        n_splits = min(5, ngrp)
        gkf = GroupKFold(n_splits=n_splits)
        y_true_all, y_pred_all, y_score_all = [], [], []
        for tr, te in gkf.split(Xs, ys, groups=gs):
            clf = GradientBoostingClassifier(random_state=0, n_estimators=200, max_depth=3)
            clf.fit(Xs[tr], ys[tr])
            y_score_all.append(clf.predict_proba(Xs[te])[:, 1])
            y_pred_all.append(clf.predict(Xs[te]))
            y_true_all.append(ys[te])
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        y_score = np.concatenate(y_score_all)
        auc = float(roc_auc_score(y_true, y_score))
        acc = float(accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred)
        pd.DataFrame(cm, index=[a, b], columns=[a, b]).to_csv(
            OUT / f"confusion_matrix_{a}_vs_{b}.csv"
        )
        out_clf[f"{a}_vs_{b}"] = {
            "macro_auc": auc, "accuracy": acc,
            "n_flows": int(len(y_true)),
            "n_groups": int(ngrp),
            "n_splits": int(n_splits),
            "confusion_matrix_rows_actual_cols_predicted": cm.tolist(),
            "class_neg": a, "class_pos": b,
        }
    # 3-way classifier (multinomial)
    from sklearn.metrics import classification_report
    ys3 = df.dataset.map({d: i for i, d in enumerate(DATASETS)}).values
    gkf3 = GroupKFold(n_splits=5)
    y_true3, y_pred3 = [], []
    for tr, te in gkf3.split(X_full, ys3, groups=groups):
        clf = GradientBoostingClassifier(random_state=0, n_estimators=200, max_depth=3)
        # GBM is binary; use OvR via sklearn's HistGradientBoosting? simpler: use RandomForest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1, class_weight="balanced")
        clf.fit(X_full[tr], ys3[tr])
        y_pred3.append(clf.predict(X_full[te]))
        y_true3.append(ys3[te])
    y_true3 = np.concatenate(y_true3); y_pred3 = np.concatenate(y_pred3)
    cm3 = confusion_matrix(y_true3, y_pred3)
    pd.DataFrame(cm3, index=DATASETS, columns=DATASETS).to_csv(
        OUT / "confusion_matrix_3way.csv"
    )
    out_clf["3way"] = {
        "accuracy": float(accuracy_score(y_true3, y_pred3)),
        "confusion_matrix_rows_actual_cols_predicted": cm3.tolist(),
        "labels": DATASETS,
    }
    (OUT / "domain_classifier_groupkfold.json").write_text(json.dumps(out_clf, indent=2))

    # ---------- 4. histograms ----------
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, c in zip(axes.flat, DESC + ["flows_per_capture"]):
        for ds, color in zip(DATASETS, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            if c == "flows_per_capture":
                x = cap_sizes[cap_sizes.dataset == ds]["flows_per_capture"].values
            else:
                x = df.loc[df.dataset == ds, c].values
            x = x[np.isfinite(x)]; x = x[x >= 0]
            ax.hist(np.log1p(x), bins=60, alpha=0.45, label=ds, color=color, density=True)
        ax.set_title(f"log1p({c})"); ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / "histograms.png", dpi=130)
    plt.close(fig)

    print("Per-dataset summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print()
    print("Pairwise KS / JSD:")
    print(pair_df.to_string(index=False))
    print()
    print("Domain classifier:")
    print(json.dumps(out_clf, indent=2))


if __name__ == "__main__":
    main()

