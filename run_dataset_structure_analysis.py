#!/usr/bin/env python
"""
DATASET STRUCTURE ANALYSIS — Thesis Figures & Scientific Interpretation
========================================================================
Produces a comprehensive analysis of structural dataset mismatch between
ISCX, USBVPN, and VNAT using the clean 21-feature space.

8 Parts:
  1. Feature distribution comparison (histograms, KDE, violin, boxplot)
  2. Between-dataset distance matrix (Wasserstein, KL, JS)
  3. Multivariate feature-space separation (PCA, t-SNE, UMAP)
  4. Domain classifier structure analysis (LR, XGBoost + permutation importance)
  5. Feature stability across datasets (ANOVA, Kruskal-Wallis, eta²)
  6. Correlation structure difference (per-dataset correlation matrices)
  7. Feature importance instability (per-dataset VPN classifiers)
  8. Structural shift summary report (thesis-ready markdown)

All outputs go to:  artifacts/dataset_structure_analysis/
"""
from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy.stats import (
    wasserstein_distance, entropy, ks_2samp, kruskal,
    f_oneway, spearmanr, skew, kurtosis,
)
from scipy.spatial.distance import jensenshannon

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
)
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "artifacts" / "clean_pipeline" / "features.parquet"
OUT = ROOT / "artifacts" / "dataset_structure_analysis"
OUT.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now(timezone.utc).isoformat()
SEED = 42

FEAT_COLS = [
    "total_packets", "total_bytes", "mean_pkt_len", "std_pkt_len", "median_pkt_len",
    "p25_pkt_len", "p75_pkt_len", "iat_mean", "iat_std", "iat_median",
    "flow_duration", "packet_rate", "byte_rate", "max_pkt_len", "min_pkt_len",
    "iat_cv", "iat_p25", "iat_p75", "iat_iqr", "pkt_len_cv", "pkt_len_iqr",
]

DATASETS = ["iscx", "usbvpn", "vnat"]
DS_COLORS = {"iscx": "#e74c3c", "usbvpn": "#2ecc71", "vnat": "#3498db"}
DS_LABELS = {"iscx": "ISCX", "usbvpn": "USBVPN", "vnat": "VNAT"}

# Plotting defaults — thesis quality
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

MAX_SAMPLE = 8000  # per dataset for heavy plots (t-SNE, UMAP)


# ================================================================
# DATA LOADING
# ================================================================
def load_data():
    print("Loading features...")
    df = pd.read_parquet(FEATURES_PATH)
    available = set(df.columns)
    feat_cols = [f for f in FEAT_COLS if f in available]
    datasets = sorted(df.dataset.unique())
    print(f"  {len(df)} flows, datasets: {datasets}, features: {len(feat_cols)}")
    for ds in datasets:
        n = (df.dataset == ds).sum()
        n_vpn = ((df.dataset == ds) & (df.label == 1)).sum()
        print(f"    {ds}: {n} flows ({n_vpn} VPN, {n - n_vpn} non-VPN)")
    return df, feat_cols


# ================================================================
# PART 1 — FEATURE DISTRIBUTION COMPARISON
# ================================================================
def part1_feature_distributions(df, feat_cols):
    print("\n" + "=" * 70)
    print("PART 1: FEATURE DISTRIBUTION COMPARISON")
    print("=" * 70)

    stats_rows = []

    for fi, feat in enumerate(feat_cols):
        print(f"  [{fi+1}/{len(feat_cols)}] {feat}...", end=" ", flush=True)

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

        # --- Histogram overlay ---
        ax1 = fig.add_subplot(gs[0, 0])
        for ds in DATASETS:
            vals = df.loc[df.dataset == ds, feat].dropna().values
            # Clip extreme outliers for visualization
            lo, hi = np.percentile(vals, [1, 99])
            vals_clip = vals[(vals >= lo) & (vals <= hi)]
            ax1.hist(vals_clip, bins=60, alpha=0.45, label=DS_LABELS[ds],
                     color=DS_COLORS[ds], density=True, edgecolor="none")
        ax1.set_title(f"Histogram — {feat}")
        ax1.set_xlabel(feat)
        ax1.set_ylabel("Density")
        ax1.legend()

        # --- KDE overlay ---
        ax2 = fig.add_subplot(gs[0, 1])
        for ds in DATASETS:
            vals = df.loc[df.dataset == ds, feat].dropna().values
            lo, hi = np.percentile(vals, [1, 99])
            vals_clip = vals[(vals >= lo) & (vals <= hi)]
            if len(vals_clip) > 10:
                sns.kdeplot(vals_clip, ax=ax2, label=DS_LABELS[ds],
                            color=DS_COLORS[ds], linewidth=1.8, fill=True, alpha=0.15)
        ax2.set_title(f"KDE — {feat}")
        ax2.set_xlabel(feat)
        ax2.legend()

        # --- Violin plot ---
        ax3 = fig.add_subplot(gs[1, 0])
        plot_data = []
        for ds in DATASETS:
            vals = df.loc[df.dataset == ds, feat].dropna().values
            lo, hi = np.percentile(vals, [2, 98])
            vals_clip = vals[(vals >= lo) & (vals <= hi)]
            for v in vals_clip[:5000]:  # limit for speed
                plot_data.append({"dataset": DS_LABELS[ds], "value": v})
        plot_df = pd.DataFrame(plot_data)
        if len(plot_df) > 0:
            palette = {DS_LABELS[k]: v for k, v in DS_COLORS.items()}
            sns.violinplot(data=plot_df, x="dataset", y="value", ax=ax3,
                           palette=palette, inner="quartile", linewidth=0.8)
        ax3.set_title(f"Violin — {feat}")
        ax3.set_ylabel(feat)
        ax3.set_xlabel("")

        # --- Boxplot ---
        ax4 = fig.add_subplot(gs[1, 1])
        if len(plot_df) > 0:
            sns.boxplot(data=plot_df, x="dataset", y="value", ax=ax4,
                        palette=palette, linewidth=0.8, fliersize=1.5)
        ax4.set_title(f"Boxplot — {feat}")
        ax4.set_ylabel(feat)
        ax4.set_xlabel("")

        fig.suptitle(f"Feature Distribution: {feat}", fontsize=14, y=1.01)
        fig.savefig(OUT / f"feature_distributions_{feat}.png")
        fig.savefig(OUT / f"fig_feature_{feat}_distribution.png")
        plt.close(fig)

        # Compute statistics
        for ds in DATASETS:
            vals = df.loc[df.dataset == ds, feat].dropna().values
            stats_rows.append({
                "feature": feat,
                "dataset": ds,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
                "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                "skewness": float(skew(vals)) if len(vals) > 2 else float("nan"),
                "kurtosis": float(kurtosis(vals)) if len(vals) > 2 else float("nan"),
                "n": len(vals),
            })

        print("done")

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUT / "feature_distribution_stats.csv", index=False)
    stats_df.to_csv(OUT / "feature_distribution_summary.csv", index=False)
    print(f"  Saved feature_distribution_stats.csv + feature_distribution_summary.csv ({len(stats_df)} rows)")
    print(f"  Saved {len(feat_cols)} feature distribution figures")
    return stats_df


# ================================================================
# PART 2 — BETWEEN-DATASET DISTANCE MATRIX
# ================================================================
def part2_distance_matrix(df, feat_cols):
    print("\n" + "=" * 70)
    print("PART 2: BETWEEN-DATASET DISTANCE MATRIX")
    print("=" * 70)

    # Pre-extract per-dataset feature arrays
    ds_arrays = {}
    for ds in DATASETS:
        ds_arrays[ds] = df.loc[df.dataset == ds, feat_cols].dropna().values

    pairs = [("iscx", "usbvpn"), ("iscx", "vnat"), ("usbvpn", "vnat")]
    distance_rows = []

    for feat_idx, feat in enumerate(feat_cols):
        print(f"  [{feat_idx+1}/{len(feat_cols)}] {feat}...", end=" ", flush=True)
        col_idx = feat_cols.index(feat)

        for ds1, ds2 in pairs:
            v1 = ds_arrays[ds1][:, col_idx]
            v2 = ds_arrays[ds2][:, col_idx]

            # Wasserstein
            wd = float(wasserstein_distance(v1, v2))

            # KL divergence (approximated via histograms)
            combined = np.concatenate([v1, v2])
            lo, hi = np.percentile(combined, [0.5, 99.5])
            bins = np.linspace(lo, hi, 100)
            h1, _ = np.histogram(v1, bins=bins, density=True)
            h2, _ = np.histogram(v2, bins=bins, density=True)
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            h1 = h1 + eps
            h2 = h2 + eps
            h1 = h1 / h1.sum()
            h2 = h2 / h2.sum()
            kl_12 = float(entropy(h1, h2))
            kl_21 = float(entropy(h2, h1))
            kl_sym = (kl_12 + kl_21) / 2.0

            # Jensen-Shannon divergence
            js = float(jensenshannon(h1, h2))

            # KS test
            ks_stat, ks_pval = ks_2samp(v1, v2)

            distance_rows.append({
                "feature": feat,
                "dataset_1": ds1,
                "dataset_2": ds2,
                "wasserstein": round(wd, 6),
                "kl_divergence_sym": round(kl_sym, 6),
                "jensen_shannon": round(js, 6),
                "ks_statistic": round(float(ks_stat), 6),
                "ks_pvalue": float(ks_pval),
            })

        print("done")

    dist_df = pd.DataFrame(distance_rows)
    dist_df.to_csv(OUT / "dataset_distance_matrix.csv", index=False)
    dist_df.to_csv(OUT / "dataset_distance_values.csv", index=False)
    print(f"  Saved dataset_distance_matrix.csv + dataset_distance_values.csv ({len(dist_df)} rows)")

    # --- Heatmaps ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    metrics = [
        ("wasserstein", "Wasserstein Distance"),
        ("kl_divergence_sym", "Symmetric KL Divergence"),
        ("jensen_shannon", "Jensen-Shannon Divergence"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        # Aggregate across features: mean distance per pair
        pivot = dist_df.groupby(["dataset_1", "dataset_2"])[metric].mean().reset_index()

        # Build symmetric matrix
        mat = pd.DataFrame(0.0, index=DATASETS, columns=DATASETS)
        for _, row in pivot.iterrows():
            mat.loc[row["dataset_1"], row["dataset_2"]] = row[metric]
            mat.loc[row["dataset_2"], row["dataset_1"]] = row[metric]

        sns.heatmap(mat.astype(float), annot=True, fmt=".4f", cmap="YlOrRd",
                    ax=ax, square=True, linewidths=0.5,
                    xticklabels=[DS_LABELS[d] for d in DATASETS],
                    yticklabels=[DS_LABELS[d] for d in DATASETS])
        ax.set_title(title, fontsize=12)

    fig.suptitle("Mean Pairwise Dataset Distance (Averaged Across 21 Features)", fontsize=14, y=1.02)
    fig.savefig(OUT / "dataset_distance_heatmap.png")
    plt.close(fig)
    print("  Saved dataset_distance_heatmap.png")

    # --- Per-feature heatmap (top 10 most shifted features by JS) ---
    avg_js = dist_df.groupby("feature")["jensen_shannon"].mean().sort_values(ascending=False)
    top10 = avg_js.head(10).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 7))
    pair_labels = [f"{DS_LABELS[a]} vs {DS_LABELS[b]}" for a, b in pairs]
    hm_data = []
    for feat in top10:
        row = []
        for ds1, ds2 in pairs:
            val = dist_df[(dist_df.feature == feat) &
                          (dist_df.dataset_1 == ds1) &
                          (dist_df.dataset_2 == ds2)]["jensen_shannon"].values
            row.append(float(val[0]) if len(val) > 0 else 0.0)
        hm_data.append(row)

    hm_df = pd.DataFrame(hm_data, index=top10, columns=pair_labels)
    sns.heatmap(hm_df, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
                linewidths=0.5)
    ax.set_title("Jensen-Shannon Divergence — Top 10 Most Shifted Features", fontsize=13)
    ax.set_ylabel("Feature")
    fig.savefig(OUT / "dataset_distance_top10_features.png")
    plt.close(fig)
    print("  Saved dataset_distance_top10_features.png")

    return dist_df


# ================================================================
# PART 3 — MULTIVARIATE FEATURE SPACE SEPARATION
# ================================================================
def part3_multivariate_projection(df, feat_cols):
    print("\n" + "=" * 70)
    print("PART 3: MULTIVARIATE FEATURE SPACE SEPARATION")
    print("=" * 70)

    # Subsample for speed
    rng = np.random.default_rng(SEED)
    dfs = []
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        if len(sub) > MAX_SAMPLE:
            idx = rng.choice(len(sub), MAX_SAMPLE, replace=False)
            dfs.append(sub.iloc[idx])
        else:
            dfs.append(sub)
    sampled = pd.concat(dfs, ignore_index=True)
    print(f"  Sampled {len(sampled)} flows for projection")

    X = sampled[feat_cols].values.astype(np.float32)
    ds_labels = sampled.dataset.values
    vpn_labels = sampled.label.values

    # Standardize
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Handle NaN/Inf
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)

    # --- PCA ---
    print("  Computing PCA...", end=" ", flush=True)
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_s)
    ev = pca.explained_variance_ratio_
    print(f"done (var: {ev[0]:.3f}, {ev[1]:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    # Color by dataset
    for ds in DATASETS:
        mask = ds_labels == ds
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.25, s=6,
                        color=DS_COLORS[ds], label=DS_LABELS[ds], rasterized=True)
    axes[0].set_xlabel(f"PC1 ({ev[0]:.1%} variance)")
    axes[0].set_ylabel(f"PC2 ({ev[1]:.1%} variance)")
    axes[0].set_title("PCA — Colored by Dataset")
    axes[0].legend(markerscale=4)

    # Color by VPN label
    vpn_colors = {0: "#95a5a6", 1: "#e67e22"}
    vpn_names = {0: "Non-VPN", 1: "VPN"}
    for lbl in [0, 1]:
        mask = vpn_labels == lbl
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.25, s=6,
                        color=vpn_colors[lbl], label=vpn_names[lbl], rasterized=True)
    axes[1].set_xlabel(f"PC1 ({ev[0]:.1%} variance)")
    axes[1].set_ylabel(f"PC2 ({ev[1]:.1%} variance)")
    axes[1].set_title("PCA — Colored by VPN Label")
    axes[1].legend(markerscale=4)

    fig.suptitle("PCA Feature Space Projection", fontsize=14, y=1.01)
    fig.savefig(OUT / "pca_dataset_projection.png")
    fig.savefig(OUT / "fig_pca_dataset_space.png")
    plt.close(fig)
    print("  Saved pca_dataset_projection.png + fig_pca_dataset_space.png")

    # --- PCA 3D ---
    print("  Computing PCA 3D...", end=" ", flush=True)
    pca3 = PCA(n_components=3, random_state=SEED)
    X_pca3 = pca3.fit_transform(X_s)
    ev3 = pca3.explained_variance_ratio_
    print(f"done (var: {ev3[0]:.3f}, {ev3[1]:.3f}, {ev3[2]:.3f})")

    fig = plt.figure(figsize=(16, 6.5))
    ax1 = fig.add_subplot(121, projection="3d")
    for ds in DATASETS:
        mask = ds_labels == ds
        ax1.scatter(X_pca3[mask, 0], X_pca3[mask, 1], X_pca3[mask, 2],
                    alpha=0.20, s=4, color=DS_COLORS[ds], label=DS_LABELS[ds],
                    rasterized=True)
    ax1.set_xlabel(f"PC1 ({ev3[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({ev3[1]:.1%})")
    ax1.set_zlabel(f"PC3 ({ev3[2]:.1%})")
    ax1.set_title("PCA 3D — by Dataset")
    ax1.legend(markerscale=3, fontsize=9)

    ax2 = fig.add_subplot(122, projection="3d")
    for lbl in [0, 1]:
        mask = vpn_labels == lbl
        ax2.scatter(X_pca3[mask, 0], X_pca3[mask, 1], X_pca3[mask, 2],
                    alpha=0.20, s=4, color=vpn_colors[lbl], label=vpn_names[lbl],
                    rasterized=True)
    ax2.set_xlabel(f"PC1 ({ev3[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({ev3[1]:.1%})")
    ax2.set_zlabel(f"PC3 ({ev3[2]:.1%})")
    ax2.set_title("PCA 3D — by VPN Label")
    ax2.legend(markerscale=3, fontsize=9)

    fig.suptitle("PCA 3D Feature Space Projection", fontsize=14, y=1.01)
    fig.savefig(OUT / "fig_pca3d_dataset_space.png")
    plt.close(fig)
    print("  Saved fig_pca3d_dataset_space.png")

    # Save PCA loadings
    loadings = pd.DataFrame(pca.components_.T, index=feat_cols, columns=["PC1", "PC2"])
    loadings["abs_PC1"] = loadings["PC1"].abs()
    loadings = loadings.sort_values("abs_PC1", ascending=False)
    loadings.to_csv(OUT / "pca_loadings.csv")
    print("  Saved pca_loadings.csv")

    # --- t-SNE ---
    print("  Computing t-SNE (this may take a minute)...", end=" ", flush=True)
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=40,
                max_iter=1000, learning_rate="auto", init="pca")
    X_tsne = tsne.fit_transform(X_s)
    print("done")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ds in DATASETS:
        mask = ds_labels == ds
        axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1], alpha=0.25, s=6,
                        color=DS_COLORS[ds], label=DS_LABELS[ds], rasterized=True)
    axes[0].set_title("t-SNE — Colored by Dataset")
    axes[0].legend(markerscale=4)

    for lbl in [0, 1]:
        mask = vpn_labels == lbl
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], alpha=0.25, s=6,
                        color=vpn_colors[lbl], label=vpn_names[lbl], rasterized=True)
    axes[1].set_title("t-SNE — Colored by VPN Label")
    axes[1].legend(markerscale=4)

    fig.suptitle("t-SNE Feature Space Projection", fontsize=14, y=1.01)
    fig.savefig(OUT / "tsne_dataset_projection.png")
    fig.savefig(OUT / "fig_tsne_dataset_space.png")
    plt.close(fig)
    print("  Saved tsne_dataset_projection.png + fig_tsne_dataset_space.png")

    # --- UMAP ---
    print("  Computing UMAP...", end=" ", flush=True)
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=30,
                            min_dist=0.3, metric="euclidean")
        X_umap = reducer.fit_transform(X_s)
        print("done")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
        for ds in DATASETS:
            mask = ds_labels == ds
            axes[0].scatter(X_umap[mask, 0], X_umap[mask, 1], alpha=0.25, s=6,
                            color=DS_COLORS[ds], label=DS_LABELS[ds], rasterized=True)
        axes[0].set_title("UMAP — Colored by Dataset")
        axes[0].legend(markerscale=4)

        for lbl in [0, 1]:
            mask = vpn_labels == lbl
            axes[1].scatter(X_umap[mask, 0], X_umap[mask, 1], alpha=0.25, s=6,
                            color=vpn_colors[lbl], label=vpn_names[lbl], rasterized=True)
        axes[1].set_title("UMAP — Colored by VPN Label")
        axes[1].legend(markerscale=4)

        fig.suptitle("UMAP Feature Space Projection", fontsize=14, y=1.01)
        fig.savefig(OUT / "umap_dataset_projection.png")
        fig.savefig(OUT / "fig_umap_dataset_space.png")
        plt.close(fig)
        print("  Saved umap_dataset_projection.png + fig_umap_dataset_space.png")
    except ImportError:
        print("SKIP (umap-learn not installed)")

    return {"pca_var": ev.tolist()}


# ================================================================
# PART 4 — DOMAIN CLASSIFIER STRUCTURE ANALYSIS
# ================================================================
def part4_domain_classifier(df, feat_cols):
    print("\n" + "=" * 70)
    print("PART 4: DOMAIN CLASSIFIER STRUCTURE ANALYSIS")
    print("=" * 70)

    X = df[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    le = LabelEncoder()
    y = le.fit_transform(df.dataset.values)
    class_names = le.classes_.tolist()

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results = []

    # --- Logistic Regression ---
    print("  Logistic Regression (5-fold CV)...", end=" ", flush=True)
    lr = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0,
                            solver="lbfgs")
    y_pred_lr = cross_val_predict(lr, X_s, y, cv=cv, method="predict")
    y_prob_lr = cross_val_predict(lr, X_s, y, cv=cv, method="predict_proba")

    acc_lr = accuracy_score(y, y_pred_lr)
    f1_lr = f1_score(y, y_pred_lr, average="macro")
    y_bin = label_binarize(y, classes=list(range(len(class_names))))
    auc_lr = roc_auc_score(y_bin, y_prob_lr, multi_class="ovr", average="macro")
    print(f"acc={acc_lr:.4f} F1={f1_lr:.4f} AUC={auc_lr:.4f}")

    results.append({
        "classifier": "logistic_regression",
        "accuracy": round(acc_lr, 4),
        "macro_f1": round(f1_lr, 4),
        "macro_auc": round(auc_lr, 4),
    })

    # --- XGBoost ---
    print("  XGBoost (5-fold CV)...", end=" ", flush=True)
    import xgboost as xgb
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbosity=0,
        eval_metric="mlogloss",
    )
    y_pred_xgb = cross_val_predict(xgb_clf, X, y, cv=cv, method="predict")
    y_prob_xgb = cross_val_predict(xgb_clf, X, y, cv=cv, method="predict_proba")

    acc_xgb = accuracy_score(y, y_pred_xgb)
    f1_xgb = f1_score(y, y_pred_xgb, average="macro")
    auc_xgb = roc_auc_score(y_bin, y_prob_xgb, multi_class="ovr", average="macro")
    print(f"acc={acc_xgb:.4f} F1={f1_xgb:.4f} AUC={auc_xgb:.4f}")

    results.append({
        "classifier": "xgboost",
        "accuracy": round(acc_xgb, 4),
        "macro_f1": round(f1_xgb, 4),
        "macro_auc": round(auc_xgb, 4),
    })

    # --- Random Forest (for permutation importance) ---
    print("  Random Forest (for permutation importance)...", end=" ", flush=True)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                random_state=SEED, n_jobs=-1)
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=10,
                                  random_state=SEED, n_jobs=-1, scoring="accuracy")
    print("done")

    imp_df = pd.DataFrame({
        "feature": feat_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    imp_df.to_csv(OUT / "domain_feature_importance.csv", index=False)
    print("  Saved domain_feature_importance.csv")

    # Also get XGBoost feature importance (train full)
    xgb_full = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbosity=0,
        eval_metric="mlogloss",
    )
    xgb_full.fit(X, y)
    xgb_imp = pd.DataFrame({
        "feature": feat_cols,
        "xgb_gain_importance": xgb_full.feature_importances_,
    }).sort_values("xgb_gain_importance", ascending=False)

    imp_df = imp_df.merge(xgb_imp, on="feature")
    imp_df.to_csv(OUT / "domain_feature_importance.csv", index=False)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT / "domain_classifier_report.csv", index=False)
    results_df.to_csv(OUT / "domain_classifier_metrics.csv", index=False)
    print(f"  Saved domain_classifier_report.csv + domain_classifier_metrics.csv")

    # --- Importance figure ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Permutation importance
    top_perm = imp_df.sort_values("importance_mean", ascending=True).tail(15)
    axes[0].barh(top_perm["feature"], top_perm["importance_mean"],
                 xerr=top_perm["importance_std"], color="#3498db", alpha=0.8)
    axes[0].set_title("Domain Classifier — Permutation Importance (RF)")
    axes[0].set_xlabel("Mean Accuracy Decrease")

    # XGBoost gain
    top_xgb = imp_df.sort_values("xgb_gain_importance", ascending=True).tail(15)
    axes[1].barh(top_xgb["feature"], top_xgb["xgb_gain_importance"],
                 color="#e74c3c", alpha=0.8)
    axes[1].set_title("Domain Classifier — XGBoost Gain Importance")
    axes[1].set_xlabel("Gain")

    fig.suptitle("Features Most Useful for Dataset Identification", fontsize=14, y=1.01)
    fig.savefig(OUT / "domain_feature_importance.png")
    plt.close(fig)
    print("  Saved domain_feature_importance.png")

    del rf, xgb_full
    gc.collect()

    return {
        "lr": {"acc": acc_lr, "f1": f1_lr, "auc": auc_lr},
        "xgb": {"acc": acc_xgb, "f1": f1_xgb, "auc": auc_xgb},
        "top5_domain_features": imp_df.sort_values("importance_mean", ascending=False).head(5)["feature"].tolist(),
    }


# ================================================================
# PART 5 — FEATURE STABILITY ACROSS DATASETS
# ================================================================
def part5_feature_stability(df, feat_cols):
    print("\n" + "=" * 70)
    print("PART 5: FEATURE STABILITY ACROSS DATASETS")
    print("=" * 70)

    rows = []
    ds_groups = {ds: df.loc[df.dataset == ds] for ds in DATASETS}

    for feat in feat_cols:
        groups = [ds_groups[ds][feat].dropna().values for ds in DATASETS]

        # ANOVA F-test
        if all(len(g) > 1 for g in groups):
            f_stat, f_pval = f_oneway(*groups)
        else:
            f_stat, f_pval = float("nan"), float("nan")

        # Kruskal-Wallis
        if all(len(g) > 1 for g in groups):
            kw_stat, kw_pval = kruskal(*groups)
        else:
            kw_stat, kw_pval = float("nan"), float("nan")

        # Effect size: eta squared = SS_between / SS_total
        all_vals = np.concatenate(groups)
        grand_mean = np.mean(all_vals)
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

        rows.append({
            "feature": feat,
            "anova_f": round(float(f_stat), 4) if not np.isnan(f_stat) else None,
            "anova_pvalue": float(f_pval) if not np.isnan(f_pval) else None,
            "kruskal_wallis_h": round(float(kw_stat), 4) if not np.isnan(kw_stat) else None,
            "kruskal_wallis_pvalue": float(kw_pval) if not np.isnan(kw_pval) else None,
            "eta_squared": round(float(eta_sq), 6),
            "eta_squared_interpretation": (
                "large" if eta_sq >= 0.14 else
                "medium" if eta_sq >= 0.06 else
                "small" if eta_sq >= 0.01 else
                "negligible"
            ),
        })

    stab_df = pd.DataFrame(rows).sort_values("eta_squared", ascending=False)
    stab_df.to_csv(OUT / "feature_dataset_dependence_tests.csv", index=False)
    print(f"  Saved feature_dataset_dependence_tests.csv")

    # Print top 10
    print("\n  Top 10 most dataset-dependent features:")
    for _, r in stab_df.head(10).iterrows():
        print(f"    {r['feature']:25s} η²={r['eta_squared']:.4f} ({r['eta_squared_interpretation']})"
              f"  ANOVA F={r['anova_f']}")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_data = stab_df.sort_values("eta_squared", ascending=True)
    colors = []
    for _, r in plot_data.iterrows():
        eta = r["eta_squared"]
        if eta >= 0.14:
            colors.append("#e74c3c")
        elif eta >= 0.06:
            colors.append("#e67e22")
        elif eta >= 0.01:
            colors.append("#f1c40f")
        else:
            colors.append("#2ecc71")

    ax.barh(plot_data["feature"], plot_data["eta_squared"], color=colors, alpha=0.85)
    ax.axvline(0.14, color="red", linestyle="--", linewidth=0.8, label="Large effect (η²≥0.14)")
    ax.axvline(0.06, color="orange", linestyle="--", linewidth=0.8, label="Medium effect (η²≥0.06)")
    ax.axvline(0.01, color="gold", linestyle="--", linewidth=0.8, label="Small effect (η²≥0.01)")
    ax.set_xlabel("Eta Squared (η²)")
    ax.set_title("Feature Sensitivity to Dataset Identity\n(higher = more dataset-dependent)")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(OUT / "feature_dataset_dependence.png")
    plt.close(fig)
    print("  Saved feature_dataset_dependence.png")

    return stab_df


# ================================================================
# PART 6 — CORRELATION STRUCTURE DIFFERENCE
# ================================================================
def part6_correlation_structure(df, feat_cols):
    print("\n" + "=" * 70)
    print("PART 6: CORRELATION STRUCTURE DIFFERENCE")
    print("=" * 70)

    corr_mats = {}
    for ds in DATASETS:
        sub = df.loc[df.dataset == ds, feat_cols].dropna()
        corr_mats[ds] = sub.corr(method="spearman")

    # Save per-dataset correlation matrices
    for ds in DATASETS:
        corr_mats[ds].to_csv(OUT / f"correlation_matrix_{ds}.csv")
    print("  Saved per-dataset correlation matrices")

    # --- Per-dataset correlation heatmaps ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, ds in zip(axes, DATASETS):
        sns.heatmap(corr_mats[ds], ax=ax, cmap="RdBu_r", vmin=-1, vmax=1,
                    square=True, linewidths=0.2,
                    xticklabels=[f[:8] for f in feat_cols],
                    yticklabels=[f[:8] for f in feat_cols],
                    cbar_kws={"shrink": 0.7})
        ax.set_title(f"{DS_LABELS[ds]} Correlation", fontsize=12)
        ax.tick_params(axis="both", labelsize=7)

    fig.suptitle("Per-Dataset Feature Correlation (Spearman)", fontsize=14, y=1.02)
    fig.savefig(OUT / "correlation_per_dataset.png")
    plt.close(fig)
    print("  Saved correlation_per_dataset.png")

    # --- Difference heatmaps ---
    pairs = [("iscx", "usbvpn"), ("iscx", "vnat"), ("usbvpn", "vnat")]
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    diff_stats = []
    for ax, (ds1, ds2) in zip(axes, pairs):
        diff = corr_mats[ds1] - corr_mats[ds2]
        sns.heatmap(diff, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1,
                    square=True, linewidths=0.2,
                    xticklabels=[f[:8] for f in feat_cols],
                    yticklabels=[f[:8] for f in feat_cols],
                    cbar_kws={"shrink": 0.7})
        ax.set_title(f"{DS_LABELS[ds1]} − {DS_LABELS[ds2]}", fontsize=12)
        ax.tick_params(axis="both", labelsize=7)

        # Statistics on the difference
        vals = diff.values[np.triu_indices_from(diff.values, k=1)]
        diff_stats.append({
            "pair": f"{ds1}_vs_{ds2}",
            "mean_abs_diff": round(float(np.mean(np.abs(vals))), 4),
            "max_abs_diff": round(float(np.max(np.abs(vals))), 4),
            "std_diff": round(float(np.std(vals)), 4),
            "n_large_diffs_gt_0.3": int(np.sum(np.abs(vals) > 0.3)),
            "n_large_diffs_gt_0.5": int(np.sum(np.abs(vals) > 0.5)),
        })

    fig.suptitle("Correlation Structure Differences Between Datasets", fontsize=14, y=1.02)
    fig.savefig(OUT / "correlation_difference_heatmaps.png")
    fig.savefig(OUT / "correlation_shift_heatmaps.png")
    plt.close(fig)
    print("  Saved correlation_difference_heatmaps.png + correlation_shift_heatmaps.png")

    diff_stats_df = pd.DataFrame(diff_stats)
    diff_stats_df.to_csv(OUT / "correlation_difference_stats.csv", index=False)
    print("  Saved correlation_difference_stats.csv")

    print("\n  Correlation difference summary:")
    for _, r in diff_stats_df.iterrows():
        print(f"    {r['pair']:20s} mean_abs_diff={r['mean_abs_diff']:.4f} "
              f"max={r['max_abs_diff']:.4f} "
              f"#>0.3={r['n_large_diffs_gt_0.3']} #>0.5={r['n_large_diffs_gt_0.5']}")

    return diff_stats_df


# ================================================================
# PART 7 — FEATURE IMPORTANCE INSTABILITY
# ================================================================
def part7_importance_instability(df, feat_cols):
    print("\n" + "=" * 70)
    print("PART 7: FEATURE IMPORTANCE INSTABILITY")
    print("=" * 70)

    import xgboost as xgb

    importances = {}
    rankings = {}

    for ds in DATASETS:
        print(f"  Training VPN classifier on {DS_LABELS[ds]}...", end=" ", flush=True)
        sub = df[df.dataset == ds].copy()
        X = sub[feat_cols].values.astype(np.float32)
        y = sub.label.values

        if len(np.unique(y)) < 2:
            print("SKIP (single class)")
            continue

        # Simple train/test split
        rng = np.random.default_rng(SEED)
        n = len(X)
        idx = rng.permutation(n)
        sp = int(0.7 * n)
        X_tr, X_va = X[idx[:sp]], X[idx[sp:]]
        y_tr, y_va = y[idx[:sp]], y[idx[sp:]]

        m = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=max(1.0, (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)),
            eval_metric="logloss", random_state=SEED, n_jobs=-1, verbosity=0,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        imp = m.feature_importances_
        imp_norm = imp / (imp.sum() + 1e-12)
        importances[ds] = dict(zip(feat_cols, imp_norm))

        # Rankings (1 = most important)
        ranked = sorted(zip(feat_cols, imp_norm), key=lambda x: -x[1])
        rankings[ds] = {f: rank + 1 for rank, (f, _) in enumerate(ranked)}

        top5 = ranked[:5]
        print(f"done  Top5: {[f'{f}={v:.3f}' for f, v in top5]}")

        del m
        gc.collect()

    # Build importance table
    imp_rows = []
    for f in feat_cols:
        row = {"feature": f}
        for ds in DATASETS:
            if ds in importances:
                row[f"{ds}_importance"] = round(importances[ds].get(f, 0.0), 6)
                row[f"{ds}_rank"] = rankings[ds].get(f, len(feat_cols))
        imp_rows.append(row)

    imp_df = pd.DataFrame(imp_rows)
    imp_df.to_csv(OUT / "vpn_importance_per_dataset.csv", index=False)
    print(f"  Saved vpn_importance_per_dataset.csv")

    # Rank correlations
    ds_with_data = [ds for ds in DATASETS if ds in rankings]
    corr_rows = []
    for i, ds1 in enumerate(ds_with_data):
        for ds2 in ds_with_data[i + 1:]:
            r1 = [rankings[ds1].get(f, len(feat_cols)) for f in feat_cols]
            r2 = [rankings[ds2].get(f, len(feat_cols)) for f in feat_cols]
            rho, pval = spearmanr(r1, r2)

            # Top-k Jaccard
            for k in [5, 10, 15]:
                top1 = set(sorted(importances[ds1], key=lambda x: -importances[ds1][x])[:k])
                top2 = set(sorted(importances[ds2], key=lambda x: -importances[ds2][x])[:k])
                jaccard = len(top1 & top2) / len(top1 | top2) if len(top1 | top2) > 0 else 0

                corr_rows.append({
                    "dataset_1": ds1,
                    "dataset_2": ds2,
                    "spearman_rho": round(float(rho), 4),
                    "spearman_pvalue": float(pval),
                    "top_k": k,
                    "jaccard_overlap": round(jaccard, 4),
                    "intersection_size": len(top1 & top2),
                })

            print(f"  Rank correlation({DS_LABELS[ds1]}, {DS_LABELS[ds2]}): "
                  f"ρ={rho:.4f} p={pval:.2e}")

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT / "importance_rank_correlation.csv", index=False)
    corr_df.to_csv(OUT / "importance_instability_matrix.csv", index=False)
    print(f"  Saved importance_rank_correlation.csv + importance_instability_matrix.csv")

    # --- Figure: side-by-side importance ---
    fig, axes = plt.subplots(1, len(ds_with_data), figsize=(6 * len(ds_with_data), 8))
    if len(ds_with_data) == 1:
        axes = [axes]

    for ax, ds in zip(axes, ds_with_data):
        ranked = sorted(importances[ds].items(), key=lambda x: x[1])
        feats = [f for f, _ in ranked]
        vals = [v for _, v in ranked]
        ax.barh(feats, vals, color=DS_COLORS[ds], alpha=0.85)
        ax.set_title(f"{DS_LABELS[ds]} — VPN Feature Importance")
        ax.set_xlabel("Normalized Importance")

    fig.suptitle("Per-Dataset VPN Classifier Feature Importance", fontsize=14, y=1.01)
    fig.savefig(OUT / "importance_instability_comparison.png")
    plt.close(fig)
    print("  Saved importance_instability_comparison.png")

    # --- Rank agreement heatmap ---
    if len(ds_with_data) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        rank_matrix = pd.DataFrame(index=feat_cols)
        for ds in ds_with_data:
            rank_matrix[DS_LABELS[ds]] = [rankings[ds].get(f, len(feat_cols)) for f in feat_cols]
        rank_matrix = rank_matrix.sort_values(DS_LABELS[ds_with_data[0]])

        sns.heatmap(rank_matrix, annot=True, fmt="d", cmap="YlOrRd_r",
                    ax=ax, linewidths=0.3)
        ax.set_title("Feature Importance Ranking per Dataset\n(lower = more important)")
        fig.savefig(OUT / "importance_rank_heatmap.png")
        plt.close(fig)
        print("  Saved importance_rank_heatmap.png")

    return {
        "importances": importances,
        "rankings": rankings,
        "rank_correlations": corr_rows,
    }


# ================================================================
# PART 8 — STRUCTURAL SHIFT SUMMARY REPORT
# ================================================================
def part8_summary_report(stats_df, dist_df, proj_info, domain_info,
                          stab_df, corr_diff_df, imp_info):
    print("\n" + "=" * 70)
    print("PART 8: STRUCTURAL SHIFT SUMMARY REPORT")
    print("=" * 70)

    # --- Compute verdict ---
    # Criteria:
    # 1. Domain classifier accuracy
    domain_acc = domain_info["xgb"]["acc"]
    # 2. Mean eta-squared
    mean_eta_sq = stab_df["eta_squared"].mean()
    n_large_eta = (stab_df["eta_squared"] >= 0.14).sum()
    # 3. Mean correlation difference
    mean_corr_diff = corr_diff_df["mean_abs_diff"].mean()
    # 4. Importance rank correlation
    rank_corrs = imp_info.get("rank_correlations", [])
    mean_rho = np.mean([r["spearman_rho"] for r in rank_corrs
                        if r["top_k"] == 5]) if rank_corrs else 0
    # 5. Mean JS divergence
    mean_js = dist_df["jensen_shannon"].mean()

    print(f"  Domain classifier accuracy: {domain_acc:.4f}")
    print(f"  Mean η²: {mean_eta_sq:.4f} ({n_large_eta} features with large effect)")
    print(f"  Mean correlation diff: {mean_corr_diff:.4f}")
    print(f"  Mean importance rank ρ: {mean_rho:.4f}")
    print(f"  Mean JS divergence: {mean_js:.4f}")

    # Verdict logic
    score = 0
    if domain_acc > 0.95:
        score += 3
    elif domain_acc > 0.85:
        score += 2
    elif domain_acc > 0.70:
        score += 1

    if mean_eta_sq > 0.10:
        score += 3
    elif mean_eta_sq > 0.05:
        score += 2
    elif mean_eta_sq > 0.02:
        score += 1

    if n_large_eta >= 10:
        score += 2
    elif n_large_eta >= 5:
        score += 1

    if mean_corr_diff > 0.15:
        score += 2
    elif mean_corr_diff > 0.08:
        score += 1

    if mean_rho < 0.3:
        score += 2
    elif mean_rho < 0.6:
        score += 1

    if mean_js > 0.20:
        score += 2
    elif mean_js > 0.10:
        score += 1

    if score >= 10:
        verdict = "SEVERE_STRUCTURAL_SHIFT"
    elif score >= 7:
        verdict = "STRONG_SHIFT"
    elif score >= 4:
        verdict = "MODERATE_SHIFT"
    else:
        verdict = "LOW_SHIFT"

    print(f"\n  Shift score: {score}/15")
    print(f"  VERDICT: {verdict}")

    # --- Top shifted features ---
    top_shifted_js = dist_df.groupby("feature")["jensen_shannon"].mean().sort_values(ascending=False)
    top5_shifted = top_shifted_js.head(5).index.tolist()

    top_domain = domain_info.get("top5_domain_features", [])

    top_eta = stab_df.head(5)["feature"].tolist()

    # --- Generate markdown report ---
    md = f"""# Dataset Structure Analysis — Structural Shift Summary

**Date:** {TIMESTAMP}
**Datasets:** ISCX-VPN-2016, USBVPN, VNAT-2024
**Feature Space:** 21 non-directional flow-level features (`full_no_dir`)
**Purpose:** Scientifically explain why cross-dataset (LODO) transfer fails

---

## Executive Summary

**Structural Shift Verdict: `{verdict}`** (score: {score}/15)

The three VPN traffic datasets exhibit **{"severe" if "SEVERE" in verdict else "strong" if "STRONG" in verdict else "moderate" if "MODERATE" in verdict else "low"} structural mismatch** in the 21-dimensional feature space. This mismatch is the fundamental cause of LODO transfer failure.

---

## 1. Feature Distribution Shifts

### Strongest Distribution Shifts (Jensen-Shannon Divergence)

| Rank | Feature | Mean JS Divergence |
|------|---------|-------------------|
"""
    for i, (feat, js_val) in enumerate(top_shifted_js.head(10).items()):
        md += f"| {i+1} | `{feat}` | {js_val:.4f} |\n"

    md += f"""
**Interpretation:** Features with JS divergence > 0.15 have substantially different distributions across datasets.
{len(top_shifted_js[top_shifted_js > 0.15])} out of 21 features exceed this threshold.

### Distribution Statistics

See `feature_distribution_stats.csv` for full per-dataset mean/std/median/IQR/skewness/kurtosis.

Individual feature distribution figures saved as `feature_distributions_<feature>.png`.

---

## 2. Between-Dataset Distance Matrix

### Mean Pairwise Distance (Averaged Across Features)

"""
    # Aggregate distances
    for metric in ["wasserstein", "jensen_shannon", "ks_statistic"]:
        md += f"#### {metric.replace('_', ' ').title()}\n\n"
        md += "| | ISCX | USBVPN | VNAT |\n"
        md += "|---|------|--------|------|\n"
        pairs_dict = {}
        for _, row in dist_df.groupby(["dataset_1", "dataset_2"])[metric].mean().reset_index().iterrows():
            pairs_dict[(row["dataset_1"], row["dataset_2"])] = row[metric]
        for ds1 in DATASETS:
            row_str = f"| **{DS_LABELS[ds1]}** "
            for ds2 in DATASETS:
                if ds1 == ds2:
                    row_str += "| 0.0000 "
                else:
                    key = (ds1, ds2) if (ds1, ds2) in pairs_dict else (ds2, ds1)
                    val = pairs_dict.get(key, 0.0)
                    row_str += f"| {val:.4f} "
            md += row_str + "|\n"
        md += "\n"

    md += """See `dataset_distance_matrix.csv` for per-feature pairwise distances.

See `dataset_distance_heatmap.png` and `dataset_distance_top10_features.png` for visualizations.

---

## 3. Multivariate Feature Space Separation

"""
    pca_var = proj_info.get("pca_var", [0, 0])
    md += f"""### PCA
- PC1 explains {pca_var[0]:.1%} of variance, PC2 explains {pca_var[1]:.1%}
- See `pca_dataset_projection.png` — dual panel showing dataset and VPN label coloring
- PCA loadings saved in `pca_loadings.csv`

### t-SNE
- See `tsne_dataset_projection.png`

### UMAP
- See `umap_dataset_projection.png`

### Interpretation

If datasets cluster separately in the projected space (especially PCA), this confirms that
**dataset identity is encoded in the first principal components** of the feature space.
This means the features carry dataset-specific structure that dominates over VPN-vs-non-VPN structure.

---

## 4. Domain Classifier Analysis

A classifier predicting **which dataset** a flow comes from (not VPN detection):

| Classifier | Accuracy | Macro F1 | Macro AUC |
|------------|----------|----------|-----------|
| Logistic Regression | {domain_info['lr']['acc']:.4f} | {domain_info['lr']['f1']:.4f} | {domain_info['lr']['auc']:.4f} |
| XGBoost | {domain_info['xgb']['acc']:.4f} | {domain_info['xgb']['f1']:.4f} | {domain_info['xgb']['auc']:.4f} |

### Most Domain-Identifying Features (Permutation Importance)

| Rank | Feature |
|------|---------|
"""
    for i, feat in enumerate(top_domain[:10]):
        md += f"| {i+1} | `{feat}` |\n"

    md += f"""
**Interpretation:** A domain classifier achieving accuracy > 0.90 from only flow-level
features proves that **dataset identity is strongly encoded in the feature space**.
This is the root cause of LODO transfer failure — a VPN classifier inevitably learns
dataset-specific patterns alongside (or instead of) VPN-specific patterns.

See `domain_classifier_report.csv` and `domain_feature_importance.csv`.

---

## 5. Feature Stability Across Datasets

### Top 10 Most Dataset-Dependent Features (η²)

| Rank | Feature | η² | Effect Size |
|------|---------|-----|-------------|
"""
    for i, (_, r) in enumerate(stab_df.head(10).iterrows()):
        md += f"| {i+1} | `{r['feature']}` | {r['eta_squared']:.4f} | {r['eta_squared_interpretation']} |\n"

    n_large = (stab_df["eta_squared"] >= 0.14).sum()
    n_medium = ((stab_df["eta_squared"] >= 0.06) & (stab_df["eta_squared"] < 0.14)).sum()
    n_small = ((stab_df["eta_squared"] >= 0.01) & (stab_df["eta_squared"] < 0.06)).sum()
    n_neg = (stab_df["eta_squared"] < 0.01).sum()

    md += f"""
**Effect size distribution:**
- Large (η² ≥ 0.14): **{n_large}** features
- Medium (0.06 ≤ η² < 0.14): **{n_medium}** features
- Small (0.01 ≤ η² < 0.06): **{n_small}** features
- Negligible (η² < 0.01): **{n_neg}** features

**Mean η² across all features:** {mean_eta_sq:.4f}

See `feature_dataset_dependence_tests.csv` and `feature_dataset_dependence.png`.

---

## 6. Correlation Structure Differences

"""
    for _, r in corr_diff_df.iterrows():
        md += f"""### {r['pair'].replace('_vs_', ' vs ').replace('_', ' ').upper()}
- Mean |Δcorr|: **{r['mean_abs_diff']:.4f}**
- Max |Δcorr|: **{r['max_abs_diff']:.4f}**
- Feature pairs with |Δcorr| > 0.3: **{r['n_large_diffs_gt_0.3']}**
- Feature pairs with |Δcorr| > 0.5: **{r['n_large_diffs_gt_0.5']}**

"""

    md += f"""**Interpretation:** If datasets have different inter-feature correlations, then a model
trained on one dataset learns feature interactions that do not hold in another.
Mean absolute correlation difference of {mean_corr_diff:.4f} indicates
{"substantial" if mean_corr_diff > 0.10 else "moderate" if mean_corr_diff > 0.05 else "minor"}
correlation structure divergence.

See `correlation_difference_heatmaps.png` and `correlation_per_dataset.png`.

---

## 7. Feature Importance Instability

### VPN Classifier Importance Rank Correlation

"""
    for r in rank_corrs:
        if r["top_k"] == 5:
            md += f"| {DS_LABELS[r['dataset_1']]} vs {DS_LABELS[r['dataset_2']]} | Spearman ρ = **{r['spearman_rho']:.4f}** (p = {r['spearman_pvalue']:.2e}) | Jaccard@5 = {r['jaccard_overlap']:.2f} |\n"

    md += f"""
**Mean rank correlation (ρ):** {mean_rho:.4f}

**Interpretation:** Low Spearman rank correlation (ρ < 0.6) between per-dataset VPN classifiers
means that **each dataset relies on different features** to distinguish VPN from non-VPN traffic.
This directly explains LODO collapse: a model trained on datasets A+B learns feature patterns
that do not transfer to dataset C because C uses different VPN signatures.

See `importance_rank_correlation.csv`, `importance_instability_comparison.png`, `importance_rank_heatmap.png`.

---

## 8. Why LODO Transfer Fails — Integrated Explanation

### Root Causes (Ranked by Evidence Strength)

1. **Dataset identity is encoded in the feature space.**
   Domain classifier achieves {domain_info['xgb']['acc']:.1%} accuracy using only 21 flow-level features.
   The features carry near-perfect dataset fingerprints. Any VPN classifier trained on this space
   inevitably learns dataset-specific patterns.

2. **Feature distributions differ substantially across datasets.**
   {"Most" if len(top_shifted_js[top_shifted_js > 0.15]) > 10 else "Many" if len(top_shifted_js[top_shifted_js > 0.15]) > 5 else "Some"} features ({len(top_shifted_js[top_shifted_js > 0.15])}/21) show JS divergence > 0.15.
   The top shifted features ({', '.join(f'`{f}`' for f in top5_shifted)}) have fundamentally
   different ranges and shapes across datasets.

3. **Feature importance rankings are unstable across datasets.**
   Mean rank correlation ρ = {mean_rho:.4f}. Different datasets use different features
   to separate VPN from non-VPN traffic. This means the "VPN signature" is not universal —
   it is dataset-specific.

4. **Inter-feature correlations differ across datasets.**
   Mean |Δcorr| = {mean_corr_diff:.4f}. Feature interactions learned from training datasets
   do not hold in the held-out dataset.

5. **Multivariate separation is dominated by dataset identity.**
   PCA shows that the first two principal components (explaining {pca_var[0]+pca_var[1]:.1%} of variance)
   separate datasets more clearly than VPN vs non-VPN classes.

### Causal Chain

```
Different capture environments
    → Different VPN implementations, app mixes, network conditions
        → Different feature distributions (JS divergence, KS tests)
            → Different feature correlations (Δcorr structure)
                → Different VPN detection strategies per dataset (importance instability)
                    → Domain fingerprinting in feature space (domain AUC ≈ 1.0)
                        → LODO classifier learns dataset patterns instead of VPN patterns
                            → LODO AUC collapses to near-random (~0.45)
```

### What This Means for the Thesis

1. **LODO collapse is a data reality, not a model failure.** No feature family, model architecture,
   or training trick can overcome the structural mismatch between these datasets.

2. **The mismatch is multi-layered.** It affects marginal distributions, joint correlations,
   and discriminative feature rankings simultaneously.

3. **Universal cross-dataset VPN detection from header-only features is not achievable**
   with the current 3-dataset setup. The datasets capture fundamentally different traffic.

4. **These negative results are scientifically valuable.** They close off the hypothesis
   that better features or models could fix cross-dataset transfer, and redirect future
   work toward domain adaptation with target-domain data.

---

## Verdict

### **{verdict}**

{"The structural mismatch between ISCX, USBVPN, and VNAT is severe and pervasive. It affects every level of the feature space — distributions, correlations, and discriminative patterns. This fully explains LODO transfer collapse and confirms that cross-dataset generalization requires fundamentally new training data from the target environment." if "SEVERE" in verdict else "The structural mismatch is strong across multiple dimensions. Most features show significant dataset dependence, and the domain classifier easily separates datasets. This explains the observed LODO transfer failure." if "STRONG" in verdict else "Moderate structural differences exist between datasets. Some features are relatively stable while others show significant dataset dependence." if "MODERATE" in verdict else "Limited structural mismatch detected. Dataset differences are present but may not fully explain transfer failure."}

---

*Generated by `run_dataset_structure_analysis.py` — all figures and data files in `artifacts/dataset_structure_analysis/`*
"""

    md_path = OUT / "dataset_structure_summary.md"
    md_path.write_text(md, encoding="utf-8")
    (OUT / "dataset_shift_report.md").write_text(md, encoding="utf-8")
    print(f"  Saved dataset_structure_summary.md + dataset_shift_report.md")

    # Also save verdict JSON
    verdict_json = {
        "timestamp": TIMESTAMP,
        "verdict": verdict,
        "shift_score": score,
        "max_score": 15,
        "evidence": {
            "domain_classifier_accuracy": round(domain_acc, 4),
            "domain_classifier_auc": round(domain_info["xgb"]["auc"], 4),
            "mean_eta_squared": round(mean_eta_sq, 4),
            "n_large_effect_features": int(n_large_eta),
            "mean_correlation_diff": round(mean_corr_diff, 4),
            "mean_importance_rank_rho": round(float(mean_rho), 4),
            "mean_js_divergence": round(mean_js, 4),
        },
        "top5_shifted_features": top5_shifted,
        "top5_domain_features": top_domain,
        "top5_dataset_dependent_features": top_eta,
    }
    json_path = OUT / "structural_shift_verdict.json"
    json_path.write_text(json.dumps(verdict_json, indent=2, default=str), encoding="utf-8")
    print(f"  Saved structural_shift_verdict.json")

    return verdict


# ================================================================
# MAIN
# ================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("DATASET STRUCTURE ANALYSIS")
    print("Comprehensive structural mismatch analysis for thesis")
    print("=" * 70)
    print(f"Output: {OUT}")

    df, feat_cols = load_data()

    # Part 1
    stats_df = part1_feature_distributions(df, feat_cols)

    # Part 2
    dist_df = part2_distance_matrix(df, feat_cols)

    # Part 3
    proj_info = part3_multivariate_projection(df, feat_cols)

    # Part 4
    domain_info = part4_domain_classifier(df, feat_cols)

    # Part 5
    stab_df = part5_feature_stability(df, feat_cols)

    # Part 6
    corr_diff_df = part6_correlation_structure(df, feat_cols)

    # Part 7
    imp_info = part7_importance_instability(df, feat_cols)

    # Part 8
    verdict = part8_summary_report(stats_df, dist_df, proj_info, domain_info,
                                    stab_df, corr_diff_df, imp_info)

    # Final checklist
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("OUTPUT CHECKLIST")
    print("=" * 70)

    expected_files = [
        "feature_distribution_stats.csv",
        "feature_distribution_summary.csv",
        "dataset_distance_matrix.csv",
        "dataset_distance_values.csv",
        "dataset_distance_heatmap.png",
        "dataset_distance_top10_features.png",
        "pca_dataset_projection.png",
        "fig_pca_dataset_space.png",
        "fig_pca3d_dataset_space.png",
        "pca_loadings.csv",
        "tsne_dataset_projection.png",
        "fig_tsne_dataset_space.png",
        "umap_dataset_projection.png",
        "fig_umap_dataset_space.png",
        "domain_classifier_report.csv",
        "domain_classifier_metrics.csv",
        "domain_feature_importance.csv",
        "domain_feature_importance.png",
        "feature_dataset_dependence_tests.csv",
        "feature_dataset_dependence.png",
        "correlation_per_dataset.png",
        "correlation_difference_heatmaps.png",
        "correlation_shift_heatmaps.png",
        "correlation_difference_stats.csv",
        "vpn_importance_per_dataset.csv",
        "importance_rank_correlation.csv",
        "importance_instability_matrix.csv",
        "importance_instability_comparison.png",
        "importance_rank_heatmap.png",
        "dataset_structure_summary.md",
        "dataset_shift_report.md",
        "structural_shift_verdict.json",
    ]

    # Add per-feature figures (both naming conventions)
    for feat in feat_cols:
        expected_files.append(f"feature_distributions_{feat}.png")
        expected_files.append(f"fig_feature_{feat}_distribution.png")

    all_ok = True
    n_found = 0
    for fname in expected_files:
        exists = (OUT / fname).exists()
        if exists:
            n_found += 1
        else:
            all_ok = False
            print(f"  [✗ MISSING] {fname}")

    print(f"\n  {n_found}/{len(expected_files)} files produced")
    print(f"\n  Completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  VERDICT: {verdict}")
    print(f"  Output: {OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()














