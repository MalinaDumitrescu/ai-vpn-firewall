"""Regenerate thesis-ready sign-reversal artifacts from the forensic audit.

Source of truth:
    artifacts/sign_reversal_forensic_audit/sign_reversal_final_verdict.csv
    artifacts/sign_reversal_forensic_audit/feature_effects_by_dataset.csv
    artifacts/sign_reversal_forensic_audit/reversal_preprocessing_comparison.csv
    artifacts/sign_reversal_forensic_audit/reversal_vs_domain_fingerprint.csv
    artifacts/sign_reversal_forensic_audit/feature_effect_strength_report.csv
    artifacts/sign_reversal_forensic_audit/feature_reversal_summary.csv
    artifacts/sign_reversal_forensic_audit/feature_sign_matrix_mean.csv
    artifacts/sign_reversal_forensic_audit/strict_vs_loose_reversal_report.csv

This script regenerates only sign-reversal artifacts. It does not touch
unrelated figures/tables.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "sign_reversal_forensic_audit"
FIG = AUDIT / "figures"
TAB = AUDIT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

DATASETS = ["iscx", "usbvpn", "vnat"]

CATEGORY_ORDER = [
    "VERIFIED REVERSAL",
    "LIKELY REVERSAL BUT SENSITIVE",
    "POSSIBLE ARTIFACT",
    "NO REAL REVERSAL",
    "INCONCLUSIVE",
]
CATEGORY_COLOR = {
    "VERIFIED REVERSAL": "#1b7837",            # strong green
    "LIKELY REVERSAL BUT SENSITIVE": "#fdae61", # amber
    "POSSIBLE ARTIFACT": "#d73027",             # red
    "NO REAL REVERSAL": "#bdbdbd",
    "INCONCLUSIVE": "#cccccc",
}

warnings: List[str] = []


def _require(path: Path) -> Path:
    if not path.exists():
        warnings.append(f"MISSING required source file: {path.relative_to(ROOT)}")
    return path


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Load source of truth
# ---------------------------------------------------------------------------
verdict = pd.read_csv(_require(AUDIT / "sign_reversal_final_verdict.csv"))
effects = pd.read_csv(_require(AUDIT / "feature_effects_by_dataset.csv"))
prep = pd.read_csv(_require(AUDIT / "reversal_preprocessing_comparison.csv"))
domain = pd.read_csv(_require(AUDIT / "reversal_vs_domain_fingerprint.csv"))
strength = pd.read_csv(_require(AUDIT / "feature_effect_strength_report.csv"))
rev_summary = pd.read_csv(_require(AUDIT / "feature_reversal_summary.csv"))
sign_mean = pd.read_csv(_require(AUDIT / "feature_sign_matrix_mean.csv"))
strict_loose = pd.read_csv(_require(AUDIT / "strict_vs_loose_reversal_report.csv"))

# Use only raw-space `all_flows` / `raw` rows for dataset-level signs.
effects_raw = effects[
    (effects["analysis_name"] == "all_flows") & (effects["transform_name"] == "raw")
].copy()

# Category counts from source of truth.
counts = verdict["final_category"].value_counts().to_dict()
for c in CATEGORY_ORDER:
    counts.setdefault(c, 0)
print("category counts:", counts)


# ---------------------------------------------------------------------------
# Task 1 — sign_reversal_heatmap.png
# ---------------------------------------------------------------------------
def fig_sign_reversal_heatmap() -> Path:
    feats = verdict["feature"].tolist()
    # Order: VERIFIED, LIKELY, POSSIBLE ARTIFACT (alphabetical within each)
    order_key = {"VERIFIED REVERSAL": 0, "LIKELY REVERSAL BUT SENSITIVE": 1,
                 "POSSIBLE ARTIFACT": 2, "NO REAL REVERSAL": 3, "INCONCLUSIVE": 4}
    ordered = (verdict.assign(_k=verdict["final_category"].map(order_key))
                      .sort_values(["_k", "feature"]))
    feats = ordered["feature"].tolist()
    cats = ordered["final_category"].tolist()

    mat = np.zeros((len(feats), len(DATASETS)), dtype=int)
    for i, f in enumerate(feats):
        for j, d in enumerate(DATASETS):
            sub = effects_raw[(effects_raw["feature"] == f) & (effects_raw["dataset"] == d)]
            if sub.empty:
                mat[i, j] = 0
            else:
                mat[i, j] = int(np.sign(sub["diff_mean"].iloc[0]))

    fig, ax = plt.subplots(figsize=(9.0, 8.2))
    cmap = ListedColormap(["#2166ac", "#f7f7f7", "#b2182b"])  # -1 blue, 0 grey, +1 red
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([d.upper() for d in DATASETS])
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)

    # Colour-coded feature labels by final_category.
    for tick, cat in zip(ax.get_yticklabels(), cats):
        tick.set_color(CATEGORY_COLOR.get(cat, "#000000"))

    # Right-side category annotation column.
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(feats)))
    short = {
        "VERIFIED REVERSAL": "VERIFIED",
        "LIKELY REVERSAL BUT SENSITIVE": "LIKELY",
        "POSSIBLE ARTIFACT": "ARTIFACT",
        "NO REAL REVERSAL": "NONE",
        "INCONCLUSIVE": "INCONC.",
    }
    ax2.set_yticklabels([short.get(c, c) for c in cats])
    for tick, cat in zip(ax2.get_yticklabels(), cats):
        tick.set_color(CATEGORY_COLOR.get(cat, "#000000"))
        tick.set_fontweight("bold")

    # Annotate cells with sign.
    for i in range(len(feats)):
        for j in range(len(DATASETS)):
            s = mat[i, j]
            lab = {-1: "-", 0: ".", 1: "+"}[s]
            ax.text(j, i, lab, ha="center", va="center",
                    color="white" if s != 0 else "#444444", fontsize=10, fontweight="bold")

    ax.set_title(
        "Raw-space class-conditional effect direction\n"
        "(sign of mean(VPN) - mean(nonVPN); rows coloured by forensic final category)"
    )
    legend = [
        Patch(color=CATEGORY_COLOR["VERIFIED REVERSAL"], label=f"VERIFIED REVERSAL (n={counts['VERIFIED REVERSAL']})"),
        Patch(color=CATEGORY_COLOR["LIKELY REVERSAL BUT SENSITIVE"],
              label=f"LIKELY REVERSAL BUT SENSITIVE (n={counts['LIKELY REVERSAL BUT SENSITIVE']})"),
        Patch(color=CATEGORY_COLOR["POSSIBLE ARTIFACT"], label=f"POSSIBLE ARTIFACT (n={counts['POSSIBLE ARTIFACT']})"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              ncol=3, frameon=False, fontsize=9)
    plt.tight_layout()
    out = FIG / "sign_reversal_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Task 2 — reversal_sensitivity_analysis.png
# ---------------------------------------------------------------------------
def fig_reversal_sensitivity_analysis() -> Path:
    # For every feature, count number of preprocessing variants with reversal.
    variant_cols = [
        "reversal_raw_space",
        "reversal_log1p",
        "reversal_global_zscore",
        "reversal_per_dataset_zscore",
        "reversal_per_dataset_robust",
        "reversal_global_quantile_normal",
        "reversal_per_dataset_quantile_normal",
    ]
    df = prep.set_index("feature")[variant_cols].astype(bool)
    df["n_variants_reversing"] = df.sum(axis=1)
    df["raw_space"] = df["reversal_raw_space"].astype(int)
    df["transformed_only"] = (
        (~df["reversal_raw_space"]) &
        df[["reversal_log1p", "reversal_global_zscore", "reversal_per_dataset_zscore",
             "reversal_per_dataset_robust", "reversal_global_quantile_normal",
             "reversal_per_dataset_quantile_normal"]].any(axis=1)
    ).astype(int)
    df = df.merge(verdict[["feature", "final_category"]], left_index=True, right_on="feature")

    # Aggregate counts.
    n_total = len(df)
    n_raw_reverse = int(df["reversal_raw_space"].sum())
    n_scaled_reverse = int(df[variant_cols[1:]].any(axis=1).sum())
    n_only_after_scaling = int(df["transformed_only"].sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 2]})

    # Panel A: counts bar.
    ax = axes[0]
    labels = [
        "Raw-space\nstrict reversal",
        "Any transformed/\nscaled variant",
        "Reversal ONLY\nafter scaling",
    ]
    vals = [n_raw_reverse, n_scaled_reverse, n_only_after_scaling]
    colors = ["#1b7837", "#4575b4", "#d73027"]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vals) + 3)
    ax.set_ylabel("Number of features (of 21)")
    ax.set_title("Reversal count by definition")

    # Panel B: per-feature heatmap of variants.
    order_key = {"VERIFIED REVERSAL": 0, "LIKELY REVERSAL BUT SENSITIVE": 1,
                 "POSSIBLE ARTIFACT": 2}
    df_plot = df.assign(_k=df["final_category"].map(order_key)).sort_values(["_k", "feature"])
    mat = df_plot[variant_cols].astype(int).values
    ax = axes[1]
    sns.heatmap(
        mat, cmap=["#f7f7f7", "#4575b4"], vmin=0, vmax=1, cbar=False,
        linewidths=0.5, linecolor="white", ax=ax,
        xticklabels=[c.replace("reversal_", "") for c in variant_cols],
        yticklabels=df_plot["feature"].tolist(),
    )
    for tick, cat in zip(ax.get_yticklabels(), df_plot["final_category"].tolist()):
        tick.set_color(CATEGORY_COLOR.get(cat, "#000000"))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_title("Per-feature reversal under raw vs transformed spaces\n(rows coloured by final forensic category)")

    fig.suptitle(
        f"Sign-reversal sensitivity to preprocessing  "
        f"(raw-strict={n_raw_reverse}, any-transformed={n_scaled_reverse}, only-after-scaling={n_only_after_scaling})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG / "reversal_sensitivity_analysis.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Task 3 — protocol_reversal_counts.png (bar chart by final_category)
# ---------------------------------------------------------------------------
def fig_protocol_reversal_counts() -> Path:
    labels = [c for c in CATEGORY_ORDER]
    vals = [counts[c] for c in labels]
    colors = [CATEGORY_COLOR[c] for c in labels]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, str(v),
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of features (of 21)")
    ax.set_title(
        "Forensic final-category counts for the `safe_core_plus_temporal` family\n"
        "(raw-space strict reversal; supersedes the older 16/21 loose/transformed-space count)"
    )
    ax.set_ylim(0, max(vals) + 3)
    ax.text(
        0.99, 0.97,
        f"Strict raw-space: VERIFIED={counts['VERIFIED REVERSAL']} + LIKELY={counts['LIKELY REVERSAL BUT SENSITIVE']}\n"
        f"Total = {sum(counts.values())} features",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="white", edgecolor="#888"),
    )
    ax.set_xticklabels(labels, rotation=15, ha="right")
    plt.tight_layout()
    out = FIG / "protocol_reversal_counts.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Task 4 — dataset_pair_reversal_contributions.png
# ---------------------------------------------------------------------------
def fig_dataset_pair_reversal_contributions() -> Path:
    """Count, per ordered dataset pair, how many features have opposite signs.

    Two panels:
      A. Strict raw-space: VERIFIED + LIKELY features only.
      B. Loose/transformed-space: any feature reversing under any preprocessing.
    """
    pairs = [("iscx", "usbvpn"), ("iscx", "vnat"), ("usbvpn", "vnat")]

    def sign(feat: str, ds: str) -> int:
        sub = effects_raw[(effects_raw["feature"] == feat) & (effects_raw["dataset"] == ds)]
        if sub.empty:
            return 0
        return int(np.sign(sub["diff_mean"].iloc[0]))

    strict_feats = verdict[verdict["final_category"].isin(
        ["VERIFIED REVERSAL", "LIKELY REVERSAL BUT SENSITIVE"]
    )]["feature"].tolist()
    loose_feats = prep[
        prep[["reversal_log1p", "reversal_global_zscore", "reversal_per_dataset_zscore",
              "reversal_per_dataset_robust", "reversal_global_quantile_normal",
              "reversal_per_dataset_quantile_normal", "reversal_raw_space"]].any(axis=1)
    ]["feature"].tolist()

    def pair_counts(feats: List[str]) -> Dict[tuple, int]:
        res = {}
        for a, b in pairs:
            n = 0
            for f in feats:
                sa, sb = sign(f, a), sign(f, b)
                if sa != 0 and sb != 0 and sa != sb:
                    n += 1
            res[(a, b)] = n
        return res

    strict_counts = pair_counts(strict_feats)
    loose_counts = pair_counts(loose_feats)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = [f"{a.upper()} vs\n{b.upper()}" for a, b in pairs]

    axes[0].bar(x, [strict_counts[p] for p in pairs], color=CATEGORY_COLOR["VERIFIED REVERSAL"])
    for i, v in enumerate([strict_counts[p] for p in pairs]):
        axes[0].text(i, v + 0.05, str(v), ha="center", va="bottom", fontweight="bold")
    axes[0].set_title(
        f"(A) Strict raw-space VERIFIED + LIKELY\n(n_features = {len(strict_feats)})"
    )
    axes[0].set_ylabel("Features reversing between dataset pair")
    max_a = max(strict_counts.values()) if strict_counts else 0
    axes[0].set_ylim(0, max(max_a, 1) + 2)

    axes[1].bar(x, [loose_counts[p] for p in pairs], color="#4575b4")
    for i, v in enumerate([loose_counts[p] for p in pairs]):
        axes[1].text(i, v + 0.05, str(v), ha="center", va="bottom", fontweight="bold")
    axes[1].set_title(
        f"(B) Loose / transformed-space candidates\n(n_features = {len(loose_feats)})"
    )
    axes[1].set_ylabel("Features reversing between dataset pair (loose)")
    max_b = max(loose_counts.values()) if loose_counts else 0
    axes[1].set_ylim(0, max(max_b, 1) + 2)

    fig.suptitle(
        "Dataset-pair reversal contributions (strict vs loose definitions are shown separately)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG / "dataset_pair_reversal_contributions.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Task 5 — top_feature_reversal_burden.png
# ---------------------------------------------------------------------------
def fig_top_feature_reversal_burden() -> Path:
    """Top-K burden analysis under three definitions.

    Rank features by mean |effect size| across datasets (absolute reversal strength
    proxy: mean of |cohen_d| across datasets from feature_effect_strength_report).
    For each top-K set, compute how many fall in each category definition.
    """
    # Rank by mean |cohen_d| across datasets (stable magnitude proxy).
    cohen = strength[strength["metric"] == "cohen_d"].copy()
    rank = (cohen.assign(abs_d=cohen["estimate"].abs())
                 .groupby("feature", as_index=False)["abs_d"].mean()
                 .sort_values("abs_d", ascending=False))
    top5 = rank.head(5)["feature"].tolist()
    top10 = rank.head(10)["feature"].tolist()

    verified = set(verdict.loc[verdict["final_category"] == "VERIFIED REVERSAL", "feature"])
    likely = set(verdict.loc[verdict["final_category"] == "LIKELY REVERSAL BUT SENSITIVE", "feature"])
    artifact = set(verdict.loc[verdict["final_category"] == "POSSIBLE ARTIFACT", "feature"])
    all_candidates = verified | likely | artifact

    rows = []
    for k, feats in [("Top-5", top5), ("Top-10", top10)]:
        rows.append({
            "scope": k,
            "VERIFIED only": sum(1 for f in feats if f in verified),
            "VERIFIED + LIKELY": sum(1 for f in feats if f in (verified | likely)),
            "ALL (incl. POSSIBLE ARTIFACT)": sum(1 for f in feats if f in all_candidates),
            "features": feats,
        })
    bdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    width = 0.25
    xs = np.arange(len(bdf))
    defs = ["VERIFIED only", "VERIFIED + LIKELY", "ALL (incl. POSSIBLE ARTIFACT)"]
    colors = ["#1b7837", "#fdae61", "#d73027"]
    for i, d in enumerate(defs):
        offs = (i - 1) * width
        vals = bdf[d].tolist()
        bars = ax.bar(xs + offs, vals, width, label=d, color=colors[i], edgecolor="black", linewidth=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, str(v),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(bdf["scope"].tolist())
    ax.set_ylabel("Features flagged as reversing (count)")
    ax.set_title(
        "Top-feature reversal burden under three forensic definitions\n"
        "(features ranked by mean |Cohen's d| across datasets)"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, max(bdf[defs].values.max(), 1) + 2)

    # Caption with top-5 list annotated by category.
    def _tag(f: str) -> str:
        if f in verified: return "V"
        if f in likely:   return "L"
        if f in artifact: return "A"
        return "?"
    caption = "Top-5: " + ", ".join(f"{f}[{_tag(f)}]" for f in top5)
    fig.text(0.01, 0.01, caption + "    (V=VERIFIED, L=LIKELY, A=POSSIBLE ARTIFACT)",
             fontsize=8, ha="left")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = FIG / "top_feature_reversal_burden.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    # Also dump the burden table as CSV for the thesis.
    bdf.drop(columns=["features"]).to_csv(TAB / "top_feature_reversal_burden.csv", index=False)
    (TAB / "top_feature_reversal_burden_features.json").write_text(
        json.dumps({"top5": top5, "top10": top10,
                    "top5_categories": {f: _tag(f) for f in top5},
                    "top10_categories": {f: _tag(f) for f in top10}}, indent=2),
        encoding="utf-8",
    )
    return out


# ---------------------------------------------------------------------------
# Task 6 — CSV + LaTeX tables
# ---------------------------------------------------------------------------
def regenerate_tables() -> Dict[str, Path]:
    produced: Dict[str, Path] = {}

    # Summary CSV
    n_total = len(verdict)
    summary = pd.DataFrame([{
        "feature_family": "safe_core_plus_temporal",
        "n_total_features": n_total,
        "n_verified_reversal": counts["VERIFIED REVERSAL"],
        "n_likely_reversal_but_sensitive": counts["LIKELY REVERSAL BUT SENSITIVE"],
        "n_possible_artifact": counts["POSSIBLE ARTIFACT"],
        "n_no_real_reversal": counts["NO REAL REVERSAL"],
        "n_inconclusive": counts["INCONCLUSIVE"],
        "n_strict_consensus_reversal_raw_space": int((verdict["reversal_raw_space"]).sum()),
        "n_reversal_only_after_scaling": int(verdict["reversal_introduced_only_after_scaling"].sum()),
        "thesis_level_claim_verdict": verdict["thesis_level_claim_verdict"].iloc[0],
    }])
    p = TAB / "sign_reversal_summary.csv"
    summary.to_csv(p, index=False)
    produced["tables/sign_reversal_summary.csv"] = p

    # Feature verdicts CSV (thesis-ready columns)
    thesis_cols = [
        "feature",
        "strict_reversal_metric_count",
        "loose_reversal_metric_count",
        "reversal_raw_space",
        "reversal_introduced_only_after_scaling",
        "reversal_stable_across_preprocessing_variants",
        "robustness_tag",
        "final_category",
        "thesis_level_claim_verdict",
    ]
    thesis_df = verdict[thesis_cols].copy()
    # Order by category then by strict count desc.
    order_key = {"VERIFIED REVERSAL": 0, "LIKELY REVERSAL BUT SENSITIVE": 1,
                 "POSSIBLE ARTIFACT": 2, "NO REAL REVERSAL": 3, "INCONCLUSIVE": 4}
    thesis_df = (thesis_df.assign(_k=thesis_df["final_category"].map(order_key))
                           .sort_values(["_k", "strict_reversal_metric_count", "feature"],
                                        ascending=[True, False, True])
                           .drop(columns="_k"))
    p = TAB / "sign_reversal_feature_verdicts.csv"
    thesis_df.to_csv(p, index=False)
    produced["tables/sign_reversal_feature_verdicts.csv"] = p

    # Preprocessing comparison (refreshed copy with final_category annotation).
    prep_annot = prep.merge(verdict[["feature", "final_category"]], on="feature", how="left")
    p = TAB / "reversal_preprocessing_comparison.csv"
    prep_annot.to_csv(p, index=False)
    produced["tables/reversal_preprocessing_comparison.csv"] = p

    # Reversal vs domain fingerprint (refreshed copy with final_category annotation).
    dom_annot = domain.merge(verdict[["feature", "final_category"]], on="feature", how="left")
    p = TAB / "reversal_vs_domain_fingerprint.csv"
    dom_annot.to_csv(p, index=False)
    produced["tables/reversal_vs_domain_fingerprint.csv"] = p

    # LaTeX (booktabs) thesis table
    def _tex_escape(s: str) -> str:
        return str(s).replace("_", r"\_")

    cat_short = {
        "VERIFIED REVERSAL": r"\textbf{VERIFIED}",
        "LIKELY REVERSAL BUT SENSITIVE": r"LIKELY",
        "POSSIBLE ARTIFACT": r"ARTIFACT",
        "NO REAL REVERSAL": r"NONE",
        "INCONCLUSIVE": r"INCONC.",
    }
    rows_tex = []
    for _, r in thesis_df.iterrows():
        rows_tex.append(
            " & ".join([
                _tex_escape(r["feature"]),
                str(int(r["strict_reversal_metric_count"])),
                str(int(r["loose_reversal_metric_count"])),
                "\\checkmark" if bool(r["reversal_raw_space"]) else "--",
                "\\checkmark" if bool(r["reversal_introduced_only_after_scaling"]) else "--",
                "\\checkmark" if bool(r["reversal_stable_across_preprocessing_variants"]) else "--",
                _tex_escape(r["robustness_tag"]),
                cat_short.get(r["final_category"], _tex_escape(r["final_category"])),
            ]) + " \\\\"
        )
    tex = (
        "% Auto-generated by scripts/regenerate_sign_reversal_thesis_artifacts.py\n"
        "% Source: artifacts/sign_reversal_forensic_audit/sign_reversal_final_verdict.csv\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Per-feature forensic verdict for the \\texttt{safe\\_core\\_plus\\_temporal} family. "
        "Strict / loose metric counts are out of 8 independent direction metrics; a feature is in "
        "\\textbf{VERIFIED} only if it reverses in raw space and survives bootstrap + capture + "
        "preprocessing checks.}\n"
        "\\label{tab:sign-reversal-verdicts}\n"
        "\\begin{tabular}{lrrcccll}\n"
        "\\toprule\n"
        "Feature & Strict & Loose & Raw-space & Only after scaling & "
        "Stable prep. & Robustness & Final category \\\\\n"
        "\\midrule\n"
        + "\n".join(rows_tex) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    p = TAB / "sign_reversal_feature_verdicts.tex"
    p.write_text(tex, encoding="utf-8")
    produced["tables/sign_reversal_feature_verdicts.tex"] = p

    return produced


# ---------------------------------------------------------------------------
# Task 9 — consistency checks
# ---------------------------------------------------------------------------
def consistency_checks() -> Dict[str, object]:
    checks: Dict[str, object] = {}
    # Label mapping
    lm = AUDIT / "label_mapping_report.md"
    if lm.exists():
        txt = lm.read_text(encoding="utf-8")
        checks["label_mapping_ok_all_datasets"] = (txt.count("mapping ok: True") >= 3)
    else:
        checks["label_mapping_ok_all_datasets"] = None
        warnings.append("label_mapping_report.md missing")

    # Recomputation match
    rec_path = AUDIT / "tables" / "recomputed_vs_existing_clean_artifact.csv"
    if rec_path.exists():
        rec = pd.read_csv(rec_path)
        checks["raw_recomputation_exact_match_rate_min"] = (
            float(rec["exact_match_rate"].min()) if "exact_match_rate" in rec.columns else None
        )
    else:
        checks["raw_recomputation_exact_match_rate_min"] = None
        warnings.append("recomputed_vs_existing_clean_artifact.csv missing")

    # Direction-safe construction
    checks["direction_safe_all_21"] = bool((verdict["direction_safe"] == "yes").all())
    # Identical computation
    checks["computed_identically_across_datasets"] = bool(
        (verdict["computed_identically_across_datasets"] == "yes").all()
    )
    # Preprocessing source identified (clean pipeline is raw per audit report).
    rep = AUDIT / "sign_reversal_audit_report.md"
    checks["preprocessing_source_identified"] = (
        rep.exists() and "apply_quantile_scaling" in rep.read_text(encoding="utf-8")
    )
    return checks


# ---------------------------------------------------------------------------
# Orchestrate
# ---------------------------------------------------------------------------
def main() -> None:
    generated: Dict[str, str] = {}

    p = fig_sign_reversal_heatmap()
    generated["figures/sign_reversal_heatmap.png"] = str(p.relative_to(ROOT))

    p = fig_reversal_sensitivity_analysis()
    generated["figures/reversal_sensitivity_analysis.png"] = str(p.relative_to(ROOT))

    p = fig_protocol_reversal_counts()
    generated["figures/protocol_reversal_counts.png"] = str(p.relative_to(ROOT))

    p = fig_dataset_pair_reversal_contributions()
    generated["figures/dataset_pair_reversal_contributions.png"] = str(p.relative_to(ROOT))

    p = fig_top_feature_reversal_burden()
    generated["figures/top_feature_reversal_burden.png"] = str(p.relative_to(ROOT))

    produced = regenerate_tables()
    for rel, pth in produced.items():
        generated[rel] = str(pth.relative_to(ROOT))

    checks = consistency_checks()

    # Thesis chapter number block.
    thesis_claim_text = (
        f"{counts['VERIFIED REVERSAL'] + counts['LIKELY REVERSAL BUT SENSITIVE']} of "
        f"{len(verdict)} features show strict consensus reversal under the completed "
        f"raw-space forensic audit, of which {counts['VERIFIED REVERSAL']} are verified and "
        f"{counts['LIKELY REVERSAL BUT SENSITIVE']} is/are likely but sensitive. "
        f"The remaining {counts['POSSIBLE ARTIFACT']} features show possible artifact behaviour, "
        f"mainly because reversal appears only after transformed-space or scaled variants."
    )

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "feature_family": "safe_core_plus_temporal",
        "max_packets": 300,
        "min_packets": 3,
        "source_csvs_used": [
            "artifacts/sign_reversal_forensic_audit/sign_reversal_final_verdict.csv",
            "artifacts/sign_reversal_forensic_audit/feature_effects_by_dataset.csv",
            "artifacts/sign_reversal_forensic_audit/reversal_preprocessing_comparison.csv",
            "artifacts/sign_reversal_forensic_audit/reversal_vs_domain_fingerprint.csv",
            "artifacts/sign_reversal_forensic_audit/feature_effect_strength_report.csv",
            "artifacts/sign_reversal_forensic_audit/feature_reversal_summary.csv",
            "artifacts/sign_reversal_forensic_audit/feature_sign_matrix_mean.csv",
            "artifacts/sign_reversal_forensic_audit/strict_vs_loose_reversal_report.csv",
            "artifacts/sign_reversal_forensic_audit/label_mapping_report.md",
            "artifacts/sign_reversal_forensic_audit/tables/recomputed_vs_existing_clean_artifact.csv",
        ],
        "category_counts": {
            "VERIFIED REVERSAL": counts["VERIFIED REVERSAL"],
            "LIKELY REVERSAL BUT SENSITIVE": counts["LIKELY REVERSAL BUT SENSITIVE"],
            "POSSIBLE ARTIFACT": counts["POSSIBLE ARTIFACT"],
            "NO REAL REVERSAL": counts["NO REAL REVERSAL"],
            "INCONCLUSIVE": counts["INCONCLUSIVE"],
        },
        "thesis_level_claim_verdict": verdict["thesis_level_claim_verdict"].iloc[0],
        "thesis_claim_text_replacement": thesis_claim_text,
        "old_claim_to_replace": "16 of 21 features show triple-consensus reversal",
        "consistency_checks": checks,
        "generated_artifacts": generated,
        "warnings": warnings,
    }
    manifest_path = AUDIT / "regenerated_artifacts_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok",
                      "manifest": str(manifest_path.relative_to(ROOT)),
                      "generated": list(generated.keys()),
                      "warnings": warnings}, indent=2))


if __name__ == "__main__":
    main()

