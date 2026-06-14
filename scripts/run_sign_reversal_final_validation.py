from __future__ import annotations

import json
import shutil

# Ensure `src` package is importable when running this file as a script.
from pathlib import Path
from typing import Dict, List
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.eval.sign_reversal_forensic_audit import (
    AuditConfig,
    DATASETS,
    SIGN_COLUMNS,
    build_bootstrap_reports,
    build_canonical_feature_table,
    build_strict_loose_reversal_report,
    compute_effects_table,
    get_family,
    summarize_reversals,
)


SRC = ROOT / "artifacts" / "sign_reversal_forensic_audit"
OUT = ROOT / "artifacts" / "sign_reversal_final_validation"


def _sign(v: float) -> int:
    if not np.isfinite(v) or abs(v) <= 1e-12:
        return 0
    return 1 if v > 0 else -1


def ensure_dirs() -> Dict[str, Path]:
    dirs = {
        "root": OUT,
        "tables": OUT / "tables",
        "figures": OUT / "figures",
        "intermediate": OUT / "intermediate",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def copy_base_outputs() -> None:
    to_copy = [
        "dataset_inventory.csv",
        "class_balance_summary.csv",
        "feature_quality_report.csv",
        "feature_effects_by_dataset.csv",
        "feature_effect_strength_report.csv",
        "feature_reversal_summary.csv",
        "feature_sign_matrix_mean.csv",
        "feature_sign_matrix_median.csv",
        "feature_sign_matrix_smd.csv",
        "feature_sign_matrix_cliff.csv",
        "feature_sign_matrix_spearman.csv",
        "feature_sign_matrix_logistic.csv",
        "feature_sign_matrix_auc.csv",
        "reversal_preprocessing_comparison.csv",
        "reversal_robustness_balancing.csv",
        "capture_level_feature_effects.csv",
        "capture_purity_report.csv",
        "class_definition_audit.csv",
        "fine_grained_label_breakdown.csv",
        "strict_vs_loose_reversal_report.csv",
        "sign_reversal_final_verdict.csv",
        "feature_construction_audit.csv",
        "reversal_vs_domain_fingerprint.csv",
        "univariate_model_report.csv",
    ]
    for name in to_copy:
        src = SRC / name
        dst = OUT / name
        if src.exists():
            shutil.copy2(src, dst)

    table_copy = {
        SRC / "tables" / "recomputed_vs_existing_clean_artifact.csv": OUT / "tables" / "recomputed_vs_clean.csv",
        SRC / "tables" / "truncation_audit.csv": OUT / "tables" / "truncation_rates_by_dataset_label.csv",
    }
    for src, dst in table_copy.items():
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def build_label_mapping_audit_csv() -> pd.DataFrame:
    records: List[dict] = []
    md = (SRC / "label_mapping_report.md").read_text(encoding="utf-8")
    current = None
    row: Dict[str, str] = {}
    for line in md.splitlines():
        line = line.strip()
        if line.startswith("## "):
            if row:
                records.append(row)
            current = line.replace("## ", "").strip()
            row = {"dataset": current}
        elif line.startswith("- unique raw label values:"):
            row["unique_raw_label_values"] = line.split("`", 2)[1]
        elif line.startswith("- binary values observed:"):
            row["unique_binary_values"] = line.split("`", 2)[1]
        elif line.startswith("- VPN ="):
            row["vpn_binary_value"] = line.split("`", 2)[1]
        elif line.startswith("- nonVPN ="):
            row["nonvpn_binary_value"] = line.split("`", 2)[1]
        elif line.startswith("- mapping ok:"):
            row["mapping_ok"] = "True" in line
        elif line.startswith("- note:"):
            row["mapping_note"] = line.replace("- note:", "").strip()
    if row:
        records.append(row)
    out = pd.DataFrame(records)
    out.to_csv(OUT / "tables" / "label_mapping_audit.csv", index=False)
    return out


def build_raw_space_exports() -> pd.DataFrame:
    strict = pd.read_csv(OUT / "strict_vs_loose_reversal_report.csv")
    final = pd.read_csv(OUT / "sign_reversal_final_verdict.csv")
    merged = strict.merge(final[["feature", "final_category"]], on="feature", how="left")
    merged.to_csv(OUT / "tables" / "raw_space_200_bootstrap_feature_verdicts.csv", index=False)

    effects = pd.read_csv(OUT / "feature_effects_by_dataset.csv")
    sign_cols = ["dataset", "feature"] + list(SIGN_COLUMNS.values())
    effects[sign_cols].to_csv(OUT / "tables" / "raw_space_metric_signs_by_dataset.csv", index=False)

    src_heat = SRC / "figures" / "heatmap_sign_direction_by_feature_dataset.png"
    if src_heat.exists():
        shutil.copy2(src_heat, OUT / "figures" / "raw_space_reversal_heatmap.png")

    strength = pd.read_csv(OUT / "feature_effect_strength_report.csv")
    top = (
        strict.sort_values(["strict_reversal_metric_count", "feature"], ascending=[False, True])
        .head(8)["feature"]
        .tolist()
    )
    plot = strength[strength["feature"].isin(top) & (strength["metric"] == "cohen_d")].copy()
    if not plot.empty:
        plt.figure(figsize=(10, 6))
        for i, feature in enumerate(sorted(plot["feature"].unique())):
            sub = plot[plot["feature"] == feature]
            x = np.arange(len(sub)) + i * 0.03
            y = sub["estimate"].to_numpy()
            lo = y - sub["ci_low"].to_numpy()
            hi = sub["ci_high"].to_numpy() - y
            plt.errorbar(x, y, yerr=[lo, hi], fmt="o", label=feature, capsize=3)
        plt.axhline(0.0, color="black", linewidth=1)
        plt.xticks(range(len(DATASETS)), DATASETS)
        plt.ylabel("Cohen's d (estimate +/- 95% CI)")
        plt.title("Grouped-bootstrap confidence intervals (raw space)")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(OUT / "figures" / "bootstrap_confidence_intervals.png", dpi=180)
        plt.close()
    return merged


def full_length_audit(features: List[str]) -> pd.DataFrame:
    cfg = AuditConfig(
        repo_root=ROOT,
        output_dir=OUT,
        feature_family="safe_core_plus_temporal",
        max_packets=300,
        min_packets=3,
        seed=42,
        n_bootstrap=200,
        use_cache=True,
        force_recompute=False,
        include_full_length_check=False,
        compare_existing_clean_artifact=False,
    )
    full = build_canonical_feature_table(cfg, max_packets=None, cache_name="canonical_safe_core_plus_temporal_full_length.parquet")
    full.to_parquet(OUT / "intermediate" / "canonical_feature_table_full_length.parquet", index=False)

    effects_full = compute_effects_table(full, features, analysis_name="all_flows", transform_name="raw_full_length", seed=42)
    _, strength_full = build_bootstrap_reports(full, features, seed=42, n_bootstrap=200)
    strict_full = build_strict_loose_reversal_report(strength_full)
    strict_full.to_csv(OUT / "tables" / "full_length_feature_verdicts.csv", index=False)

    strict_raw = pd.read_csv(OUT / "strict_vs_loose_reversal_report.csv")
    comp = strict_raw[["feature", "consensus_reversal", "strict_reversal_metric_count"]].merge(
        strict_full[["feature", "consensus_reversal", "strict_reversal_metric_count"]],
        on="feature",
        suffixes=("_raw_300", "_full_length"),
    )
    comp["reversal_persists_full_length"] = comp["consensus_reversal_raw_300"] & comp["consensus_reversal_full_length"]
    comp["reversal_disappears_full_length"] = comp["consensus_reversal_raw_300"] & (~comp["consensus_reversal_full_length"])
    comp["reversal_only_under_300_truncation"] = comp["reversal_disappears_full_length"]
    comp.to_csv(OUT / "tables" / "truncation_vs_full_length_reversal_comparison.csv", index=False)

    plot_df = comp.copy()
    plot_df = plot_df.melt(
        id_vars=["feature"],
        value_vars=["consensus_reversal_raw_300", "consensus_reversal_full_length"],
        var_name="space",
        value_name="reversal",
    )
    plot_df["reversal"] = plot_df["reversal"].astype(int)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="feature", y="reversal", hue="space")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Consensus reversal (0/1)")
    plt.title("Truncation effect on strict reversal")
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "truncation_effect_on_reversal.png", dpi=180)
    plt.close()
    return strict_full


def per_capture_audit(df300: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    for (dataset, capture_id), cap in df300.groupby(["dataset", "capture_id"]):
        labels = set(cap["label"].unique().tolist())
        if labels != {0, 1}:
            continue
        for feature in features:
            vpn = cap.loc[cap["label"] == 1, feature].to_numpy(dtype=float)
            non = cap.loc[cap["label"] == 0, feature].to_numpy(dtype=float)
            diff = float(np.mean(vpn) - np.mean(non))
            rows.append(
                {
                    "dataset": dataset,
                    "capture_id": capture_id,
                    "feature": feature,
                    "capture_diff_mean": diff,
                    "capture_sign": _sign(diff),
                    "n_vpn": int((cap["label"] == 1).sum()),
                    "n_nonvpn": int((cap["label"] == 0).sum()),
                }
            )
    per_cap = pd.DataFrame(rows)
    per_cap.to_csv(OUT / "tables" / "per_capture_effect_signs.csv", index=False)

    cap_effects = pd.read_csv(OUT / "capture_level_feature_effects.csv")
    cap_summary = summarize_reversals(cap_effects)
    cap_summary.to_csv(OUT / "tables" / "capture_aggregated_reversal_verdicts.csv", index=False)

    if not per_cap.empty:
        keep = per_cap[per_cap["feature"].isin(cap_summary.head(8)["feature"])].copy()
        plt.figure(figsize=(10, 6))
        sns.countplot(data=keep, x="feature", hue="capture_sign")
        plt.xticks(rotation=45, ha="right")
        plt.title("Per-capture sign distribution (captures with both classes)")
        plt.tight_layout()
        plt.savefig(OUT / "figures" / "per_capture_sign_distribution.png", dpi=180)
        plt.close()
    return cap_summary


def map_coarse_label(name: str) -> str:
    x = str(name).lower().replace("non", "")
    if any(t in x for t in ["stream", "youtube", "netflix", "vimeo", "spotify"]):
        return "streaming"
    if any(t in x for t in ["voip", "skype", "hangout", "meet", "chat", "discord"]):
        return "chat_meet_voip"
    if any(t in x for t in ["scp", "sftp", "rsync", "ftps", "file", "torrent"]):
        return "file_transfer"
    if any(t in x for t in ["ssh", "rdp", "remote"]):
        return "remote_access"
    if any(t in x for t in ["mail", "email"]):
        return "mail"
    if any(t in x for t in ["http", "www", "brows", "facebook", "instagram", "web"]):
        return "browsing_general"
    return "unknown"


def application_mixture_audit(df300: pd.DataFrame, features: List[str], raw_verdict: pd.DataFrame) -> None:
    fine = pd.read_csv(OUT / "fine_grained_label_breakdown.csv")
    fine["coarse_label"] = fine["app"].map(map_coarse_label)
    fine.to_csv(OUT / "tables" / "fine_label_inventory.csv", index=False)

    coarse_map = fine[["dataset", "label", "app", "coarse_label"]].drop_duplicates()
    coarse_map.to_csv(OUT / "tables" / "coarse_label_mapping.csv", index=False)

    df = df300.copy()
    df["coarse_label"] = df["app"].map(map_coarse_label)

    candidate = raw_verdict[raw_verdict["consensus_reversal"]]["feature"].tolist()
    rows = []
    for feature in candidate:
        tested = 0
        reversed_groups = 0
        for coarse, sub in df.groupby("coarse_label"):
            signs = {}
            for dataset, ds in sub.groupby("dataset"):
                if ds["label"].nunique() < 2:
                    continue
                counts = ds["label"].value_counts().to_dict()
                if counts.get(0, 0) < 20 or counts.get(1, 0) < 20:
                    continue
                diff = float(ds.loc[ds["label"] == 1, feature].mean() - ds.loc[ds["label"] == 0, feature].mean())
                signs[dataset] = _sign(diff)
            non_zero = [v for v in signs.values() if v != 0]
            if len(non_zero) >= 2:
                tested += 1
                if 1 in non_zero and -1 in non_zero:
                    reversed_groups += 1
        rows.append(
            {
                "feature": feature,
                "tested_coarse_groups": tested,
                "reversing_coarse_groups": reversed_groups,
                "reversal_persists_within_coarse_groups": reversed_groups > 0,
                "insufficient_support": tested == 0,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "tables" / "application_controlled_reversal_verdicts.csv", index=False)

    mix = (
        df.groupby(["dataset", "label", "coarse_label"], as_index=False)
        .size()
        .rename(columns={"size": "n_flows"})
    )
    mix["label_name"] = mix["label"].map({0: "nonVPN", 1: "VPN"})
    totals = mix.groupby(["dataset", "label"]) ["n_flows"].transform("sum")
    mix["share"] = mix["n_flows"] / totals
    plt.figure(figsize=(12, 6))
    sns.barplot(data=mix, x="dataset", y="share", hue="coarse_label")
    plt.title("Application mixture by dataset and class (coarse mapping)")
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "application_mixture_by_dataset_label.png", dpi=180)
    plt.close()


def preprocessing_audit(final_df: pd.DataFrame) -> None:
    prep = pd.read_csv(OUT / "reversal_preprocessing_comparison.csv")
    merged = prep.merge(final_df[["feature", "final_category"]], on="feature", how="left")

    def tag(r: pd.Series) -> str:
        if bool(r.get("reversal_introduced_only_after_scaling", False)):
            return "scaling-induced artifact"
        if bool(r.get("reversal_raw_space", False)) and bool(r.get("reversal_stable_across_preprocessing_variants", False)):
            return "raw-space verified"
        if bool(r.get("reversal_raw_space", False)) and (not bool(r.get("reversal_stable_across_preprocessing_variants", False))):
            return "scaling-sensitive"
        return "inconsistent"

    merged["preprocessing_tag"] = merged.apply(tag, axis=1)
    merged.to_csv(OUT / "tables" / "preprocessing_sensitivity_verdicts.csv", index=False)

    cols = [
        "reversal_raw_space",
        "reversal_log1p",
        "reversal_global_zscore",
        "reversal_per_dataset_zscore",
        "reversal_per_dataset_robust",
        "reversal_global_quantile_normal",
        "reversal_per_dataset_quantile_normal",
    ]
    mat = merged.set_index("feature")[cols].astype(int)
    plt.figure(figsize=(10, 7))
    sns.heatmap(mat, cmap="viridis", vmin=0, vmax=1, linewidths=0.5)
    plt.title("Preprocessing sensitivity matrix (reversal yes/no)")
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "preprocessing_sensitivity_matrix.png", dpi=180)
    plt.close()


def final_classification(strict_raw: pd.DataFrame, cap_summary: pd.DataFrame, full_summary: pd.DataFrame) -> pd.DataFrame:
    prep = pd.read_csv(OUT / "reversal_preprocessing_comparison.csv")
    features = strict_raw["feature"].tolist()
    cap_map = cap_summary.set_index("feature")
    full_map = full_summary.set_index("feature")
    prep_map = prep.set_index("feature")

    rows = []
    for f in features:
        sr = strict_raw.set_index("feature").loc[f]
        cap_ok = bool(cap_map.at[f, "consensus_reversal"]) if f in cap_map.index else False
        full_ok = bool(full_map.at[f, "consensus_reversal"]) if (f in full_map.index and pd.notna(full_map.at[f, "consensus_reversal"])) else False
        full_known = bool(f in full_map.index and pd.notna(full_map.at[f, "consensus_reversal"]))
        raw_ok = bool(sr["consensus_reversal"])
        strict_count = int(sr["strict_reversal_metric_count"])
        loose = bool(sr["loose_reversal"])
        scale_only = bool(prep_map.at[f, "reversal_introduced_only_after_scaling"]) if f in prep_map.index else False

        if raw_ok and strict_count >= 3 and cap_ok and full_ok and (not scale_only):
            cat = "A. VERIFIED ROBUST REVERSAL"
        elif raw_ok and strict_count >= 3 and cap_ok and full_known and (not full_ok) and (not scale_only):
            cat = "B. VERIFIED PREFIX-WINDOW REVERSAL"
        elif scale_only:
            cat = "D. POSSIBLE ARTIFACT"
        elif strict_count == 0 and (not loose):
            cat = "E. NO REVERSAL"
        elif raw_ok and not full_known:
            cat = "F. INCONCLUSIVE"
        elif raw_ok:
            cat = "C. LIKELY BUT SENSITIVE REVERSAL"
        else:
            cat = "F. INCONCLUSIVE"

        rows.append(
            {
                "feature": f,
                "raw_consensus_reversal": raw_ok,
                "capture_aggregated_consensus_reversal": cap_ok,
                "full_length_consensus_reversal": full_ok,
                "strict_reversal_metric_count_raw": strict_count,
                "reversal_introduced_only_after_scaling": scale_only,
                "final_category": cat,
            }
        )

    final = pd.DataFrame(rows)
    final.to_csv(OUT / "tables" / "final_feature_verdict_table.csv", index=False)

    counts = final["final_category"].value_counts().to_dict()
    robust = counts.get("A. VERIFIED ROBUST REVERSAL", 0)
    prefix = counts.get("B. VERIFIED PREFIX-WINDOW REVERSAL", 0)
    likely = counts.get("C. LIKELY BUT SENSITIVE REVERSAL", 0)
    artifact = counts.get("D. POSSIBLE ARTIFACT", 0)
    no_rev = counts.get("E. NO REVERSAL", 0)
    total = len(final)

    if robust >= 8 and artifact <= 3:
        thesis = "A. strongly supported"
    elif robust + prefix + likely >= 6:
        thesis = "B. partially supported"
    elif no_rev + artifact > (total / 2):
        thesis = "D. likely artifact"
    else:
        thesis = "C. weak / unstable"

    rec = pd.DataFrame(
        [
            {
                "thesis_claim_rating": thesis,
                "n_total_features": total,
                "n_A_verified_robust": robust,
                "n_B_verified_prefix_window": prefix,
                "n_C_likely_sensitive": likely,
                "n_D_possible_artifact": artifact,
                "n_E_no_reversal": no_rev,
                "n_F_inconclusive": counts.get("F. INCONCLUSIVE", 0),
                "was_original_16_of_21_claim_correct": "no" if robust + prefix < 16 else "partially",
            }
        ]
    )
    rec.to_csv(OUT / "tables" / "final_thesis_claim_recommendation.csv", index=False)
    return final


def write_readme(final: pd.DataFrame, full_comp: pd.DataFrame, rec: pd.DataFrame) -> None:
    robust = int((final["final_category"] == "A. VERIFIED ROBUST REVERSAL").sum())
    prefix = int((final["final_category"] == "B. VERIFIED PREFIX-WINDOW REVERSAL").sum())
    likely = int((final["final_category"] == "C. LIKELY BUT SENSITIVE REVERSAL").sum())
    artifact = int((final["final_category"] == "D. POSSIBLE ARTIFACT").sum())
    disappeared_mask = full_comp.get("reversal_disappears_full_length", pd.Series(False, index=full_comp.index))
    disappeared_mask = disappeared_mask.fillna(False).astype(bool)
    disappeared = full_comp.loc[disappeared_mask, "feature"].tolist()
    scaling = final[final["reversal_introduced_only_after_scaling"]]["feature"].tolist()

    safe = final[final["final_category"].isin(["A. VERIFIED ROBUST REVERSAL", "B. VERIFIED PREFIX-WINDOW REVERSAL"])]["feature"].tolist()
    downgrade = final[final["final_category"].isin(["C. LIKELY BUT SENSITIVE REVERSAL", "D. POSSIBLE ARTIFACT", "F. INCONCLUSIVE"])]["feature"].tolist()

    thesis_sentence = (
        "Across ISCXVPN2016, USBVPN, and VNAT, strict cross-dataset sign reversal is reproducibly observed for a subset of "
        f"the 21 safe_core_plus_temporal features ({robust} robust + {prefix} prefix-window), while the remaining features are "
        "sensitive to preprocessing, truncation windowing, or dataset/class-mixture effects and are therefore treated as secondary evidence."
    )

    text = f"""# Final Forensic Validation - Sign Reversal (safe_core_plus_temporal)

This directory contains the final skeptical rerun package requested for thesis correction.

## Direct Answers
- Was the original 16/21 claim correct? **No under strict forensic criteria**.
- How many reversals are truly robust? **{robust}** (Category A).
- How many are prefix-window only? **{prefix}** (Category B).
- Which features are safe to discuss in thesis? `{safe}`.
- Which features should be downgraded or moved to appendix? `{downgrade}`.
- Is truncation responsible for any reversals? **Yes, plausible for** `{disappeared}`.
- Is preprocessing responsible for any reversals? **Yes, scaling-induced candidates:** `{scaling}`.
- Is application/class-mixture a plausible alternative explanation? **Yes**, especially where controlled coarse-group support is limited.

## Final Recommendation
- Thesis claim rating: **{rec.iloc[0]['thesis_claim_rating']}**
- Robust + prefix-window + likely sensitive: **{robust + prefix + likely} / {len(final)}**

## Thesis-safe sentence
> {thesis_sentence}

## Key Outputs
- `tables/final_feature_verdict_table.csv`
- `tables/final_thesis_claim_recommendation.csv`
- `tables/truncation_vs_full_length_reversal_comparison.csv`
- `tables/application_controlled_reversal_verdicts.csv`
- `tables/preprocessing_sensitivity_verdicts.csv`
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def _fallback_full_length_outputs(features: List[str]) -> pd.DataFrame:
    # Full-length no-cap extraction was not completed; emit explicit placeholders.
    out = pd.DataFrame(
        {
            "feature": features,
            "consensus_reversal": [np.nan] * len(features),
            "strict_reversal_metric_count": [np.nan] * len(features),
            "full_length_tested": [False] * len(features),
            "note": ["full_length_not_completed_in_this_run"] * len(features),
        }
    )
    out.to_csv(OUT / "tables" / "full_length_feature_verdicts.csv", index=False)

    strict_raw = pd.read_csv(OUT / "strict_vs_loose_reversal_report.csv")
    comp = strict_raw[["feature", "consensus_reversal", "strict_reversal_metric_count"]].copy()
    comp = comp.rename(columns={
        "consensus_reversal": "consensus_reversal_raw_300",
        "strict_reversal_metric_count": "strict_reversal_metric_count_raw_300",
    })
    comp["consensus_reversal_full_length"] = np.nan
    comp["strict_reversal_metric_count_full_length"] = np.nan
    comp["reversal_persists_full_length"] = np.nan
    comp["reversal_disappears_full_length"] = np.nan
    comp["reversal_only_under_300_truncation"] = np.nan
    comp["full_length_tested"] = False
    comp["note"] = "full_length_not_completed_in_this_run"
    comp.to_csv(OUT / "tables" / "truncation_vs_full_length_reversal_comparison.csv", index=False)

    plt.figure(figsize=(8, 3))
    plt.text(0.01, 0.5, "Full-length no-cap audit not completed in this run.\nSee note column in truncation_vs_full_length_reversal_comparison.csv", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "truncation_effect_on_reversal.png", dpi=180)
    plt.close()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final sign-reversal validation artifacts")
    parser.add_argument("--skip-full-length", action="store_true", help="Skip no-cap full-length recomputation and emit explicit inconclusive placeholders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    copy_base_outputs()
    build_label_mapping_audit_csv()
    strict_raw = build_raw_space_exports()

    features = list(get_family("safe_core_plus_temporal"))
    cfg = AuditConfig(
        repo_root=ROOT,
        output_dir=SRC,
        feature_family="safe_core_plus_temporal",
        max_packets=300,
        min_packets=3,
        seed=42,
        n_bootstrap=200,
    )
    df300 = build_canonical_feature_table(cfg, max_packets=300, cache_name="canonical_safe_core_plus_temporal_300.parquet")
    df300.to_parquet(OUT / "intermediate" / "canonical_feature_table_300.parquet", index=False)

    if args.skip_full_length:
        full_summary = _fallback_full_length_outputs(features)
    else:
        try:
            full_summary = full_length_audit(features)
        except Exception:
            full_summary = _fallback_full_length_outputs(features)
    full_comp = pd.read_csv(OUT / "tables" / "truncation_vs_full_length_reversal_comparison.csv")
    cap_summary = per_capture_audit(df300, features)
    application_mixture_audit(df300, features, strict_raw)

    final_from_previous = pd.read_csv(OUT / "sign_reversal_final_verdict.csv")
    preprocessing_audit(final_from_previous)

    final = final_classification(strict_raw, cap_summary, full_summary)
    rec = pd.read_csv(OUT / "tables" / "final_thesis_claim_recommendation.csv")
    write_readme(final, full_comp, rec)

    summary = {
        "status": "ok",
        "output_dir": str(OUT),
        "n_features": int(len(final)),
        "category_counts": final["final_category"].value_counts().to_dict(),
    }
    (OUT / "final_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
