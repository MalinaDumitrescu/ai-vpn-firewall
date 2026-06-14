"""
Numerical-edge-case audit for the cross-dataset sign-reversal claim.

For every (dataset, feature) in the canonical 21-feature `safe_core_plus_temporal`
table, this script:

  1. Counts NaN / +inf / -inf / zeros / negatives, and reports the percentile
     spectrum  p0, p0.1, p1, p50, p99, p99.9, p100  per (dataset, feature).
  2. Inspects flow_duration near zero and its propagation into packet_rate /
     byte_rate (the extractor floors duration at 1e-9).
  3. Counts zero-duration flows (raw vs floored).
  4. Re-runs the existing reversal statistic (8 signed metrics, consensus =
     >=3 metrics flagging cross-dataset sign disagreement) under five regimes:
        baseline           - identity
        drop_nan_inf       - drop rows with any NaN / +-inf in the 21 features
        drop_zero_dur      - drop rows whose RAW flow_duration is <= 1e-9
        winsor             - per-(dataset, feature) clip to [q0.001, q0.999]
        log1p_heavy        - log1p the heavy-tailed positive features (skew>2)
  5. Builds a per-feature verdict matrix across the five regimes and tags
     robustness (stable_no_reversal / stable_reversal / cleaning_resolves /
     cleaning_induces / unstable).

Outputs land under
  artifacts/thesis_finalization/nb53_sign_reversal_audit/numerical_edge_audit/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.clean_pipeline.feature_families import get_family
from src.eval.sign_reversal_forensic_audit import (
    DATASETS,
    SIGN_COLUMNS,
    compute_effects_table,
    save_sign_matrices,
    summarize_reversals,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = (
    REPO_ROOT
    / "artifacts"
    / "sign_reversal_forensic_audit"
    / "intermediate"
    / "canonical_safe_core_plus_temporal_300.parquet"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "artifacts"
    / "thesis_finalization"
    / "nb53_sign_reversal_audit"
    / "numerical_edge_audit"
)
SKEW_THRESHOLD = 2.0
DUR_EPS = 1e-9   # extractor floor
WINSOR_LOW = 0.001
WINSOR_HIGH = 0.999

MODES: Tuple[str, ...] = ("baseline", "drop_nan_inf", "drop_zero_dur", "winsor", "log1p_heavy")


# ---------------------------------------------------------------------------
# Inventory + edge-case characterisation
# ---------------------------------------------------------------------------

def compute_value_inventory(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        for feat in features:
            v = sub[feat].to_numpy(dtype=np.float64)
            n = len(v)
            isnan = np.isnan(v)
            isposinf = np.isposinf(v)
            isneginf = np.isneginf(v)
            finite = v[np.isfinite(v)]
            pcts = np.percentile(finite, [0, 0.1, 1, 50, 99, 99.9, 100]) if finite.size else [np.nan] * 7
            skew = float(pd.Series(finite).skew()) if finite.size > 2 else 0.0
            kurt = float(pd.Series(finite).kurt()) if finite.size > 3 else 0.0
            rows.append({
                "dataset": ds,
                "feature": feat,
                "n": int(n),
                "n_nan": int(isnan.sum()),
                "n_pos_inf": int(isposinf.sum()),
                "n_neg_inf": int(isneginf.sum()),
                "n_zero": int(np.sum(finite == 0.0)),
                "n_negative": int(np.sum(finite < 0.0)),
                "p0":    float(pcts[0]),
                "p0_1":  float(pcts[1]),
                "p1":    float(pcts[2]),
                "p50":   float(pcts[3]),
                "p99":   float(pcts[4]),
                "p99_9": float(pcts[5]),
                "p100":  float(pcts[6]),
                "skew": skew,
                "kurtosis": kurt,
            })
    return pd.DataFrame(rows)


def compute_duration_propagation(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 0.0, DUR_EPS, 1e-6, 1e-3, 1.0, 10.0, 60.0, 600.0, np.inf]
    bin_labels = [
        "neg_or_zero", "0_to_1e-9", "1e-9_to_1e-6", "1e-6_to_1e-3",
        "1e-3_to_1", "1_to_10", "10_to_60", "60_to_600", "ge_600",
    ]
    rows: List[Dict] = []
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        dur_raw = sub["raw_flow_duration_full"].to_numpy(dtype=np.float64)
        dur_win = sub["flow_duration"].to_numpy(dtype=np.float64)
        pr = sub["packet_rate"].to_numpy(dtype=np.float64)
        br = sub["byte_rate"].to_numpy(dtype=np.float64)
        cats = pd.cut(dur_win, bins=bins, labels=bin_labels, right=False, include_lowest=True)
        cat_counts = cats.value_counts().reindex(bin_labels, fill_value=0).to_dict()
        n_zero_raw = int(np.sum(dur_raw <= 0.0))
        n_floor = int(np.sum(dur_win <= DUR_EPS))
        floor_mask = dur_win <= DUR_EPS
        rows.append({
            "dataset": ds,
            "n_total": int(len(sub)),
            "n_raw_duration_le_zero": n_zero_raw,
            "n_window_duration_le_eps": n_floor,
            "frac_floor_engaged": float(n_floor / max(len(sub), 1)),
            "packet_rate_p99":  float(np.nanpercentile(pr, 99)),
            "packet_rate_p999": float(np.nanpercentile(pr, 99.9)),
            "packet_rate_p100": float(np.nanmax(pr)) if len(pr) else np.nan,
            "byte_rate_p99":    float(np.nanpercentile(br, 99)),
            "byte_rate_p999":   float(np.nanpercentile(br, 99.9)),
            "byte_rate_p100":   float(np.nanmax(br)) if len(br) else np.nan,
            "packet_rate_p100_among_floor": float(np.nanmax(pr[floor_mask])) if floor_mask.any() else np.nan,
            "byte_rate_p100_among_floor":   float(np.nanmax(br[floor_mask])) if floor_mask.any() else np.nan,
            **{f"dur_bin__{k}": int(v) for k, v in cat_counts.items()},
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cleaning regimes
# ---------------------------------------------------------------------------

def apply_cleaning(
    df: pd.DataFrame,
    mode: str,
    features: Sequence[str],
    heavy_tail_features: Sequence[str],
) -> Tuple[pd.DataFrame, Dict]:
    n_in = len(df)
    info: Dict = {"mode": mode, "n_in": n_in}

    if mode == "baseline":
        out = df.copy()

    elif mode == "drop_nan_inf":
        X = df[list(features)].to_numpy(dtype=np.float64)
        bad = ~np.isfinite(X).all(axis=1)
        out = df.loc[~bad].copy()
        info["n_dropped_bad_rows"] = int(bad.sum())

    elif mode == "drop_zero_dur":
        bad = (df["raw_flow_duration_full"].to_numpy(dtype=np.float64) <= DUR_EPS)
        out = df.loc[~bad].copy()
        info["n_dropped_zero_dur"] = int(bad.sum())

    elif mode == "winsor":
        out = df.copy()
        for ds in DATASETS:
            mask = out["dataset"] == ds
            for feat in features:
                v = out.loc[mask, feat].to_numpy(dtype=np.float64)
                finite = v[np.isfinite(v)]
                if finite.size == 0:
                    continue
                lo = float(np.quantile(finite, WINSOR_LOW))
                hi = float(np.quantile(finite, WINSOR_HIGH))
                out.loc[mask, feat] = np.clip(v, lo, hi)
        info["winsor_pct"] = [WINSOR_LOW, WINSOR_HIGH]

    elif mode == "log1p_heavy":
        out = df.copy()
        for feat in heavy_tail_features:
            v = out[feat].to_numpy(dtype=np.float64)
            v = np.where(np.isfinite(v), v, 0.0)
            out[feat] = np.log1p(np.clip(v, 0.0, None))
        info["log1p_features"] = list(heavy_tail_features)

    else:
        raise ValueError(f"Unknown cleaning mode: {mode}")

    info["n_out"] = len(out)
    info["n_dropped_total"] = n_in - len(out)
    return out, info


# ---------------------------------------------------------------------------
# Verdict matrix + robustness classification
# ---------------------------------------------------------------------------

def _signs_for_feature(effects: pd.DataFrame, feature: str, sign_col: str) -> Dict[str, int]:
    sub = effects[effects["feature"] == feature].set_index("dataset")
    return {ds: int(sub.loc[ds, sign_col]) if ds in sub.index else 0 for ds in DATASETS}


def _pattern_str(signs: Dict[str, int]) -> str:
    sym = {1: "+", -1: "-", 0: "0"}
    return "/".join(sym[signs[ds]] for ds in DATASETS)


def build_verdict_matrix(
    effects_by_mode: Dict[str, pd.DataFrame],
    summaries_by_mode: Dict[str, pd.DataFrame],
    features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: List[Dict] = []
    for mode in MODES:
        eff = effects_by_mode[mode]
        summ = summaries_by_mode[mode].set_index("feature")
        for feat in features:
            smd_signs = _signs_for_feature(eff, feat, "sign_smd")
            auc_signs = _signs_for_feature(eff, feat, "sign_auc")
            long_rows.append({
                "feature": feat,
                "mode": mode,
                "smd_iscx":   smd_signs["iscx"],
                "smd_usbvpn": smd_signs["usbvpn"],
                "smd_vnat":   smd_signs["vnat"],
                "smd_pattern": _pattern_str(smd_signs),
                "auc_pattern": _pattern_str(auc_signs),
                "loose_reversal_metric_count": int(summ.at[feat, "loose_reversal_metric_count"]) if feat in summ.index else 0,
                "consensus_reversal": bool(summ.at[feat, "consensus_reversal"]) if feat in summ.index else False,
                "reversal_any_metric": bool(summ.at[feat, "reversal_any_metric"]) if feat in summ.index else False,
            })
    long_df = pd.DataFrame(long_rows)

    # Wide
    wide = long_df.pivot(index="feature", columns="mode", values="smd_pattern")
    wide = wide.reindex(columns=list(MODES))
    cons = long_df.pivot(index="feature", columns="mode", values="consensus_reversal").reindex(columns=list(MODES))
    loose_cnt = long_df.pivot(index="feature", columns="mode", values="loose_reversal_metric_count").reindex(columns=list(MODES))
    wide.columns = [f"smd_pattern__{c}" for c in wide.columns]
    cons.columns = [f"consensus__{c}" for c in cons.columns]
    loose_cnt.columns = [f"loose_metric_count__{c}" for c in loose_cnt.columns]

    wide_out = pd.concat([wide, cons, loose_cnt], axis=1).reset_index()

    # Robustness flag
    flags: List[str] = []
    n_distinct: List[int] = []
    for feat in wide_out["feature"]:
        patterns = [wide_out.loc[wide_out["feature"] == feat, f"smd_pattern__{m}"].iloc[0] for m in MODES]
        cons_vals = [bool(wide_out.loc[wide_out["feature"] == feat, f"consensus__{m}"].iloc[0]) for m in MODES]
        n_distinct.append(len(set(patterns)))
        baseline_cons = cons_vals[0]
        all_same_pattern = len(set(patterns)) == 1
        all_cons = all(cons_vals)
        any_cons = any(cons_vals)
        if all_same_pattern and not any_cons:
            flags.append("stable_no_reversal")
        elif all_same_pattern and all_cons:
            flags.append("stable_reversal")
        elif baseline_cons and not all_cons:
            flags.append("cleaning_resolves_reversal")
        elif (not baseline_cons) and any_cons:
            flags.append("cleaning_induces_reversal")
        else:
            flags.append("unstable")
    wide_out["n_distinct_smd_patterns"] = n_distinct
    wide_out["robustness_flag"] = flags
    return long_df, wide_out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_sign_heatmap(long_df: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sym_to_num = {"+": 1, "-": -1, "0": 0}
    rows = []
    for _, r in long_df.iterrows():
        for i, ds in enumerate(DATASETS):
            rows.append({
                "feature": r["feature"],
                "label": f"{r['mode']}::{ds}",
                "sign": sym_to_num[r["smd_pattern"].split("/")[i]],
            })
    plot_df = pd.DataFrame(rows).pivot(index="feature", columns="label", values="sign")
    label_order = [f"{m}::{ds}" for m in MODES for ds in DATASETS]
    plot_df = plot_df.reindex(columns=label_order)
    sns.set_theme(style="white")
    plt.figure(figsize=(12, max(6, len(plot_df) * 0.32)))
    sns.heatmap(plot_df, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                cbar_kws={"label": "SMD sign"}, linewidths=0.4)
    plt.title("SMD sign per feature × (cleaning mode, dataset)")
    plt.tight_layout()
    plt.savefig(fig_dir / "sign_pattern_heatmap.png", dpi=170)
    plt.close()


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(
    out_dir: Path,
    inventory: pd.DataFrame,
    dur_prop: pd.DataFrame,
    cleaning_log: pd.DataFrame,
    heavy_tail_features: List[str],
    skew_table: pd.DataFrame,
    wide: pd.DataFrame,
) -> None:
    counts = wide["robustness_flag"].value_counts().to_dict()

    lines: List[str] = []
    lines.append("# Numerical-Edge-Case Audit of Cross-Dataset Sign Reversal\n")
    lines.append("Re-runs the 8-metric / consensus reversal verdict from "
                 "`src/eval/sign_reversal_forensic_audit.py` under five "
                 "numerical-cleaning regimes.\n")
    lines.append("## Reversal-robustness summary\n")
    for k in ["stable_no_reversal", "stable_reversal", "cleaning_resolves_reversal",
              "cleaning_induces_reversal", "unstable"]:
        lines.append(f"- **{k}**: {counts.get(k, 0)}")
    lines.append("")
    lines.append("`stable_reversal` ⇒ sign-reversal is robust to NaN/inf removal, "
                 "zero-duration removal, ±0.1% winsorisation, and log1p of heavy-"
                 "tailed features. These are the trustworthy reversals.\n")
    lines.append("`cleaning_induces_reversal` ⇒ reversal appears only after a "
                 "cleaning step — strongly suspicious of a numerical-edge artifact. "
                 "`unstable` ⇒ pattern depends on the cleaning regime — also "
                 "suspicious. `cleaning_resolves_reversal` ⇒ baseline reversal "
                 "disappears under cleaning — the numerical edges were driving it.\n")
    lines.append("## Per-feature verdict (SMD sign pattern iscx/usbvpn/vnat)\n")
    cols = ["feature"] + [f"smd_pattern__{m}" for m in MODES] + \
           [f"consensus__{m}" for m in MODES] + ["n_distinct_smd_patterns", "robustness_flag"]
    lines.append(wide[cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Zero-duration handling\n")
    lines.append(dur_prop.to_markdown(index=False))
    lines.append("")
    lines.append("## Heavy-tailed features (skew > 2 in any dataset, log1p applied)\n")
    lines.append(", ".join(heavy_tail_features) if heavy_tail_features else "_none_")
    lines.append("\n## Cleaning row-count log\n")
    lines.append(cleaning_log.to_markdown(index=False))
    lines.append("\n## Inventory & edge-case counts\n")
    lines.append("Top 30 (dataset, feature) rows by NaN+inf+negative count:\n")
    inv2 = inventory.copy()
    inv2["n_bad"] = inv2["n_nan"] + inv2["n_pos_inf"] + inv2["n_neg_inf"] + inv2["n_negative"]
    lines.append(inv2.sort_values("n_bad", ascending=False).head(30).to_markdown(index=False))
    lines.append("")
    lines.append("Full inventory: `tables/feature_value_inventory.csv`. "
                 "Per-feature skew table: `tables/feature_skew_table.csv`.\n")

    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skew-threshold", type=float, default=SKEW_THRESHOLD)
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.canonical_parquet}")
    df = pd.read_parquet(args.canonical_parquet)
    features = list(get_family("safe_core_plus_temporal"))
    print(f"[load] rows={len(df):,}  features={len(features)}")

    # Step 1: inventory
    print("[step 1] value inventory + percentiles")
    inventory = compute_value_inventory(df, features)
    inventory.to_csv(tables / "feature_value_inventory.csv", index=False)

    # Heavy-tailed selection
    skew_wide = inventory.pivot(index="feature", columns="dataset", values="skew")
    skew_wide["max_skew"] = skew_wide.max(axis=1)
    skew_wide["selected_for_log1p"] = skew_wide["max_skew"] > args.skew_threshold
    skew_wide.reset_index().to_csv(tables / "feature_skew_table.csv", index=False)
    heavy_tail_features = skew_wide.index[skew_wide["selected_for_log1p"]].tolist()
    print(f"[heavy-tail] {len(heavy_tail_features)} features (skew>{args.skew_threshold}): {heavy_tail_features}")

    # Step 2/3: zero-duration propagation
    print("[step 2/3] zero-duration propagation")
    dur_prop = compute_duration_propagation(df)
    dur_prop.to_csv(tables / "zero_duration_propagation.csv", index=False)

    # Step 4: re-run reversal under each regime
    print("[step 4] reversal under cleaning regimes")
    effects_by_mode: Dict[str, pd.DataFrame] = {}
    summaries_by_mode: Dict[str, pd.DataFrame] = {}
    cleaning_logs: List[Dict] = []
    for mode in MODES:
        cleaned, info = apply_cleaning(df, mode, features, heavy_tail_features)
        info["n_iscx"] = int((cleaned["dataset"] == "iscx").sum())
        info["n_usbvpn"] = int((cleaned["dataset"] == "usbvpn").sum())
        info["n_vnat"] = int((cleaned["dataset"] == "vnat").sum())
        cleaning_logs.append(info)
        eff = compute_effects_table(cleaned, features, analysis_name="numerical_edge",
                                    transform_name=mode, seed=args.seed)
        eff.to_csv(tables / f"effects_{mode}.csv", index=False)
        save_sign_matrices(eff, tables, suffix=f"_{mode}")
        summ = summarize_reversals(eff)
        summ.to_csv(tables / f"reversal_summary_{mode}.csv", index=False)
        effects_by_mode[mode] = eff
        summaries_by_mode[mode] = summ
        n_cons = int(summ["consensus_reversal"].sum())
        print(f"  mode={mode:18s}  rows={info['n_out']:>6,}  consensus_reversal_features={n_cons}")
    cleaning_log_df = pd.DataFrame(cleaning_logs)
    cleaning_log_df.to_csv(tables / "cleaning_row_counts.csv", index=False)

    # Step 5: verdict matrix + robustness
    print("[step 5] verdict matrix")
    long_df, wide = build_verdict_matrix(effects_by_mode, summaries_by_mode, features)
    long_df.to_csv(tables / "verdict_matrix_long.csv", index=False)
    wide.to_csv(tables / "verdict_matrix_wide.csv", index=False)

    # Final feature classification
    classification = wide[["feature", "robustness_flag"]].copy()
    classification["baseline_pattern"] = wide["smd_pattern__baseline"]
    classification["modes_disagreeing_with_baseline"] = wide.apply(
        lambda r: ",".join(m for m in MODES[1:] if r[f"smd_pattern__{m}"] != r["smd_pattern__baseline"]),
        axis=1,
    )
    classification.to_csv(tables / "feature_robustness_summary.csv", index=False)

    # Plot
    plot_sign_heatmap(long_df, figures)

    # Report + config
    write_report(out_dir, inventory, dur_prop, cleaning_log_df, heavy_tail_features,
                 skew_wide, wide)
    (out_dir / "audit_config.json").write_text(json.dumps({
        "canonical_parquet": str(args.canonical_parquet),
        "seed": args.seed,
        "skew_threshold": args.skew_threshold,
        "duration_eps": DUR_EPS,
        "winsor_quantiles": [WINSOR_LOW, WINSOR_HIGH],
        "modes": list(MODES),
        "heavy_tail_features": heavy_tail_features,
    }, indent=2), encoding="utf-8")

    # Final printout
    print("\n[final] robustness flag counts:")
    for k, v in wide["robustness_flag"].value_counts().items():
        print(f"  {k:30s} {v}")
    print(f"\n[final] outputs in {out_dir}")


if __name__ == "__main__":
    main()

