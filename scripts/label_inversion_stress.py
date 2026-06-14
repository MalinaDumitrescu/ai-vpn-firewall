"""
Label-inversion stress test for the cross-dataset sign-reversal claim.

Hypothesis under test:
  H_invert(D) := "dataset D has its VPN/nonVPN labels accidentally inverted
                  (e.g. a labelling-convention bug) — the apparent cross-
                  dataset sign reversal is the artefact of that bug, not a
                  property of the underlying flow distributions."

If H_invert(D) is true for some D, then artificially inverting the labels of
D in the canonical table should produce a *coherent* world where signs agree
across all three datasets (≈ zero strict reversals).  Conversely, if the real
reversal pattern reflects genuinely heterogeneous class structure across
datasets, no single full-dataset inversion will resolve it — most features
will still disagree across at least one dataset pair.

For each scenario S in
    { baseline,
      invert_iscx, invert_usbvpn, invert_vnat,
      invert_iscx_and_usbvpn, invert_iscx_and_vnat, invert_usbvpn_and_vnat,
      invert_all }
we recompute the 8-metric reversal verdict (loose + strict-magnitude) and
compare the resulting per-(feature, dataset) SMD-sign matrix against the real
pattern.

Outputs:
  artifacts/thesis_finalization/nb53_sign_reversal_audit/label_inversion_stress/
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.clean_pipeline.feature_families import get_family
from src.eval.sign_reversal_forensic_audit import (
    DATASETS,
    METRIC_THRESHOLDS,
    compute_effects_table,
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
    / "label_inversion_stress"
)

STRICT_METRICS = {m: thr for m, thr in METRIC_THRESHOLDS.items() if thr > 0.0}


def invert_labels(df: pd.DataFrame, datasets_to_invert: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    if datasets_to_invert:
        mask = out["dataset"].isin(list(datasets_to_invert)).to_numpy()
        out.loc[mask, "label"] = 1 - out.loc[mask, "label"].to_numpy()
    return out


def signs_matrix(eff: pd.DataFrame, features: Sequence[str], sign_col: str = "sign_smd") -> pd.DataFrame:
    """Return a (feature × dataset) DataFrame of integer signs in {-1,0,+1}."""
    mat = (
        eff.set_index(["feature", "dataset"])[sign_col]
        .unstack("dataset")
        .reindex(index=list(features), columns=list(DATASETS))
        .fillna(0)
        .astype(int)
    )
    return mat


def strict_reversal_features(eff: pd.DataFrame, features: Sequence[str]) -> Tuple[int, List[str]]:
    flagged = []
    for feat in features:
        sub = eff[eff["feature"] == feat]
        any_strict = False
        for metric, thr in STRICT_METRICS.items():
            vals = sub.set_index("dataset")[metric].reindex(list(DATASETS)).to_numpy(dtype=float)
            pos = [v for v in vals if np.isfinite(v) and v > thr]
            neg = [v for v in vals if np.isfinite(v) and v < -thr]
            if pos and neg:
                any_strict = True
                break
        if any_strict:
            flagged.append(feat)
    return len(flagged), flagged


def evaluate_scenario(df: pd.DataFrame, features: Sequence[str], datasets_to_invert: Sequence[str], seed: int) -> Dict:
    df_s = invert_labels(df, datasets_to_invert)
    eff = compute_effects_table(df_s, features,
                                analysis_name="label_inversion",
                                transform_name="+".join(datasets_to_invert) or "baseline",
                                seed=seed)
    summ = summarize_reversals(eff)
    n_loose = int(summ["consensus_reversal"].sum())
    n_strict, strict_features = strict_reversal_features(eff, features)
    smd_mat = signs_matrix(eff, features, "sign_smd")
    return {
        "effects": eff,
        "summary": summ,
        "n_loose_consensus": n_loose,
        "n_strict_magnitude": n_strict,
        "strict_features": strict_features,
        "smd_matrix": smd_mat,
    }


def pattern_similarity(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[float, int, int]:
    """Fraction (and counts) of (feature, dataset) cells where a and b have identical sign."""
    arr_a = a.to_numpy()
    arr_b = b.to_numpy()
    matches = int(np.sum(arr_a == arr_b))
    total = int(arr_a.size)
    return matches / total, matches, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.canonical_parquet}")
    df = pd.read_parquet(args.canonical_parquet)
    features = list(get_family("safe_core_plus_temporal"))
    n_feats = len(features)
    print(f"[load] rows={len(df):,}  features={n_feats}")

    # All scenarios
    scenarios: List[Tuple[str, List[str]]] = [("baseline", [])]
    for d in DATASETS:
        scenarios.append((f"invert_{d}", [d]))
    for a, b in combinations(DATASETS, 2):
        scenarios.append((f"invert_{a}_and_{b}", [a, b]))
    scenarios.append(("invert_all", list(DATASETS)))

    # Run baseline first to anchor similarity comparisons
    print("[run] baseline (real labels)")
    base = evaluate_scenario(df, features, [], args.seed)
    base_smd = base["smd_matrix"]

    rows: List[Dict] = []
    rows.append({
        "scenario": "baseline",
        "datasets_inverted": "",
        "n_loose_reversal_features": base["n_loose_consensus"],
        "n_strict_reversal_features": base["n_strict_magnitude"],
        "smd_cells_matching_real": int(base_smd.size),
        "smd_total_cells": int(base_smd.size),
        "similarity_to_real": 1.0,
        "strict_features": "; ".join(base["strict_features"]),
    })
    base_smd.to_csv(tables / "smd_sign_matrix_baseline.csv")

    # Save real pattern as the reference
    print(f"  baseline: loose={base['n_loose_consensus']}  strict={base['n_strict_magnitude']}")

    # All other scenarios
    for name, ds_list in scenarios[1:]:
        print(f"[run] {name}")
        s = evaluate_scenario(df, features, ds_list, args.seed)
        sim, matches, total = pattern_similarity(s["smd_matrix"], base_smd)
        rows.append({
            "scenario": name,
            "datasets_inverted": ",".join(ds_list),
            "n_loose_reversal_features": s["n_loose_consensus"],
            "n_strict_reversal_features": s["n_strict_magnitude"],
            "smd_cells_matching_real": matches,
            "smd_total_cells": total,
            "similarity_to_real": sim,
            "strict_features": "; ".join(s["strict_features"]),
        })
        s["smd_matrix"].to_csv(tables / f"smd_sign_matrix_{name}.csv")
        print(f"  {name}: loose={s['n_loose_consensus']}  strict={s['n_strict_magnitude']}  "
              f"sim_to_real={sim:.3f}  ({matches}/{total} cells)")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "scenario_summary.csv", index=False)

    # ---- Conclusion -----------------------------------------------------------
    real_strict = base["n_strict_magnitude"]
    # Single-dataset inversion candidates
    single = [r for r in rows if r["scenario"].startswith("invert_") and "," not in r["datasets_inverted"] and r["scenario"] != "invert_all"]
    pair = [r for r in rows if r["scenario"].startswith("invert_") and "," in r["datasets_inverted"]]
    invert_all_row = next(r for r in rows if r["scenario"] == "invert_all")

    min_single = min(single, key=lambda r: r["n_strict_reversal_features"])
    min_pair = min(pair, key=lambda r: r["n_strict_reversal_features"])

    # An inversion "resolves" the reversal if it drops strict reversals from
    # ~real_strict down to <= 2 (essentially zero on 21 features).
    RESOLVED_THR = max(2, int(0.1 * n_feats))

    diagnoses: List[str] = []
    if invert_all_row["similarity_to_real"] >= 0.95:
        diagnoses.append("Inverting ALL datasets reproduces the real sign matrix → "
                         "real pattern is sign-symmetric (every feature has flipped signs "
                         "in every dataset relative to a hypothetical coherent world). "
                         "This is mathematically equivalent to a global label-convention "
                         "flip and does not localise the reversal.")
    if min_single["n_strict_reversal_features"] <= RESOLVED_THR:
        diagnoses.append(
            f"CRITICAL: inverting only `{min_single['datasets_inverted']}` collapses strict "
            f"reversals from {real_strict} to {min_single['n_strict_reversal_features']}. "
            f"This is the signature of an accidental full-dataset label inversion in "
            f"`{min_single['datasets_inverted']}`."
        )
    elif min_pair["n_strict_reversal_features"] <= RESOLVED_THR:
        diagnoses.append(
            f"WARNING: inverting `{min_pair['datasets_inverted']}` together collapses "
            f"strict reversals from {real_strict} to {min_pair['n_strict_reversal_features']}. "
            f"A two-dataset pair is unlikely to be a labelling bug but is worth a manual "
            f"label-convention check."
        )
    else:
        diagnoses.append(
            f"Best single-dataset inversion ({min_single['scenario']}) leaves "
            f"{min_single['n_strict_reversal_features']}/{n_feats} strict reversals "
            f"(real = {real_strict}); best pair-inversion ({min_pair['scenario']}) "
            f"leaves {min_pair['n_strict_reversal_features']}/{n_feats}. No single "
            f"or paired full-dataset label flip reduces the strict reversal count to "
            f"≤ {RESOLVED_THR}. The observed reversal pattern is therefore **not** "
            f"consistent with an accidental full-dataset label inversion."
        )

    # JSON + Markdown
    (out_dir / "label_inversion_summary.json").write_text(
        json.dumps({
            "n_features": n_feats,
            "real_loose_reversal_features": base["n_loose_consensus"],
            "real_strict_reversal_features": real_strict,
            "resolved_threshold": RESOLVED_THR,
            "scenarios": rows,
            "diagnoses": diagnoses,
        }, indent=2),
        encoding="utf-8",
    )

    md: List[str] = []
    md.append("# Label-Inversion Stress Test\n")
    md.append(
        "For each candidate label-inversion hypothesis, the canonical 21-feature × "
        "3-dataset table is artificially modified by flipping the VPN/nonVPN label "
        "for every flow in the named dataset(s). The reversal verdict is then "
        "recomputed and the resulting per-(feature, dataset) SMD-sign matrix is "
        "compared against the real-data sign matrix.\n"
    )
    md.append("## Scenario summary\n")
    cols = ["scenario", "n_loose_reversal_features", "n_strict_reversal_features",
            "smd_cells_matching_real", "smd_total_cells", "similarity_to_real"]
    md.append(summary[cols].to_markdown(index=False))
    md.append("")
    md.append("## Strict-reversal feature lists per scenario\n")
    md.append(summary[["scenario", "strict_features"]].to_markdown(index=False))
    md.append("")
    md.append("## Conclusion\n")
    for d in diagnoses:
        md.append(f"- {d}")
    md.append("")
    (out_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print()
    print("=" * 70)
    print(f"real strict-reversal features:        {real_strict} / {n_feats}")
    print(f"best single-dataset inversion:        {min_single['scenario']:30s} "
          f"strict={min_single['n_strict_reversal_features']}  sim={min_single['similarity_to_real']:.3f}")
    print(f"best pair inversion:                  {min_pair['scenario']:30s} "
          f"strict={min_pair['n_strict_reversal_features']}  sim={min_pair['similarity_to_real']:.3f}")
    print(f"invert_all sim_to_real:               {invert_all_row['similarity_to_real']:.3f}")
    print()
    for d in diagnoses:
        print(d)
    print(f"\nOutputs in {out_dir}")


if __name__ == "__main__":
    main()

