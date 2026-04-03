#!/usr/bin/env python
"""Quick audit of all pipeline outputs."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

pipeline = [
    ("STEP 1: run_clean_pipeline_full.py", "Feature extraction + base models", [
        "artifacts/clean_pipeline/features.parquet",
        "artifacts/clean_pipeline/run_metadata.json",
        "artifacts/clean_pipeline/features_train.parquet",
        "artifacts/clean_pipeline/features_val.parquet",
        "artifacts/clean_pipeline/features_test.parquet",
    ]),
    ("STEP 2: run_clean_evaluation_full.py", "Comprehensive eval (Parts 1-7)", [
        "artifacts/clean_pipeline/eval_v3/xgb_results.csv",
        "artifacts/clean_pipeline/eval_v3/family_search_leaderboard.csv",
        "artifacts/clean_pipeline/eval_v3/family_search_verdict.json",
        "artifacts/clean_pipeline/eval_v3/final_honest_verdict.json",
        "artifacts/clean_pipeline/eval_v3/ensemble_mean_results.csv",
        "artifacts/clean_pipeline/eval_v3/majority_voting_results.csv",
        "artifacts/clean_pipeline/eval_v3/logistic_stacking_results.csv",
        "artifacts/clean_pipeline/eval_v3/clean_lodo_results.csv",
        "artifacts/clean_pipeline/eval_v3/clean_policy_grid.csv",
        "artifacts/clean_pipeline/eval_v3/clean_deployment_recommendation.json",
        "artifacts/clean_pipeline/eval_v3/repeated_split_summary.csv",
        "artifacts/clean_pipeline/eval_v3/feature_stability_rank.csv",
    ]),
    ("STEP 3: run_final_thesis_deliverables.py", "Final thesis deliverables (8 parts)", [
        "artifacts/thesis_finalization/final/final_feature_family_decision.json",
        "artifacts/thesis_finalization/final/final_feature_family_decision.md",
        "artifacts/thesis_finalization/final/cross_dataset_recalibration.csv",
        "artifacts/thesis_finalization/final/cross_dataset_recalibration.md",
        "artifacts/thesis_finalization/final/cross_dataset_recalibration_summary.json",
        "artifacts/thesis_finalization/final/per_dataset_feature_importance.csv",
        "artifacts/thesis_finalization/final/per_dataset_feature_importance.md",
        "artifacts/thesis_finalization/final/per_dataset_top10_overlap.csv",
        "artifacts/thesis_finalization/final/per_dataset_rank_correlation.csv",
        "artifacts/thesis_finalization/final/representation_domain_tradeoff.csv",
        "artifacts/thesis_finalization/final/representation_domain_tradeoff.md",
        "artifacts/thesis_finalization/final/final_model_comparison_table.csv",
        "artifacts/thesis_finalization/final/final_model_comparison_table.md",
        "artifacts/thesis_finalization/final/deployment_architecture.md",
        "artifacts/thesis_finalization/final/deployment_modes_table.csv",
        "artifacts/thesis_finalization/final/drift_aware_deployment_interpretation.md",
        "artifacts/thesis_finalization/final/final_thesis_safe_conclusion.json",
        "artifacts/thesis_finalization/final/final_thesis_safe_conclusion.md",
    ]),
    ("STEP 4: run_robustness_methods.py", "Robustness methods evaluation", [
        "artifacts/thesis_finalization/final/robustness_methods_comparison.csv",
        "artifacts/thesis_finalization/final/robustness_methods_summary.md",
        "artifacts/thesis_finalization/final/final_improvement_verdict.json",
    ]),
    ("STEP 5: run_dataset_structure_analysis.py", "Dataset structure analysis (8 parts)", [
        "artifacts/dataset_structure_analysis/feature_distribution_stats.csv",
        "artifacts/dataset_structure_analysis/dataset_distance_matrix.csv",
        "artifacts/dataset_structure_analysis/dataset_distance_heatmap.png",
        "artifacts/dataset_structure_analysis/dataset_distance_top10_features.png",
        "artifacts/dataset_structure_analysis/pca_dataset_projection.png",
        "artifacts/dataset_structure_analysis/pca_loadings.csv",
        "artifacts/dataset_structure_analysis/tsne_dataset_projection.png",
        "artifacts/dataset_structure_analysis/umap_dataset_projection.png",
        "artifacts/dataset_structure_analysis/domain_classifier_report.csv",
        "artifacts/dataset_structure_analysis/domain_feature_importance.csv",
        "artifacts/dataset_structure_analysis/domain_feature_importance.png",
        "artifacts/dataset_structure_analysis/feature_dataset_dependence_tests.csv",
        "artifacts/dataset_structure_analysis/feature_dataset_dependence.png",
        "artifacts/dataset_structure_analysis/correlation_per_dataset.png",
        "artifacts/dataset_structure_analysis/correlation_difference_heatmaps.png",
        "artifacts/dataset_structure_analysis/correlation_difference_stats.csv",
        "artifacts/dataset_structure_analysis/vpn_importance_per_dataset.csv",
        "artifacts/dataset_structure_analysis/importance_rank_correlation.csv",
        "artifacts/dataset_structure_analysis/importance_instability_comparison.png",
        "artifacts/dataset_structure_analysis/importance_rank_heatmap.png",
        "artifacts/dataset_structure_analysis/dataset_structure_summary.md",
        "artifacts/dataset_structure_analysis/structural_shift_verdict.json",
    ]),
]

print("=" * 70)
print("FULL PIPELINE AUDIT")
print("=" * 70)

grand_expected = 0
grand_found = 0
step_status = []

for step_name, desc, outputs in pipeline:
    print(f"\n{'─' * 70}")
    print(f"  {step_name}")
    print(f"  {desc}")
    print(f"{'─' * 70}")

    step_ok = True
    step_found = 0
    step_total = len(outputs)

    for out_path in outputs:
        grand_expected += 1
        full = ROOT / out_path
        if full.exists():
            sz = os.path.getsize(full)
            grand_found += 1
            step_found += 1
            if sz < 10:
                print(f"  [WARN] {out_path}  ({sz} bytes — suspiciously small!)")
                step_ok = False
            else:
                print(f"  [ OK ] {out_path}  ({sz:,} bytes)")
        else:
            print(f"  [MISS] {out_path}")
            step_ok = False

    status = "COMPLETE" if step_ok else ("PARTIAL" if step_found > 0 else "NOT RUN")
    step_status.append((step_name, status, step_found, step_total))
    print(f"\n  => {status} ({step_found}/{step_total} files)")

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
for name, status, found, total in step_status:
    icon = "OK" if status == "COMPLETE" else "!!" if status == "PARTIAL" else "XX"
    print(f"  [{icon}] {name}: {status} ({found}/{total})")

print(f"\n  GRAND TOTAL: {grand_found}/{grand_expected} key outputs found")

if grand_found == grand_expected:
    print("\n  *** ALL PIPELINE STEPS COMPLETED SUCCESSFULLY ***")
else:
    missing = grand_expected - grand_found
    print(f"\n  *** {missing} OUTPUTS MISSING — see details above ***")

print(f"{'=' * 70}")

