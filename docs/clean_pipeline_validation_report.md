# Clean Pipeline Validation Report

## 1. Executive summary

This validation suite for the clean VPN detection pipeline is designed to rigorously separate two classes of evidence:
- **Implementation correctness**: Ensures the pipeline is free from technical flaws such as data leakage, split contamination, feature inconsistency, improper preprocessing, and threshold provenance errors. These tests guarantee that reported performance is not inflated by methodological mistakes.
- **Scientific and deployment limitations**: Evaluates the model's stability, robustness to domain shift, and risk of false positives on realistic benign traffic. These validations reveal the true deployment risks and limitations that remain even after implementation correctness is assured.

## 2. Validation matrix

| Category                        | Tests                                               | Main artifacts                                         | Status   | Main finding                                                                 |
|----------------------------------|-----------------------------------------------------|--------------------------------------------------------|----------|------------------------------------------------------------------------------|
| Data leakage                     | test_no_capture_overlap_between_splits, test_thresholds_are_validation_derived_only | figures/validation/capture_overlap_matrix.png           | PASS     | No evidence of split leakage; thresholds are validation-derived only         |
| Split integrity                  | test_every_flow_assigned_to_exactly_one_split, test_each_dataset_split_contains_both_classes, test_no_single_capture_dominates_split, test_splitter_stability_across_random_seeds | figures/validation/split_composition_per_dataset.png, figures/validation/capture_size_distribution.png | PASS     | Splits are disjoint, balanced, and stable across seeds                      |
| Feature consistency              | test_no_metadata_columns_in_model_features, test_extracted_features_match_feature_columns_json, test_final_feature_family_is_direction_invariant, test_rate_features_recomputed_match_stored_values, test_raw_flow_schema_contains_required_columns | figures/validation/rate_feature_formula_consistency.png | PASS     | Model features are consistent, direction-invariant, and match schema        |
| Preprocessing and scaling        | test_scaler_is_fit_only_on_training_data            | (No figure)                                            | PASS     | Preprocessing/scaling is fit only on training data                           |
| Model stability                  | test_model_metric_stability_across_seeds, test_validation_selected_threshold_stability, test_session_level_bootstrap_confidence_intervals, test_model_performance_does_not_regress_below_baseline | figures/validation/metric_variability_across_seeds.png | PASS     | Model metrics and thresholds are stable across seeds and bootstraps         |
| Cross-dataset domain shift       | test_lodo_target_dataset_excluded_from_training, audit_dataset_fingerprinting_strength, audit_feature_distribution_shift_across_datasets, audit_cross_dataset_feature_sign_reversal | figures/validation/domain_fingerprinting_confusion_matrix.png, figures/validation/sign_reversal_heatmap.png | PASS     | Domain shift is measurable; sign reversals and fingerprinting are auditable |
| Benign false-positive scenarios  | test_benign_persona_false_positive_rate, audit_benign_persona_score_distributions, test_threshold_safety_tradeoff_for_benign_personas, test_session_level_false_positive_rate_for_benign_personas, test_benign_persona_schema_matches_model_features | figures/validation/benign_persona_block_rates.png, figures/validation/firewall_threshold_tradeoff.png, figures/validation/session_vs_flow_false_block_rate.png | XFAIL/NOT RUN | Structure in place, but real benign persona data is missing; tests xfail    |

## 3. Test inventory

| Test name | File path | Purpose | Status | CI-lightweight? |
|---|---|---|---|---|
| test_no_capture_overlap_between_splits | tests/test_split_integrity.py | No capture appears in more than one split | PASS | Yes |
| test_no_exact_duplicate_feature_rows_across_splits | tests/test_split_integrity.py | No duplicate feature rows across splits | PASS | Yes |
| test_thresholds_are_validation_derived_only | tests/test_data_leakage.py | Thresholds are derived only from validation data | PASS | Yes |
| test_scaler_is_fit_only_on_training_data | tests/test_preprocessing.py | Scaler/preprocessing fit only on training data | PASS | Yes |
| test_no_metadata_columns_in_model_features | tests/test_feature_consistency.py | No metadata columns in model features | PASS | Yes |
| test_every_flow_assigned_to_exactly_one_split | tests/test_split_integrity.py | Every flow assigned to one split | PASS | Yes |
| test_each_dataset_split_contains_both_classes | tests/test_split_integrity.py | Each split contains both classes | PASS | Yes |
| test_no_single_capture_dominates_split | tests/test_split_integrity.py | No single capture dominates a split | PASS | Yes |
| test_splitter_stability_across_random_seeds | tests/test_split_integrity.py | Splitter is stable across seeds | PASS | Yes |
| test_raw_flow_schema_contains_required_columns | tests/test_feature_consistency.py | Raw flow schema contains required columns | PASS | Yes |
| test_extracted_features_match_feature_columns_json | tests/test_feature_consistency.py | Extracted features match feature_columns.json | PASS | Yes |
| test_final_feature_family_is_direction_invariant | tests/test_feature_consistency.py | Final feature family is direction-invariant | PASS | Yes |
| test_rate_features_recomputed_match_stored_values | tests/test_feature_consistency.py | Rate features recomputed match stored values | PASS | Yes |
| test_model_metric_stability_across_seeds | tests/test_model_stability.py | Model metrics stable across seeds | PASS | Yes |
| test_validation_selected_threshold_stability | tests/test_model_stability.py | Validation-selected thresholds are stable | PASS | Yes |
| test_session_level_bootstrap_confidence_intervals | tests/test_model_stability.py | Session-level bootstrap CIs | PASS | No (notebook/heavy) |
| test_model_performance_does_not_regress_below_baseline | tests/test_model_stability.py | Model does not regress below baseline | PASS | Yes |
| test_benign_persona_false_positive_rate | tests/test_benign_false_positive_scenarios.py | Benign personas not blocked at high rate | XFAIL | No (requires data) |
| audit_benign_persona_score_distributions | src/clean_pipeline/validation/benign_scenarios.py | Audit score distributions for benign personas | NOT RUN | No (notebook/manual) |
| test_threshold_safety_tradeoff_for_benign_personas | tests/test_benign_false_positive_scenarios.py | Threshold tradeoff for benign block rate | XFAIL | No (requires data) |
| test_session_level_false_positive_rate_for_benign_personas | tests/test_benign_false_positive_scenarios.py | Session vs flow false block rates | XFAIL | No (requires data) |
| test_benign_persona_schema_matches_model_features | tests/test_benign_false_positive_scenarios.py | Persona features match model schema | XFAIL | No (requires data) |
| test_lodo_target_dataset_excluded_from_training | tests/test_cross_dataset.py | LODO target dataset excluded from training | PASS | Yes |
| audit_dataset_fingerprinting_strength | src/clean_pipeline/validation/domain_shift.py | Audit dataset fingerprinting | PASS | No (notebook/manual) |
| audit_feature_distribution_shift_across_datasets | src/clean_pipeline/validation/domain_shift.py | Audit feature distribution shift | PASS | No (notebook/manual) |
| audit_cross_dataset_feature_sign_reversal | src/clean_pipeline/validation/domain_shift.py | Audit cross-dataset feature sign reversal | PASS | No (notebook/manual) |
| test_prediction_score_calibration | tests/test_model_stability.py | Score calibration (if implemented) | MISSING | - |
| audit_firewall_zone_composition | src/clean_pipeline/validation/benign_scenarios.py | Firewall zone composition (if implemented) | MISSING | - |
| test_deployment_robustness_under_feature_perturbations | tests/test_model_stability.py | Deployment robustness (if implemented) | MISSING | - |

## 4. Figure inventory

| Figure | Path | What it shows | Slide | Interpretation |
|---|---|---|---|---|
| Capture overlap matrix | figures/validation/capture_overlap_matrix.png | Overlap between splits | Data leakage | Should be zero outside diagonal; no leakage detected |
| Split composition per dataset | figures/validation/split_composition_per_dataset.png | Class/split composition | Split integrity | Splits are balanced and disjoint |
| Capture size distribution | figures/validation/capture_size_distribution.png | Distribution of capture sizes | Split integrity | No single capture dominates |
| Metric variability across seeds | figures/validation/metric_variability_across_seeds.png | Model metric stability | Model stability | Metrics are stable across seeds |
| Rate feature formula consistency | figures/validation/rate_feature_formula_consistency.png | Rate feature correctness | Feature consistency | Recomputed rates match stored values |
| Domain fingerprinting confusion matrix | figures/validation/domain_fingerprinting_confusion_matrix.png | Dataset fingerprinting | Domain shift | Datasets are distinguishable; domain shift exists |
| Sign reversal heatmap | figures/validation/sign_reversal_heatmap.png | Feature sign reversals | Domain shift | Some features reverse sign across datasets |
| Benign persona block rates | figures/validation/benign_persona_block_rates.png | Block/monitor/pass rates for benign personas | Benign false-positive safety | MISSING: Awaiting real data |
| Firewall threshold tradeoff | figures/validation/firewall_threshold_tradeoff.png | Threshold vs block/recall tradeoff | Benign false-positive safety | MISSING: Awaiting real data |
| Session vs flow false block rate | figures/validation/session_vs_flow_false_block_rate.png | Session vs flow false block rates | Benign false-positive safety | MISSING: Awaiting real data |

## 5. Presentation outline

**Slide 1:** Validation Test Plan for the Clean VPN Detection Pipeline
- Overview of validation goals
- Clean separation of correctness vs deployment risk
- Figure: Validation map
- Speaker note: "This slide introduces the comprehensive validation plan, highlighting the dual focus on implementation correctness and real-world deployment risk."

**Slide 2:** Why validation is necessary
- Prevents accidental overfitting/leakage
- Ensures scientific credibility
- Supports safe deployment
- Speaker note: "Validation is essential to ensure that our results are trustworthy and that the firewall is safe for real-world use."

**Slide 3:** Clean pipeline validation map
- Categories: leakage, splits, features, scaling, stability, domain, benign
- Figure: Validation matrix/table
- Speaker note: "This map shows the main validation categories and how they interconnect."

**Slide 4:** Data leakage tests
- No capture overlap
- Thresholds from validation only
- Figure: Capture overlap matrix
- Speaker note: "We confirm that no data leaks between splits and that thresholds are set using only validation data."

**Slide 5:** Split integrity tests
- Disjoint, balanced splits
- Stable across seeds
- Figure: Split composition, capture size
- Speaker note: "Splits are constructed to be disjoint and balanced, with no single capture dominating."

**Slide 6:** Feature consistency tests
- No metadata in features
- Direction-invariant, schema-matched
- Figure: Rate feature consistency
- Speaker note: "Feature engineering is consistent and robust, ensuring no hidden information leaks into the model."

**Slide 7:** Model stability tests
- Metrics stable across seeds
- No regression below baseline
- Figure: Metric variability
- Speaker note: "Model performance is stable and does not regress below the established baseline."

**Slide 8:** Cross-dataset validation
- Domain fingerprinting
- Feature sign reversals
- Figure: Domain confusion, sign reversal heatmap
- Speaker note: "We audit for domain shift and feature sign reversals to understand transfer risks."

**Slide 9:** Benign false-positive safety
- Structure for benign persona tests
- Awaiting real data
- Figure: (Placeholder for block rates)
- Speaker note: "The suite is ready to evaluate benign false positives as soon as real data is available."

**Slide 10:** Proposed benign scenarios
- HTTPS, video call, gaming, RDP, etc.
- Fixture format and documentation
- Table: Scenario list
- Speaker note: "We have defined the required benign scenarios and documented the fixture format for future data."

**Slide 11:** Summary matrix
- Validation matrix/table
- Status of each category
- Figure: Validation matrix
- Speaker note: "This summary table shows the status and main findings for each validation category."

**Slide 12:** Final message
- Clean pipeline is methodologically correct
- Remaining risk is real, not a bug
- Speaker note: "Our validation suite proves the pipeline is correct; remaining risks are due to real-world challenges, not evaluation flaws."

## 6. Final conclusion

The validation suite is designed to prove two things separately:
1. The clean pipeline is methodologically correct: no leakage, consistent features, valid splits, and validation-only thresholds.
2. The remaining transfer weakness is real: it comes from structural dataset mismatch and realistic deployment risk, not from an evaluation bug.
