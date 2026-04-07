# Next-Stage Fix Plan — 3-Dataset VPN Firewall Detection Pipeline

**Generated:** 2025-04-02
**Status:** Implementation-ready
**Scope:** Fix remaining domain fingerprinting, complete leakage ablation, harden deployment policy

---

## 1. Executive Diagnosis

**What is working:**
- Detector quality is genuinely strong (session AUC 0.9879, flow AUC 0.9780)
- Train-test gap is small (0.0164) — no serious overfitting
- wt5+isotonic deployment policy achieves excellent pooled metrics (FPR 0.0099, recall 0.9444)
- Oracle/diagnostic thresholds are now properly isolated
- The A/B layered evaluation (detector vs policy) is correct

**What is still broken:**
1. **Domain fingerprinting remains catastrophically high** (AUC 0.9769). Features encode dataset identity almost perfectly. This means any threshold calibrated on one domain mixture will fail on a different mixture.
2. **ISCX FPR is still 0.0588** under the best deployment policy — 6× the pooled FPR. This is the direct symptom of domain fingerprinting.
3. **Leakage ablation is incomplete** — you have domain AUC for 4 subsets but full VPN metrics for only the 5f baseline. Without retrained models for 4f/3f subsets, you cannot say whether dropping domain-leaky features hurts VPN detection.
4. **`threshold_at_fpr(target_fpr=0.0)` returns `max(benign_scores)`**, which is the single most extreme benign session score. With ~100 benign val sessions, this is dominated by one outlier. One noisy benign session changes deployment behavior.
5. **The "no sweep alternative improved the tradeoff" wording** is misleading because the sweep was already restricted to deployable budgets — it should say the sweep confirmed that the wt5 mode outperforms p90 under deployable FPR budgets.

**Root cause chain:**
```
sz_p25_median_ratio and sz_p75_median_ratio encode packet-size distribution shape
  → ISCX has structurally different packet-size distributions from VNAT/USBVPN
  → domain detector achieves AUC 0.9769
  → any global threshold is locally miscalibrated per domain
  → ISCX benign scores overlap with VPN scores
  → ISCX FPR explodes under global threshold
```

---

## 2. Highest-Priority Fixes (Strict Order)

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| **P0** | Complete retrained leakage ablation (3f, 4f×2) | 2–3 hours compute | Determines whether domain fingerprinting can be reduced without destroying VPN detection |
| **P1** | Harden `threshold_at_fpr` for low-N val sets | 30 min code | Eliminates outlier-driven threshold instability |
| **P2** | Add per-dataset calibration shift analysis | 1 hour code | Quantifies exactly how much ISCX calibration differs |
| **P3** | Fix sweep wording | 5 min | Thesis honesty |
| **P4** | Generate all deliverable tables | 1 hour | Thesis completeness |
| **P5** | Evaluate domain-adversarial normalization (if P0 shows 3f is viable) | 3–4 hours | Potential step-change in domain robustness |

---

## 3. Exact Experiments to Run

### Experiment A: Full Retrained 4f (no sz_p25_median_ratio)

**Feature set:** `sz_coef_variation`, `sz_p75_median_ratio`, `sz_iqr_norm_median`, `dispersion_symmetry`

**Hypothesis:** `sz_p25_median_ratio` has the highest per-feature domain AUC among the 5f set. Dropping it should reduce domain fingerprinting with minimal VPN detection loss.

**How to run:**
```python
# In a new notebook cell or script — AFTER loading data pool via load_and_prepare_data()
from src.models.train_balanced_bagging_ensemble import run_balanced_bagging

FEATS_4F_NO_P25 = ['sz_coef_variation', 'sz_p75_median_ratio',
                    'sz_iqr_norm_median', 'dispersion_symmetry']

# Use IDENTICAL pipeline/params as the 5f primary
result_4f_no_p25 = run_experiment(
    name='4f_no_p25',
    df_pool=df_pool,          # same pool as 5f primary
    feature_cols=FEATS_4F_NO_P25,
    output_subdir='ablation_4f_no_p25',
)
```

**Exact metrics to collect:** (same for all ablation experiments)
| Metric | Source |
|--------|--------|
| flow_roc_auc | `results['isotonic']['test_overall']['auc']` |
| flow_pr_auc | `results['isotonic']['test_overall']['pr_auc']` |
| session_roc_auc_p90 | `session_eval_p90(preds)['session_roc_auc']` |
| session_pr_auc_p90 | `session_eval_p90(preds)['session_pr_auc']` |
| block_recall_p90 | `session_eval_p90(preds)['block_recall_at_zero_fp']` |
| block_fpr_pooled | `session_eval_p90(preds)['block_fpr']` |
| block_fpr_iscx | `session_eval_p90(preds, ds_filter='iscx')['block_fpr']` |
| block_fpr_vnat | `session_eval_p90(preds, ds_filter='vnat')['block_fpr']` |
| block_fpr_usbvpn | `session_eval_p90(preds, ds_filter='usbvpn')['block_fpr']` |
| precision_p90 | Compute from confusion_at_threshold |
| ece | `expected_calibration_error(y_test, prob_iso)['ece']` |
| brier | `brier_score_loss(y_test, prob_iso)` |
| train_test_gap | `train_auc - test_auc` |
| domain_det_auc | `train_dataset_detector(X_train, y_domain_train, X_val, y_domain_val)` |
| thr_range | `max(per_ds_thresholds) - min(per_ds_thresholds)` |
| session_roc_auc_wt5 | `session_eval_wt5(preds)['session_roc_auc']` |
| block_recall_wt5 | `session_eval_wt5(preds)['block_recall_at_zero_fp']` |
| block_fpr_wt5 | `session_eval_wt5(preds)['block_fpr']` |
| wt5_beats_p90 | `block_fpr_wt5 < block_fpr_p90 and block_recall_wt5 >= block_recall_p90 * 0.95` |

### Experiment B: Full Retrained 4f (no sz_p75_median_ratio)

**Feature set:** `sz_coef_variation`, `sz_p25_median_ratio`, `sz_iqr_norm_median`, `dispersion_symmetry`

```python
FEATS_4F_NO_P75 = ['sz_coef_variation', 'sz_p25_median_ratio',
                    'sz_iqr_norm_median', 'dispersion_symmetry']

result_4f_no_p75 = run_experiment(
    name='4f_no_p75',
    df_pool=df_pool,
    feature_cols=FEATS_4F_NO_P75,
    output_subdir='ablation_4f_no_p75',
)
```

### Experiment C: Full Retrained 3f Core

**Feature set:** `sz_coef_variation`, `sz_iqr_norm_median`, `dispersion_symmetry`

**Hypothesis:** These 3 features have the lowest per-feature domain AUC. If VPN detection holds with 3f, this is the most domain-robust option.

```python
FEATS_3F_CORE = ['sz_coef_variation', 'sz_iqr_norm_median', 'dispersion_symmetry']

result_3f = run_experiment(
    name='3f_core',
    df_pool=df_pool,
    feature_cols=FEATS_3F_CORE,
    output_subdir='ablation_3f_core',
)
```

### Experiment D: wt5 Deployment Policy Evaluation for Each Ablation

For each retrained model (A, B, C), evaluate the full deployment-policy matrix:

```python
def evaluate_deployment_policies(preds_csv_path, experiment_name):
    """Evaluate all deployment policies for one retrained model."""
    preds = pd.read_csv(preds_csv_path)
    val_df = preds[preds['split'] == 'val']
    test_df = preds[preds['split'] == 'test']

    agg_rules = {
        'p90': lambda x: np.percentile(x, 90),
        'wt5': weighted_top5,
    }
    fpr_targets = [0.0, 0.01]
    policies = []

    for agg_name, agg_fn in agg_rules.items():
        for target_fpr in fpr_targets:
            # Val threshold
            vl = val_df.groupby('capture_id')['label'].max()
            vs = val_df.groupby('capture_id')['prob_iso'].agg(agg_fn)
            vc = vl.index.intersection(vs.index)
            thr = threshold_at_fpr(vl.loc[vc].values, vs.loc[vc].values, target_fpr)

            # Test metrics — pooled + per-dataset
            tl = test_df.groupby('capture_id')['label'].max()
            ts = test_df.groupby('capture_id')['prob_iso'].agg(agg_fn)
            tc = tl.index.intersection(ts.index)
            y_t = tl.loc[tc].values; s_t = ts.loc[tc].values
            cm = confusion_at_threshold(y_t, s_t, thr)

            row = {
                'experiment': experiment_name,
                'aggregation': agg_name,
                'val_target_fpr': target_fpr,
                'threshold': thr,
                'block_recall': cm['recall'],
                'block_fpr': cm['fpr'],
                'precision': cm['precision'],
            }

            for ds in ['iscx', 'vnat', 'usbvpn']:
                ds_sub = test_df[test_df['dataset'] == ds]
                dsl = ds_sub.groupby('capture_id')['label'].max()
                dss = ds_sub.groupby('capture_id')['prob_iso'].agg(agg_fn)
                dc = dsl.index.intersection(dss.index)
                dcm = confusion_at_threshold(dsl.loc[dc].values, dss.loc[dc].values, thr)
                row[f'fpr_{ds}'] = dcm['fpr']
                row[f'recall_{ds}'] = dcm['recall']

            policies.append(row)

    return pd.DataFrame(policies)
```

### Critical Rules for All Experiments

1. **Same `df_pool`** — loaded once via `load_and_prepare_data()`, used for all experiments
2. **Same `FeaturePipeline`** — fitted on the same train split (inside `run_experiment`)
3. **Same Optuna hyperparameters** — load from `artifacts/optuna_*_firewall_best_params.json`
4. **Same seed=42**
5. **Same `bags_per_family=3`, `majority_ratio=1.0`**
6. **Same split files** — capture-level splits are deterministic from text files
7. **Same `target_fprs='0.0,0.001,0.005,0.01'`**
8. **Never use test labels for threshold selection** — thresholds always from val

---

## 4. Code / Pipeline Changes

### 4A. Harden `threshold_at_fpr` for Low-N Validation

**Problem:** With ~100 benign val sessions, `target_fpr=0.0` returns `max(benign_scores)`. One outlier benign session controls the entire deployment threshold.

**Fix:** Add a `min_fpr_resolution` parameter that warns when the target FPR is below the achievable resolution.

```python
# In src/eval/metrics.py — modify threshold_at_fpr

def threshold_at_fpr(y_true, p, target_fpr, *, warn_resolution=True):
    y_true = _to_numpy_1d(y_true).astype(int)
    p = _safe_probs(_to_numpy_1d(p).astype(float))

    neg_scores = p[y_true == 0]
    if neg_scores.size == 0:
        return 1.000001

    n_neg = len(neg_scores)
    fpr_resolution = 1.0 / n_neg

    if warn_resolution and target_fpr < fpr_resolution and target_fpr > 0:
        import warnings
        warnings.warn(
            f"target_fpr={target_fpr} is below achievable resolution "
            f"1/{n_neg}={fpr_resolution:.4f}. Threshold will be identical "
            f"to target_fpr=0. Consider using target_fpr >= {fpr_resolution:.4f}.",
            UserWarning,
            stacklevel=2,
        )

    t = np.quantile(neg_scores, 1.0 - target_fpr)
    return float(t)
```

**Also add** a `threshold_at_fpr_robust` function that returns the threshold AND metadata:

```python
def threshold_at_fpr_robust(y_true, p, target_fpr):
    """Returns (threshold, metadata_dict) with resolution diagnostics."""
    y_true = _to_numpy_1d(y_true).astype(int)
    p = _safe_probs(_to_numpy_1d(p).astype(float))

    neg_scores = p[y_true == 0]
    n_neg = len(neg_scores)
    n_unique = len(np.unique(neg_scores))
    fpr_resolution = 1.0 / max(n_neg, 1)

    if neg_scores.size == 0:
        thr = 1.000001
    else:
        thr = float(np.quantile(neg_scores, 1.0 - target_fpr))

    return thr, {
        'n_negatives': n_neg,
        'n_unique_scores': n_unique,
        'fpr_resolution': fpr_resolution,
        'target_fpr': target_fpr,
        'target_achievable': target_fpr >= fpr_resolution or target_fpr == 0,
        'implied_false_positives': int(round(target_fpr * n_neg)),
    }
```

### 4B. Replace `target_fpr=0.0` with Minimum Feasible FPR

**Recommendation:** Do NOT stop using `target_fpr=0.0`. It is mathematically well-defined (returns max of benign scores). Instead:

1. Always report the achievable FPR resolution alongside the threshold
2. Use `target_fpr=0.01` as the **primary deployment target** (1 FP per 100 benign sessions is operationally meaningful)
3. Keep `target_fpr=0.0` as the **strict/conservative** option
4. In the thesis, phrase it as: "threshold calibrated for minimal achievable FPR on validation (resolution = 1/N_benign = 0.01)"

### 4C. Fix Feature Extraction Mismatch Across Datasets

**Likely problem:** USBVPN loads pre-computed features from `flows.parquet` while VNAT/ISCX extract features from raw flow data via `extract_features_from_flows()`. If the extraction code differs between the USBVPN preprocessing pipeline and the VNAT/ISCX pipeline, the same physical feature (e.g., `sz_coef_variation`) could have subtly different semantics.

**How to detect:**
```python
# Compare feature distributions across datasets for TRAINING data only
for feat in FEATS_5F:
    for ds in ['iscx', 'vnat', 'usbvpn']:
        subset = df_pool[(df_pool['dataset'] == ds) & (df_pool['split'] == 'train')]
        print(f'{feat} | {ds}: mean={subset[feat].mean():.4f}, '
              f'std={subset[feat].std():.4f}, '
              f'median={subset[feat].median():.4f}, '
              f'p5={subset[feat].quantile(0.05):.4f}, '
              f'p95={subset[feat].quantile(0.95):.4f}')
```

If distributions differ dramatically between datasets (especially for benign traffic), this confirms the feature extraction mismatch is a domain fingerprinting source.

**How to fix:** If mismatch is found:
1. Re-extract USBVPN features using the exact same `extract_features_from_flows()` function as VNAT/ISCX
2. OR: apply per-dataset z-normalization before model training (risky — changes feature meaning)
3. OR: accept the mismatch and document it as a data limitation

### 4D. Fix Sweep Wording

In `31_final_3dataset_firewall_evaluation.py`, replace the current wording at the end of Section 6:

**Current (misleading):**
> "No alternative operating point improved the tradeoff"

**Replace with:**
> "The restricted sweep over deployable FPR budgets [0.00, 0.01, 0.02, 0.05, 0.10] confirmed that the reference threshold (val-derived, target_fpr=0) provides the best available recall/FPR tradeoff within the achievable resolution of this validation set. The wt5+isotonic deployment mode (Section 11) outperforms p90 under the same FPR budget."

---

## 5. Fair Comparison Protocol (Checklist)

### Mandatory for Every Experiment

- [ ] **Same data pool**: loaded via `load_and_prepare_data()` with identical filters
- [ ] **Same splits**: capture-level splits from canonical text files (never re-split)
- [ ] **Same pipeline transform**: `FeaturePipeline().fit(train).transform(pool)` — fitted only on train
- [ ] **Same hyperparameters**: Optuna-tuned params from `artifacts/optuna_*_firewall_best_params.json`
- [ ] **Same bagging config**: `bags_per_family=3`, `majority_ratio=1.0`, `seed=42`
- [ ] **Same calibration**: isotonic fitted on val, applied to all splits
- [ ] **Same aggregation comparison**: always report BOTH p90 and wt5 for session metrics
- [ ] **Same threshold source**: val-derived only for deployable claims
- [ ] **No test labels in threshold selection**: oracle thresholds allowed only in diagnostic tables
- [ ] **No dataset-identity features**: no `dataset` column in training features
- [ ] **Report per-dataset FPR**: ISCX, VNAT, USBVPN — not just pooled
- [ ] **Report domain detector AUC**: train the same XGBoost domain classifier on the same train/val splits
- [ ] **Report threshold transferability range**: `max(per_ds_thresholds) - min(per_ds_thresholds)` using test-derived (oracle) thresholds — labeled DIAGNOSTIC ONLY

### What NOT to Do

1. **Do not compare flow AUC across different feature counts** as the sole criterion — fewer features may have lower AUC but better domain robustness
2. **Do not claim "zero FPR"** unless the observed test FPR is literally 0.0000
3. **Do not mix p90 and wt5 session metrics** in the same ranking row — compare like for like
4. **Do not use domain-detector AUC as a training signal** — it is a diagnostic, not an optimization target (unless doing adversarial training, which is a separate experiment)
5. **Do not re-tune Optuna hyperparameters per feature subset** — this would confound feature-set effects with hyperparameter effects. Use the same tuned params.
6. **Do not filter or reweight test data** — the test set must remain identical across experiments

---

## 6. Reporting Artifacts to Generate

### 6A. `ablation_results.csv`

**Columns:**
```
experiment, n_features, features,
flow_roc_auc, flow_pr_auc,
session_roc_auc_p90, session_pr_auc_p90,
block_recall_p90, block_fpr_pooled,
block_fpr_iscx, block_fpr_vnat, block_fpr_usbvpn,
precision_p90, ece, brier, train_test_gap,
domain_det_auc, thr_range,
session_roc_auc_wt5, block_recall_wt5, block_fpr_wt5,
wt5_beats_p90
```

**Rows:** 5f_baseline, 4f_no_p25, 4f_no_p75, 3f_core

**Why it matters:** This is the central table for the feature-leakage ablation. It shows the tradeoff between VPN detection quality and domain robustness for each feature subset.

**Thesis use:** Table in the "Feature Selection and Domain Robustness" section. Answers: "Can we reduce domain fingerprinting without destroying VPN detection?"

### 6B. `domain_leakage_comparison.csv`

**Columns:**
```
feature_set, n_features,
domain_det_auc, domain_det_auc_delta_vs_5f,
per_feature_domain_aucs (as JSON string or separate columns),
iscx_fpr, iscx_fpr_delta_vs_5f,
session_auc, session_auc_delta_vs_5f
```

**Why it matters:** Directly shows the domain AUC vs VPN detection quality tradeoff curve.

**Thesis use:** Scatter plot: x = domain_det_auc, y = session_roc_auc_p90. Each point is a feature subset. Shows the Pareto frontier.

### 6C. `deployment_policy_ranking.csv`

**Columns:**
```
experiment, aggregation, val_target_fpr,
threshold, block_recall, block_fpr,
precision, fpr_iscx, fpr_vnat, fpr_usbvpn,
rank_by_pooled_fpr, rank_by_iscx_fpr,
deployment_status
```

**Rows:** One row per (experiment × aggregation × fpr_target) combination

**Why it matters:** Directly compares deployment policies across feature subsets. Answers: "Which combination of features + aggregation + threshold gives the best firewall behavior?"

**Thesis use:** Table in the "Deployment Policy Evaluation" section.

### 6D. `threshold_resolution_report.csv`

**Columns:**
```
experiment, split, aggregation,
n_benign_sessions, n_unique_benign_scores,
fpr_resolution, target_fpr, achieved_threshold,
target_achievable, implied_false_positives
```

**Why it matters:** Documents the quantile resolution limit for each experiment's validation set. Makes the thesis honest about what FPR granularity is achievable.

### 6E. `per_dataset_operating_points.csv`

**Columns:**
```
experiment, dataset, aggregation, threshold_source,
block_recall, block_fpr, precision,
session_auc, n_sessions, n_vpn, n_benign,
avg_benign_score, avg_vpn_score, score_separation
```

**Rows:** One row per (experiment × dataset × aggregation)

**Why it matters:** Shows exactly how each deployment policy performs per domain. ISCX should always be highlighted.

### 6F. `feature_distribution_by_dataset.csv`

**Columns:**
```
feature, dataset, split, mean, std, median, p5, p25, p75, p95,
ks_stat_vs_pooled, ks_pvalue_vs_pooled
```

**Why it matters:** Quantifies the distributional mismatch that causes domain fingerprinting. A KS test per feature per dataset against the pooled distribution shows which features are most domain-discriminative.

**Thesis use:** Table or heatmap in the "Domain Sensitivity Analysis" section.

### 6G. `final_thesis_safe_summary.md`

Updated markdown with:
- Dual recommendation (detector + policy)
- Ablation results table
- Domain leakage comparison
- Deployment policy ranking
- Honest threshold resolution disclosure
- Updated thesis wording

---

## 7. Success Criteria

### Minimum Acceptable (to claim improvement)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Session AUC (p90) | >= 0.95 | Below this, detector quality is degraded too much |
| Block Recall (best policy) | >= 0.90 | Below this, too many VPN sessions are missed |
| Pooled Block FPR | < current 0.0099 | Must not get worse |
| ISCX Block FPR | < current 0.0588 | This is the main target for improvement |
| Domain Detector AUC | < 0.96 | Meaningful reduction from 0.9769 |
| Train-Test Gap | < 0.03 | No overfitting |
| Precision | >= 0.90 | Firewall must not block too much benign traffic |

### Strong Improvement

| Metric | Threshold |
|--------|-----------|
| Domain Detector AUC | < 0.90 |
| ISCX Block FPR | < 0.03 |
| Session AUC (p90) | >= 0.97 |
| Block Recall | >= 0.94 |

### "Not Worth It" Cases

- Session AUC drops below 0.93 (detector fundamentally weakened)
- ISCX FPR gets worse despite lower domain AUC (wrong tradeoff)
- Train-test gap increases above 0.05 (feature subset causes overfitting)
- Pooled FPR increases above 0.02 (deployment policy regression)
- Domain AUC decreases by less than 0.01 (noise, not signal)

### Warning Signs of Biased Comparison

1. **A subset "wins" only because its threshold is derived from a different quantile** — check that the same `target_fpr` and same val set are used
2. **Session AUC improved but block recall at the deployment threshold dropped** — means the improvement is at a different operating point than deployment uses
3. **Domain AUC dropped but ISCX FPR didn't improve** — the fingerprinting reduction didn't help the deployment problem
4. **Pooled FPR improved but ISCX FPR got worse** — the improvement is in VNAT/USBVPN, not where it matters

---

## 8. Domain Robustness Improvement Proposals

### 8A. Feature Removal (P0 — Do This First)

**What:** Run experiments A, B, C from Section 3.

**Expected benefit:** 3f core may reduce domain AUC from 0.977 to ~0.96. Based on the existing domain-only ablation, the 3f set has domain AUC 0.9605.

**Main risk:** VPN detection quality may drop if `sz_p25_median_ratio` and `sz_p75_median_ratio` carry genuine VPN signal.

**Implementation difficulty:** Low — just change the feature list.

**Thesis-safe:** Yes — this is standard feature ablation.

**How to verify:** Compare 3f vs 5f on all metrics in `ablation_results.csv`. If session AUC >= 0.95 and ISCX FPR < 0.04, the 3f set is a viable alternative.

### 8B. Per-Dataset Z-Normalization (Pre-Pipeline)

**What:** Before `FeaturePipeline.fit()`, z-normalize each feature within each dataset using only that dataset's training statistics.

```python
for ds in df_pool['dataset'].unique():
    mask = (df_pool['dataset'] == ds) & (df_pool['split'] == 'train')
    for feat in feature_cols:
        mu = df_pool.loc[mask, feat].mean()
        sigma = df_pool.loc[mask, feat].std()
        ds_mask = df_pool['dataset'] == ds
        df_pool.loc[ds_mask, feat] = (df_pool.loc[ds_mask, feat] - mu) / max(sigma, 1e-8)
```

**Expected benefit:** Removes dataset-level location/scale differences. Could reduce domain AUC significantly.

**Main risk:** If VPN vs benign separation is at different absolute scales per dataset, this could erase the signal. Also: the z-normalization parameters are dataset-specific, which means at deployment time you need to know which dataset the traffic resembles — this is a form of domain knowledge leakage.

**Implementation difficulty:** Low.

**Thesis-safe:** YES, if you:
1. Fit z-normalization only on train split per dataset
2. Apply to val/test using train statistics
3. Disclose that deployment requires domain-specific normalization parameters

**How to verify:** Run the domain detector on z-normalized features. If domain AUC drops below 0.90 but session AUC stays above 0.95, this is a win.

**Verdict:** Worth trying as Experiment E, but only AFTER the feature ablation (P0) results are in.

### 8C. Domain Adversarial Training

**What:** Add a gradient reversal layer that penalizes domain-predictive representations during training.

**Expected benefit:** Could reduce domain fingerprinting at the representation level.

**Main risk:** Complex to implement for a bagging ensemble of XGBoost/LightGBM/CatBoost (gradient reversal is a neural network technique). Would require switching to a neural backbone.

**Implementation difficulty:** HIGH — requires new model architecture.

**Thesis-safe:** Yes, but adds significant complexity.

**Verdict:** NOT recommended for this stage. The feature-level ablation (8A) and normalization (8B) are simpler and more interpretable. If those fail, consider this for a future chapter.

### 8D. Sample Reweighting by Dataset

**What:** Upweight ISCX samples during training so the model pays more attention to getting ISCX right.

```python
# In run_balanced_bagging, add sample weights
df_t['sample_weight'] = 1.0
iscx_mask = df_t['dataset'] == 'iscx'
df_t.loc[iscx_mask, 'sample_weight'] = 2.0  # or proportional to 1/dataset_size
```

**Expected benefit:** Could improve ISCX-specific metrics.

**Main risk:** May overfit to ISCX patterns. May hurt VNAT/USBVPN performance.

**Implementation difficulty:** Low — XGBoost/LightGBM/CatBoost all support `sample_weight`.

**Thesis-safe:** Yes, if clearly documented.

**Verdict:** Worth trying as Experiment F, but ONLY after P0 ablation. Not a priority.

### 8E. Leave-One-Dataset-Out Calibration

**What:** For each test dataset, calibrate the threshold using only validation data from the OTHER two datasets. This tests whether calibration transfers across domains.

**Expected benefit:** Reveals whether the threshold is actually transferable or just happens to work on the pooled val set.

**Main risk:** None — this is purely diagnostic.

**Implementation difficulty:** Low.

**Thesis-safe:** Yes — it is a proper cross-domain evaluation.

**How to implement:**
```python
for held_out_ds in ['iscx', 'vnat', 'usbvpn']:
    # Calibrate on val data EXCLUDING held_out_ds
    cal_val = val_df[val_df['dataset'] != held_out_ds]
    # Evaluate on test data FROM held_out_ds only
    test_sub = test_df[test_df['dataset'] == held_out_ds]
    # Compute threshold from cal_val, evaluate on test_sub
```

**Verdict:** Strongly recommended as a diagnostic addition. Add to Section 8 of the notebook.

### 8F. Adding New Features

**What:** Add 1-2 features that capture VPN behavior without encoding dataset identity.

**Candidates:**
- `sz_entropy` — entropy of packet size distribution (high for encrypted/padded traffic)
- `iat_coef_variation` — coefficient of variation of inter-arrival times (if IAT features are available)

**Main risk:** New features may introduce new domain fingerprinting channels.

**Implementation difficulty:** Medium — need to add extraction logic to `extract_features_from_flows()` and ensure USBVPN computes them identically.

**Thesis-safe:** Yes, if treated as a new experiment with proper ablation.

**Verdict:** Deferred. Only consider after P0 results show that the current feature set is insufficient.

---

## 9. Recommended Thesis Wording After Next Round

### If ablation shows 3f core is viable (domain AUC < 0.96, session AUC >= 0.95):

> "Feature ablation reveals that the two percentile-ratio features (`sz_p25_median_ratio`, `sz_p75_median_ratio`) are the primary carriers of dataset-identity signal. A compact 3-feature model using only `sz_coef_variation`, `sz_iqr_norm_median`, and `dispersion_symmetry` reduces domain-detector AUC from 0.977 to [X] while maintaining session-level ROC-AUC of [Y]. Under weighted-top-5-mean aggregation with isotonic calibration, this model achieves Block Recall = [Z], Block FPR = [W], and crucially reduces ISCX FPR from 0.059 to [V]. Deployment remains conditional on local calibration, but the reduced domain fingerprinting makes threshold transfer more reliable across network environments."

### If ablation shows 3f core is NOT viable (session AUC < 0.95):

> "Feature ablation shows that the percentile-ratio features contribute both to domain fingerprinting and to VPN detection. Removing them reduces domain-detector AUC by [X] but also reduces session-level ROC-AUC from 0.988 to [Y], which falls below the deployment-acceptable threshold. The 5-feature model therefore represents a deliberate tradeoff: strong VPN detection at the cost of domain sensitivity. Deployment requires per-environment threshold calibration, and the ISCX false-positive rate ([Z]) must be managed through local calibration or adaptive thresholds."

### If detector quality stays excellent but domain robustness improves only a little:

> "The detector's ranking quality is excellent (session AUC = [X]), confirming that the multi-dataset training paradigm substantially improves cross-domain VPN detection. However, domain fingerprinting remains inherent to the packet-size distribution features: even the most reduced 3-feature set achieves domain-detector AUC = [Y], indicating that dataset identity is encoded in the fundamental statistical properties of network traffic across capture environments. This finding suggests that domain-robust VPN detection will require either feature-level normalization strategies or fundamentally different feature families (e.g., timing-only features), and that any size-based feature set will carry some degree of domain sensitivity. Under the current best deployment policy (weighted_top5_mean + isotonic, val FPR->0), the system achieves [metrics], which is operationally acceptable if complemented by periodic threshold recalibration."

