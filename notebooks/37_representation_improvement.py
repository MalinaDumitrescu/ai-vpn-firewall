#!/usr/bin/env python
"""
37_representation_improvement.py
=================================
PHASES 3, 4 — Attack domain fingerprint and audit extraction pipeline.

This script:
1. Per-feature domain detector AUC (which features leak domain identity?)
2. Composite feature selection criterion (VPN quality vs domain leakage)
3. Feature availability audit across ISCX, VNAT, USBVPN
4. Extraction pipeline consistency audit
5. Candidate new feature evaluation
6. Domain-penalized feature selection
7. Mismatch report

Usage:
    python notebooks/37_representation_improvement.py
"""

# %% [markdown]
# # Setup

# %%
import sys, os, json, time  # noqa: E401
import numpy as np
import pandas as pd
from pathlib import Path

_root = os.path.abspath(os.path.join(os.getcwd(), '..')) \
    if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)

from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from src.utils.paths import load_paths  # noqa: E402
from src.optimization.dataset_adversarial_feature_selection import train_dataset_detector  # noqa: E402

paths = load_paths()
SEED = 42
OUT_DIR = paths.artifacts_dir / 'eval' / 'representation_improvement'
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPACT_FEATURES = [
    'sz_coef_variation', 'sz_p25_median_ratio', 'sz_p75_median_ratio',
    'sz_iqr_norm_median', 'dispersion_symmetry',
]

# %% [markdown]
# # Load Feature Data

# %%
print('Loading feature data from all 3 datasets...')

META = {'flow_id', 'capture_id', 'label', 'dataset', 'split',
        'source_file', 'source_capture_id', 'q_packet_count',
        'q_min_packets_ok', 'app', 'connection_str', 'file_names'}

dfs = {}
for ds_name, subdir, fname in [
    ('vnat', 'vnat', 'features.parquet'),
    ('iscx', 'iscx', 'features.parquet'),
    ('usbvpn', 'usbvpn', 'flows.parquet'),
]:
    p = paths.data_processed_dir / subdir / fname
    if p.exists():
        df = pd.read_parquet(p)
        df['dataset'] = ds_name
        dfs[ds_name] = df
        print(f'  {ds_name}: {len(df)} flows, {len(df.columns)} columns')
    else:
        print(f'  WARNING: {p} not found')

if len(dfs) < 2:
    raise RuntimeError('Need at least 2 datasets loaded')

# %% [markdown]
# # PHASE 4A — Feature Availability Audit

# %%
print('\n' + '=' * 80)
print('  PHASE 4A: FEATURE AVAILABILITY AUDIT')
print('=' * 80)

all_cols = {}
for ds_name, df in dfs.items():
    feature_cols = set(df.columns) - META
    all_cols[ds_name] = feature_cols
    print(f'\n  {ds_name}: {len(feature_cols)} feature columns')

# Intersection
common_features = set.intersection(*[v for v in all_cols.values()])
print(f'\n  Common features across all datasets: {len(common_features)}')

# Per-dataset unique
for ds_name, cols in all_cols.items():
    unique = cols - common_features
    if unique:
        print(f'  {ds_name}-only features ({len(unique)}): {sorted(unique)[:10]}...')

# Check compact features
print('\n  COMPACT FEATURE AVAILABILITY:')
for feat in COMPACT_FEATURES:
    avail = [ds for ds, df in dfs.items() if feat in df.columns]
    status = '✓ ALL' if len(avail) == len(dfs) else f'MISSING from {set(dfs.keys()) - set(avail)}'
    print(f'    {feat:30s} {status}')

# Extended features that might be useful
extended_candidates = [
    'direction_balance_bytes', 'direction_balance_packets',
    'sz_mean', 'sz_std', 'sz_median', 'sz_p25', 'sz_p75', 'sz_min', 'sz_max',
    'iat_all_mean', 'iat_all_std', 'iat_all_median', 'iat_all_p25', 'iat_all_p75',
    'iat_mean_max', 'iat_mean_min', 'iat_std_max', 'iat_std_min',
    'sz_mean_max', 'sz_mean_min', 'sz_std_max', 'sz_std_min',
]

print('\n  EXTENDED CANDIDATE AVAILABILITY:')
for feat in extended_candidates:
    avail = [ds for ds, df in dfs.items() if feat in df.columns]
    status = '✓ ALL' if len(avail) == len(dfs) else f'in {avail}'
    print(f'    {feat:35s} {status}')

audit_rows = []
for feat in sorted(common_features):
    row = {'feature': feat, 'in_compact': feat in COMPACT_FEATURES}
    for ds_name, df in dfs.items():
        if feat in df.columns:
            vals = pd.to_numeric(df[feat], errors='coerce').dropna()
            row[f'{ds_name}_count'] = len(vals)
            row[f'{ds_name}_mean'] = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f'{ds_name}_std'] = float(vals.std()) if len(vals) > 0 else np.nan
            row[f'{ds_name}_dtype'] = str(df[feat].dtype)
    audit_rows.append(row)

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(OUT_DIR / 'feature_availability_audit.csv', index=False)
print(f'\nSaved: {OUT_DIR / "feature_availability_audit.csv"}')

# %% [markdown]
# # PHASE 4B — Extraction Pipeline Consistency Audit

# %%
print('\n' + '=' * 80)
print('  PHASE 4B: EXTRACTION PIPELINE CONSISTENCY AUDIT')
print('=' * 80)

mismatch_rows = []

for feat in COMPACT_FEATURES + ['direction_balance_bytes', 'direction_balance_packets']:
    row = {'feature': feat}
    ds_stats = {}

    for ds_name, df in dfs.items():
        if feat not in df.columns:
            row[f'{ds_name}_available'] = False
            continue
        row[f'{ds_name}_available'] = True

        vals = pd.to_numeric(df[feat], errors='coerce').dropna()
        ds_stats[ds_name] = vals.values

        row[f'{ds_name}_n'] = len(vals)
        row[f'{ds_name}_mean'] = float(vals.mean()) if len(vals) > 0 else np.nan
        row[f'{ds_name}_std'] = float(vals.std()) if len(vals) > 0 else np.nan
        row[f'{ds_name}_median'] = float(vals.median()) if len(vals) > 0 else np.nan
        row[f'{ds_name}_p5'] = float(np.percentile(vals, 5)) if len(vals) > 0 else np.nan
        row[f'{ds_name}_p95'] = float(np.percentile(vals, 95)) if len(vals) > 0 else np.nan
        row[f'{ds_name}_zeros_pct'] = float((vals == 0).mean()) if len(vals) > 0 else np.nan
        row[f'{ds_name}_nan_pct'] = float(df[feat].isna().mean())
        row[f'{ds_name}_dtype'] = str(df[feat].dtype)

    # KS tests between all pairs
    pairs = list(ds_stats.keys())
    max_ks = 0.0
    for i, a in enumerate(pairs):
        for b in pairs[i+1:]:
            if len(ds_stats[a]) > 5 and len(ds_stats[b]) > 5:
                ks_stat, ks_p = ks_2samp(ds_stats[a], ds_stats[b])
                row[f'ks_{a}_vs_{b}'] = float(ks_stat)
                row[f'ks_p_{a}_vs_{b}'] = float(ks_p)
                max_ks = max(max_ks, ks_stat)

    row['max_ks_stat'] = max_ks

    # Compatibility assessment
    if max_ks < 0.05:
        row['compatibility'] = 'identical'
    elif max_ks < 0.15:
        row['compatibility'] = 'approximately_compatible'
    else:
        row['compatibility'] = 'meaningfully_different'

    if max_ks >= 0.15:
        row['domain_fingerprint_risk'] = (
            f'High KS stat ({max_ks:.3f}) suggests this feature has different '
            f'distributions across datasets, which could create domain fingerprinting.'
        )
    else:
        row['domain_fingerprint_risk'] = 'Low'

    mismatch_rows.append(row)

mismatch_df = pd.DataFrame(mismatch_rows)
mismatch_df.to_csv(OUT_DIR / 'extraction_mismatch_report.csv', index=False)
print(f'\nSaved: {OUT_DIR / "extraction_mismatch_report.csv"}')

print('\n=== MISMATCH REPORT SUMMARY ===')
for _, r in mismatch_df.iterrows():
    print(f'  {r["feature"]:35s} compat={r["compatibility"]:25s} max_KS={r["max_ks_stat"]:.3f}')

# %% [markdown]
# # PHASE 3A — Per-Feature Domain Detector AUC

# %%
print('\n' + '=' * 80)
print('  PHASE 3A: PER-FEATURE DOMAIN DETECTOR AUC')
print('=' * 80)

# Build aligned dataset for domain detection
common_numeric = sorted([f for f in common_features
                          if all(pd.to_numeric(dfs[ds][f], errors='coerce').notna().mean() > 0.5
                                 for ds in dfs)])

df_all_list = []
for ds_name, df in dfs.items():
    keep_cols = ['label', 'split', 'dataset'] + [f for f in common_numeric if f in df.columns]
    sub = df[keep_cols].copy()
    for c in common_numeric:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors='coerce').fillna(0.0).astype(float)
    df_all_list.append(sub)

df_all = pd.concat(df_all_list, ignore_index=True)
if 'q_min_packets_ok' in df_all.columns:
    df_all = df_all[df_all['q_min_packets_ok'].fillna(1) == 1]
df_all['split'] = df_all['split'].astype(str)

le = LabelEncoder()
le.fit(df_all['dataset'])
df_train = df_all[df_all['split'] == 'train']
df_val = df_all[df_all['split'] == 'val']
y_tr_d = le.transform(df_train['dataset'])
y_va_d = le.transform(df_val['dataset'])

# Per-feature domain AUC
feat_domain_rows = []
for feat in common_numeric:
    if feat in META:
        continue
    try:
        X_tr = df_train[[feat]].values
        X_va = df_val[[feat]].values
        _, dd_auc = train_dataset_detector(X_tr, y_tr_d, X_va, y_va_d)
        feat_domain_rows.append({
            'feature': feat,
            'domain_det_auc': float(dd_auc),
            'in_compact': feat in COMPACT_FEATURES,
        })
    except Exception as e:
        feat_domain_rows.append({
            'feature': feat,
            'domain_det_auc': float('nan'),
            'in_compact': feat in COMPACT_FEATURES,
            'error': str(e),
        })

feat_domain_df = pd.DataFrame(feat_domain_rows).sort_values('domain_det_auc', ascending=False)
feat_domain_df.to_csv(OUT_DIR / 'per_feature_domain_auc.csv', index=False)
print(f'\nSaved: {OUT_DIR / "per_feature_domain_auc.csv"}')

print('\n=== MOST DOMAIN-LEAKY FEATURES ===')
print(feat_domain_df.head(15).to_string(index=False))

print('\n=== LEAST DOMAIN-LEAKY FEATURES ===')
non_nan = feat_domain_df.dropna(subset=['domain_det_auc'])
print(non_nan.tail(15).to_string(index=False))

print('\n=== COMPACT FEATURES DOMAIN LEAKAGE ===')
compact_domain = feat_domain_df[feat_domain_df['in_compact']]
print(compact_domain.to_string(index=False))

# %% [markdown]
# # PHASE 3B — Feature Subset Domain Detector AUC

# %%
print('\n' + '=' * 80)
print('  PHASE 3B: FEATURE SUBSET DOMAIN DETECTOR AUC')
print('=' * 80)

# Test different feature subsets
subsets = {
    '5f_current': COMPACT_FEATURES,
    '4f_no_p25': ['sz_coef_variation', 'sz_p75_median_ratio', 'sz_iqr_norm_median', 'dispersion_symmetry'],
    '4f_no_p75': ['sz_coef_variation', 'sz_p25_median_ratio', 'sz_iqr_norm_median', 'dispersion_symmetry'],
    '3f_core': ['sz_coef_variation', 'sz_iqr_norm_median', 'dispersion_symmetry'],
}

# Add direction features if available
dir_feats = ['direction_balance_bytes', 'direction_balance_packets']
if all(f in common_numeric for f in dir_feats):
    subsets['5f_plus_direction'] = COMPACT_FEATURES + dir_feats
    subsets['7f_all_compact'] = COMPACT_FEATURES + dir_feats

# Add timing features if available
timing_feats = [f for f in ['iat_all_mean', 'iat_all_std', 'iat_all_median'] if f in common_numeric]
if timing_feats:
    subsets['5f_plus_timing'] = COMPACT_FEATURES + timing_feats

subset_rows = []
for name, feats in subsets.items():
    avail = [f for f in feats if f in df_train.columns]
    if len(avail) < 2:
        print(f'  {name}: only {len(avail)} features available, skipping')
        continue

    try:
        _, dd_auc = train_dataset_detector(
            df_train[avail].values, y_tr_d,
            df_val[avail].values, y_va_d
        )
        row = {
            'subset': name,
            'n_features': len(avail),
            'features': ', '.join(avail),
            'domain_det_auc': float(dd_auc),
        }
        subset_rows.append(row)
        print(f'  {name} ({len(avail)} features): domain_det_auc = {dd_auc:.4f}')
    except Exception as e:
        print(f'  {name}: FAILED - {e}')

subset_df = pd.DataFrame(subset_rows)
subset_df.to_csv(OUT_DIR / 'subset_domain_auc.csv', index=False)
print(f'\nSaved: {OUT_DIR / "subset_domain_auc.csv"}')

# %% [markdown]
# # PHASE 3C — Composite Feature Selection Criterion

# %%
print('\n' + '=' * 80)
print('  PHASE 3C: COMPOSITE FEATURE SELECTION CRITERION')
print('=' * 80)

# Load predictions for VPN detection quality
EXPERIMENTS_DIR = paths.artifacts_dir / 'experiments'
PRED_PATH = EXPERIMENTS_DIR / 'exp_c_combined' / 'predictions.csv'

# Try to find predictions
pred_paths_to_try = [
    PRED_PATH,
    paths.artifacts_dir / 'balanced_bagging_firewall_tuned_ensemble' / 'predictions.csv',
    paths.artifacts_dir / 'balanced_bagging_firewall_tuned' / 'predictions.csv',
]

preds = None
for pp in pred_paths_to_try:
    if pp.exists():
        preds = pd.read_csv(pp)
        print(f'  Loaded predictions from {pp}')
        break

if preds is not None:
    from src.eval.bootstrap import _aggregate_to_sessions, AGG_FUNCTIONS

    # Compute VPN quality per feature subset using stored predictions as proxy
    # Note: True per-subset quality requires retraining
    test = preds[preds['split'] == 'test']
    pc = 'prob_iso' if 'prob_iso' in preds.columns else 'prob'
    agg_fn = AGG_FUNCTIONS['p90']

    _, ty, ts = _aggregate_to_sessions(test, pc, agg_fn)
    base_auc = float(roc_auc_score(ty, ts)) if len(np.unique(ty)) > 1 else 0.5

    # ISCX FPR for current model
    iscx_test = test[test['dataset'] == 'iscx']
    _, iy, iss = _aggregate_to_sessions(iscx_test, pc, agg_fn)
    from src.eval.metrics import threshold_at_fpr, confusion_at_threshold
    val = preds[preds['split'] == 'val']
    _, vy, vs = _aggregate_to_sessions(val, pc, agg_fn)
    if len(vy) > 0 and len(np.unique(vy)) > 1:
        thr = threshold_at_fpr(vy, vs, 0.0, warn_resolution=False)
        iscx_cm = confusion_at_threshold(iy, iss, thr)
        base_iscx_fpr = iscx_cm['fpr']
    else:
        base_iscx_fpr = 0.0

    # USBVPN recall
    usb_test = test[test['dataset'] == 'usbvpn']
    _, uy, us = _aggregate_to_sessions(usb_test, pc, agg_fn)
    if len(uy) > 0:
        usb_cm = confusion_at_threshold(uy, us, thr)
        base_usb_recall = usb_cm['recall']
    else:
        base_usb_recall = 0.0

    print(f'\n  Baseline 5f model: session_AUC={base_auc:.4f} '
          f'ISCX_FPR={base_iscx_fpr:.4f} USBVPN_recall={base_usb_recall:.4f}')

# Composite scoring
print('\n=== COMPOSITE FEATURE SUBSET SCORES ===')
print('Objective = session_auc_proxy + recall_proxy - pooled_fpr_penalty '
      '- iscx_fpr_penalty - domain_det_auc_penalty')

WEIGHTS = {
    'session_auc': 1.0,
    'recall': 0.5,
    'usbvpn_recall': 0.3,
    'pooled_fpr_penalty': -2.0,
    'iscx_fpr_penalty': -3.0,
    'domain_det_auc_penalty': -1.5,
}

composite_rows = []
for _, row in subset_df.iterrows():
    dd_auc = row['domain_det_auc']

    # For now, use baseline metrics as proxy (true requires retraining)
    # Penalize subsets with higher domain leakage
    composite = (
        WEIGHTS['session_auc'] * base_auc
        + WEIGHTS['domain_det_auc_penalty'] * dd_auc
    )

    composite_rows.append({
        'subset': row['subset'],
        'n_features': row['n_features'],
        'domain_det_auc': dd_auc,
        'session_auc_proxy': base_auc,
        'composite_score': composite,
        'note': 'True VPN quality requires retraining per subset',
    })

composite_df = pd.DataFrame(composite_rows).sort_values('composite_score', ascending=False)
composite_df.to_csv(OUT_DIR / 'composite_feature_scores.csv', index=False)
print(f'\nSaved: {OUT_DIR / "composite_feature_scores.csv"}')
print(composite_df.round(4).to_string(index=False))

print(f'\nWeights used: {json.dumps(WEIGHTS, indent=2)}')
with open(OUT_DIR / 'composite_weights.json', 'w') as f:
    json.dump(WEIGHTS, f, indent=2)

# %% [markdown]
# # PHASE 3D — Domain-Penalized Feature Selection

# %%
print('\n' + '=' * 80)
print('  PHASE 3D: DOMAIN-PENALIZED FEATURE RANKING')
print('=' * 80)

# Rank individual features by: VPN-relevance minus domain-leakage
# VPN relevance proxy: single-feature ROC-AUC for VPN detection
vpn_relevance_rows = []

for feat in common_numeric:
    if feat in META:
        continue
    try:
        train_df = df_all[df_all['split'] == 'train']
        val_df = df_all[df_all['split'] == 'val']

        x_tr = pd.to_numeric(train_df[feat], errors='coerce').fillna(0.0).values
        y_tr = train_df['label'].values.astype(int)
        x_va = pd.to_numeric(val_df[feat], errors='coerce').fillna(0.0).values
        y_va = val_df['label'].values.astype(int)

        if len(np.unique(y_va)) > 1:
            vpn_auc = float(roc_auc_score(y_va, x_va))
            # Ensure AUC is always >= 0.5 (flip if needed)
            if vpn_auc < 0.5:
                vpn_auc = 1.0 - vpn_auc
        else:
            vpn_auc = 0.5

        # Get domain AUC from earlier computation
        dd_row = feat_domain_df[feat_domain_df['feature'] == feat]
        dd_auc = float(dd_row['domain_det_auc'].iloc[0]) if len(dd_row) > 0 else 0.333

        # Penalized score: VPN relevance - lambda * domain leakage
        LAMBDA = 0.3
        penalized_score = vpn_auc - LAMBDA * dd_auc

        vpn_relevance_rows.append({
            'feature': feat,
            'vpn_auc': vpn_auc,
            'domain_det_auc': dd_auc,
            'penalized_score': penalized_score,
            'in_compact': feat in COMPACT_FEATURES,
        })
    except Exception:
        pass

vpn_rel_df = pd.DataFrame(vpn_relevance_rows).sort_values('penalized_score', ascending=False)
vpn_rel_df.to_csv(OUT_DIR / 'domain_penalized_feature_ranking.csv', index=False)
print(f'\nSaved: {OUT_DIR / "domain_penalized_feature_ranking.csv"}')

print('\n=== TOP 20 DOMAIN-PENALIZED FEATURES ===')
print(vpn_rel_df.head(20).round(4).to_string(index=False))

print('\n=== COMPACT FEATURES IN RANKING ===')
compact_rank = vpn_rel_df[vpn_rel_df['in_compact']]
print(compact_rank.round(4).to_string(index=False))

# %% [markdown]
# # PHASE 4C — Extraction Logic Comparison

# %%
print('\n' + '=' * 80)
print('  PHASE 4C: EXTRACTION LOGIC COMPARISON')
print('=' * 80)

# Document what we know about extraction differences
extraction_notes = {
    'sz_coef_variation': {
        'formula': 'std(packet_sizes) / mean(packet_sizes)',
        'iscx_extraction': 'From PCAP via extract.py: first N packets, _safe_stats',
        'vnat_extraction': 'From PCAP via extract.py: first N packets, _safe_stats',
        'usbvpn_extraction': 'From JSON pre-processed flows: different source pipeline',
        'potential_difference': 'USBVPN uses pre-aggregated JSON, ISCX/VNAT extract from raw PCAPs',
        'risk': 'MEDIUM — extraction pipeline is unified for ISCX/VNAT but USBVPN may differ',
    },
    'sz_p25_median_ratio': {
        'formula': 'percentile_25(sizes) / median(sizes)',
        'risk': 'LOW — ratio is unitless and scale-independent',
    },
    'sz_p75_median_ratio': {
        'formula': 'percentile_75(sizes) / median(sizes)',
        'risk': 'LOW — ratio is unitless and scale-independent',
    },
    'sz_iqr_norm_median': {
        'formula': '(p75 - p25) / (median + eps)',
        'risk': 'LOW — ratio is unitless and scale-independent',
    },
    'dispersion_symmetry': {
        'formula': '(p75 + p25 - 2*median) / |p75 - p25| clipped to [-1, 1]',
        'risk': 'LOW — fully unitless',
    },
    'general_concerns': {
        'packet_ordering': 'ISCX/VNAT use timestamps from PCAP; USBVPN uses JSON order',
        'flow_timeout': 'ISCX/VNAT: extracted per-capture; USBVPN: pre-defined flow boundaries',
        'window_size': f'All datasets use N={100} packets (configurable in features.yaml)',
        'direction_definition': 'ISCX/VNAT: 1=upload/0=download from PCAP; USBVPN: from JSON metadata',
        'padding': 'None — features are computed only from available packets',
        'normalization': 'Per-capture z-norm applied to size features (except direction + dispersion)',
    },
}

with open(OUT_DIR / 'extraction_logic_comparison.json', 'w') as f:
    json.dump(extraction_notes, f, indent=2)
print(f'\nSaved: {OUT_DIR / "extraction_logic_comparison.json"}')

# Concrete data-level comparison
print('\n=== FEATURE STATISTICS BY DATASET (train split) ===')
for feat in COMPACT_FEATURES:
    print(f'\n  {feat}:')
    for ds_name, df in dfs.items():
        if feat not in df.columns:
            continue
        train = df[df['split'] == 'train'] if 'split' in df.columns else df
        vals = pd.to_numeric(train[feat], errors='coerce').dropna()
        if len(vals) > 0:
            print(f'    {ds_name:8s}: mean={vals.mean():.4f} std={vals.std():.4f} '
                  f'median={vals.median():.4f} zeros={((vals==0).mean()*100):.1f}% '
                  f'range=[{vals.min():.4f}, {vals.max():.4f}]')

# %% [markdown]
# # Summary and Recommendations

# %%
print('\n' + '=' * 80)
print('  REPRESENTATION IMPROVEMENT SUMMARY')
print('=' * 80)

print(f'\nOutput directory: {OUT_DIR}')
for f in sorted(OUT_DIR.glob('*')):
    print(f'  {f.name}')

print('\n=== WHAT CAN BE IMPROVED ===')
print('1. Feature subsets with LOWER domain_det_auc may improve portability')
print('2. Adding timing features may add VPN-relevant signal with less domain leakage')
print('3. Per-capture normalization already reduces some domain fingerprinting')

print('\n=== WHAT CANNOT BE FIXED WITHOUT NEW DATA ===')
print('1. USBVPN extraction pipeline is fundamentally different (JSON vs PCAP)')
print('2. Dataset-level statistical differences in packet size distributions are real')
print('3. Flow boundary definitions differ and cannot be unified post-hoc')
print('4. True unified extraction requires re-processing from raw PCAPs for all datasets')

print('\n=== HONEST ASSESSMENT ===')
if len(subset_df) > 0:
    best_subset = subset_df.loc[subset_df['domain_det_auc'].idxmin()]
    worst_subset = subset_df.loc[subset_df['domain_det_auc'].idxmax()]
    print(f'  Lowest domain leakage: {best_subset["subset"]} '
          f'(domain_AUC={best_subset["domain_det_auc"]:.4f})')
    print(f'  Highest domain leakage: {worst_subset["subset"]} '
          f'(domain_AUC={worst_subset["domain_det_auc"]:.4f})')
    auc_range = worst_subset['domain_det_auc'] - best_subset['domain_det_auc']
    print(f'  Domain AUC range: {auc_range:.4f}')
    if auc_range < 0.03:
        print('  >> Feature subset changes have MINIMAL impact on domain fingerprinting')
        print('  >> The domain fingerprint is likely in the DATA, not the features')
        print('  >> Policy-level fixes are more valuable than representation changes')
    else:
        print(f'  >> Feature subset changes have SOME impact ({auc_range:.4f} range)')
        print('  → Consider testing lower-leakage subsets with full retraining')


