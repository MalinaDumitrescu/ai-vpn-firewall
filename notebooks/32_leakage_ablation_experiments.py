#!/usr/bin/env python
"""
32_leakage_ablation_experiments.py
==================================
Full retrained leakage-ablation experiments for 3-dataset VPN detection.

Trains 3 additional models (4f_no_p25, 4f_no_p75, 3f_core) using the
IDENTICAL pipeline, hyperparameters, and splits as the 5f primary baseline.
Evaluates each on all deployment-relevant metrics including per-dataset FPR,
domain detector AUC, and wt5 vs p90 deployment-policy comparison.

Usage:
    python notebooks/32_leakage_ablation_experiments.py

Output:
    artifacts/experiments/ablation_4f_no_p25/
    artifacts/experiments/ablation_4f_no_p75/
    artifacts/experiments/ablation_3f_core/
    artifacts/eval/ablation_results.csv
    artifacts/eval/deployment_policy_ranking.csv
    artifacts/eval/domain_leakage_comparison.csv
    artifacts/eval/feature_distribution_by_dataset.csv
    artifacts/eval/threshold_resolution_report.csv
    artifacts/eval/per_dataset_operating_points.csv
"""

# %% [markdown]
# # Setup

# %%
import sys, os, json, time  # noqa: E401
import numpy as np
import pandas as pd

_root = os.path.abspath(os.path.join(os.getcwd(), '..')) \
    if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from src.pipeline.data_preparation import load_and_prepare_data  # noqa: E402
from src.pipeline.feature_pipeline import FeaturePipeline  # noqa: E402
from src.models.train_balanced_bagging_ensemble import run_balanced_bagging  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    threshold_at_fpr, threshold_at_fpr_robust, confusion_at_threshold
)
from src.eval.calibration_diagnostics import expected_calibration_error  # noqa: E402
from src.optimization.dataset_adversarial_feature_selection import train_dataset_detector  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402

paths = load_paths()
SEED = 42
EXPERIMENTS_DIR = paths.artifacts_dir / 'experiments'
EVAL_DIR = paths.artifacts_dir / 'eval'
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ── Aggregation helpers ──
def p90_agg(x):
    return float(np.percentile(x, 90))

def weighted_top5(x):
    s = np.sort(x)[::-1][:5]
    w = np.arange(len(s), 0, -1, dtype=float)
    return float(np.average(s, weights=w))

# %% [markdown]
# # Load Data and Hyperparameters

# %%
print('Loading 3-dataset pool...')
df_pool = load_and_prepare_data()
print(f'Pool: {len(df_pool):,} flows, '
      f'{df_pool["dataset"].nunique()} datasets, '
      f'{df_pool["capture_id"].nunique()} captures')
for ds in sorted(df_pool['dataset'].unique()):
    for sp in ['train', 'val', 'test']:
        n = ((df_pool['dataset'] == ds) & (df_pool['split'] == sp)).sum()
        print(f'  {ds}/{sp}: {n:,} flows')

# Load Optuna-tuned hyperparameters (SAME as 5f primary)
with open(paths.artifacts_dir / 'optuna_xgboost_firewall_best_params.json') as f:
    xgb_params = json.load(f)
xgb_params.update({
    'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'n_estimators': 1000, 'random_state': SEED, 'n_jobs': 1,
    'early_stopping_rounds': 150,
})

with open(paths.artifacts_dir / 'optuna_catboost_firewall_best_params.json') as f:
    cat_params = json.load(f)
cat_params.update({
    'iterations': 1000, 'loss_function': 'Logloss', 'random_seed': SEED,
    'verbose': False, 'allow_writing_files': False, 'early_stopping_rounds': 150,
})

with open(paths.artifacts_dir / 'optuna_lgbm_firewall_best_params.json') as f:
    lgbm_params = json.load(f)
lgbm_params.update({
    'objective': 'binary', 'metric': 'binary_logloss',
    'boosting_type': 'gbdt', 'n_estimators': 1000,
    'verbose': -1, 'random_state': SEED, 'n_jobs': 1,
})

print('Hyperparameters loaded.')

# %% [markdown]
# # Define Feature Subsets

# %%
FEATS_5F = ['sz_coef_variation', 'sz_p25_median_ratio', 'sz_p75_median_ratio',
            'sz_iqr_norm_median', 'dispersion_symmetry']

ABLATIONS = {
    '5f_baseline': {
        'features': FEATS_5F,
        'subdir': 'exp_c_combined',  # already trained — just load
        'train': False,
    },
    '4f_no_p25': {
        'features': ['sz_coef_variation', 'sz_p75_median_ratio',
                      'sz_iqr_norm_median', 'dispersion_symmetry'],
        'subdir': 'ablation_4f_no_p25',
        'train': True,
    },
    '4f_no_p75': {
        'features': ['sz_coef_variation', 'sz_p25_median_ratio',
                      'sz_iqr_norm_median', 'dispersion_symmetry'],
        'subdir': 'ablation_4f_no_p75',
        'train': True,
    },
    '3f_core': {
        'features': ['sz_coef_variation', 'sz_iqr_norm_median',
                      'dispersion_symmetry'],
        'subdir': 'ablation_3f_core',
        'train': True,
    },
}

# %% [markdown]
# # Session-Level Evaluation Helpers

# %%
def session_eval(preds_df, agg_fn, prob_col='prob_iso', split='test', ds_filter=None):
    """Compute session-level metrics with specified aggregation."""
    t = preds_df[preds_df['split'] == split].copy()
    if ds_filter is not None:
        t = t[t['dataset'] == ds_filter]
    if len(t) == 0 or t['label'].nunique() < 2:
        return {}

    pc = prob_col if prob_col in t.columns else 'prob'
    sl = t.groupby('capture_id')['label'].max()
    ss = t.groupby('capture_id')[pc].agg(agg_fn)
    c = sl.index.intersection(ss.index)
    y = sl.loc[c].values
    s = ss.loc[c].values

    out = {}
    if len(np.unique(y)) > 1:
        out['session_roc_auc'] = float(roc_auc_score(y, s))
        out['session_pr_auc'] = float(average_precision_score(y, s))

    # Val threshold for block metrics
    v = preds_df[preds_df['split'] == 'val']
    if ds_filter is not None:
        # For per-dataset eval, still use POOLED val for threshold (fair comparison)
        pass
    vl = v.groupby('capture_id')['label'].max()
    vs = v.groupby('capture_id')[pc].agg(agg_fn)
    vc = vl.index.intersection(vs.index)
    if len(vc) > 0 and vl.loc[vc].nunique() > 1:
        thr = threshold_at_fpr(vl.loc[vc].values, vs.loc[vc].values, 0.0,
                               warn_resolution=False)
    else:
        thr = 0.5

    cm = confusion_at_threshold(y, s, thr)
    out['block_recall'] = cm['recall']
    out['block_fpr'] = cm['fpr']
    out['precision'] = cm['precision']
    out['block_threshold'] = float(thr)

    return out


def full_eval(preds_df, experiment_name, feature_list):
    """Full evaluation of one experiment — all metrics needed for ablation table."""
    t = preds_df[preds_df['split'] == 'test']
    v = preds_df[preds_df['split'] == 'val']
    tr = preds_df[preds_df['split'] == 'train']
    pc = 'prob_iso'

    row = {'experiment': experiment_name, 'n_features': len(feature_list),
           'features': ', '.join(feature_list)}

    # Flow-level
    if t['label'].nunique() > 1:
        row['flow_roc_auc'] = float(roc_auc_score(t['label'], t[pc]))
        row['flow_pr_auc'] = float(average_precision_score(t['label'], t[pc]))
        row['brier'] = float(brier_score_loss(t['label'], t[pc]))
        ece_data = expected_calibration_error(t['label'].values, t[pc].values)
        row['ece'] = ece_data['ece']

    # Train AUC
    if len(tr) > 0 and tr['label'].nunique() > 1:
        row['train_auc'] = float(roc_auc_score(tr['label'], tr[pc]))
        row['train_test_gap'] = row['train_auc'] - row.get('flow_roc_auc', 0)

    # Session p90
    sess_p90 = session_eval(preds_df, p90_agg)
    row['session_roc_auc_p90'] = sess_p90.get('session_roc_auc', float('nan'))
    row['session_pr_auc_p90'] = sess_p90.get('session_pr_auc', float('nan'))
    row['block_recall_p90'] = sess_p90.get('block_recall', float('nan'))
    row['block_fpr_pooled'] = sess_p90.get('block_fpr', float('nan'))
    row['precision_p90'] = sess_p90.get('precision', float('nan'))

    # Per-dataset p90
    for ds in ['iscx', 'vnat', 'usbvpn']:
        ds_sess = session_eval(preds_df, p90_agg, ds_filter=ds)
        row[f'block_fpr_{ds}'] = ds_sess.get('block_fpr', float('nan'))
        row[f'block_recall_{ds}'] = ds_sess.get('block_recall', float('nan'))

    # Session wt5
    sess_wt5 = session_eval(preds_df, weighted_top5)
    row['session_roc_auc_wt5'] = sess_wt5.get('session_roc_auc', float('nan'))
    row['block_recall_wt5'] = sess_wt5.get('block_recall', float('nan'))
    row['block_fpr_wt5'] = sess_wt5.get('block_fpr', float('nan'))
    row['precision_wt5'] = sess_wt5.get('precision', float('nan'))

    # wt5 beats p90?
    row['wt5_beats_p90'] = (
        row.get('block_fpr_wt5', 1) < row.get('block_fpr_pooled', 0)
        and row.get('block_recall_wt5', 0) >= row.get('block_recall_p90', 0) * 0.95
    )

    # Threshold transferability (oracle — diagnostic only)
    thrs = []
    for ds in ['iscx', 'vnat', 'usbvpn']:
        ds_t = t[t['dataset'] == ds]
        dsl = ds_t.groupby('capture_id')['label'].max()
        dss = ds_t.groupby('capture_id')[pc].agg(p90_agg)
        dc = dsl.index.intersection(dss.index)
        if len(dc) > 0 and dsl.loc[dc].nunique() > 1:
            thrs.append(threshold_at_fpr(dsl.loc[dc].values, dss.loc[dc].values, 0.0,
                                         warn_resolution=False))
    row['thr_range'] = max(thrs) - min(thrs) if len(thrs) >= 2 else float('nan')

    return row

# %% [markdown]
# # Run Experiments

# %%
ablation_rows = []

for exp_name, cfg in ABLATIONS.items():
    print(f'\n{"=" * 72}')
    print(f'  EXPERIMENT: {exp_name}')
    print(f'  Features ({len(cfg["features"])}): {cfg["features"]}')
    print(f'{"=" * 72}')

    out_dir = EXPERIMENTS_DIR / cfg['subdir']

    if cfg['train']:
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        # Pipeline transform
        pipe = FeaturePipeline().fit(df_pool[df_pool['split'] == 'train'].copy())
        df_t = pipe.transform(df_pool)
        for col in ['label', 'split', 'capture_id', 'dataset', 'flow_id',
                     'source_file', 'source_capture_id']:
            if col in df_pool.columns:
                df_t[col] = df_pool[col].values

        mcols = [c for c in cfg['features'] if c in df_t.columns]
        missing = [c for c in cfg['features'] if c not in df_t.columns]
        if missing:
            print(f'  WARNING: missing features {missing}, skipping')
            continue

        results = run_balanced_bagging(
            df=df_t, label_col='label', group_col='capture_id',
            dataset_col='dataset', split_col='split',
            bags_per_family=3, majority_ratio=1.0,
            target_fprs='0.0,0.001,0.005,0.01', seed=SEED,
            output_dir=str(out_dir), model_types=['xgb', 'lgbm', 'cat'],
            feature_cols=mcols,
            weight_xgb=1.0, weight_lgbm=1.0, weight_cat=1.0,
            xgb_params=xgb_params, cat_params=cat_params, lgbm_params=lgbm_params,
        )
        elapsed = time.time() - t0
        print(f'  Training completed in {elapsed:.0f}s')
    else:
        print(f'  Loading pre-trained model from {out_dir}')

    # Load predictions
    pred_path = out_dir / 'predictions.csv'
    if not pred_path.exists():
        print(f'  ERROR: predictions.csv not found at {pred_path}')
        continue

    preds = pd.read_csv(pred_path)
    row = full_eval(preds, exp_name, cfg['features'])
    ablation_rows.append(row)
    print(f'  Session AUC (p90): {row.get("session_roc_auc_p90", 0):.4f}')
    print(f'  Block FPR (pooled): {row.get("block_fpr_pooled", 0):.4f}')
    print(f'  Block FPR (ISCX): {row.get("block_fpr_iscx", 0):.4f}')

# %% [markdown]
# # Domain Detector AUC per Ablation

# %%
META = {'flow_id', 'capture_id', 'label', 'dataset', 'split',
        'source_file', 'source_capture_id', 'q_packet_count',
        'q_min_packets_ok', 'app', 'connection_str', 'file_names'}

# Build feature pool for domain detection
vnat_feats = pd.read_parquet(paths.data_processed_dir / 'vnat' / 'features.parquet')
vnat_feats['dataset'] = 'vnat'
iscx_feats = pd.read_parquet(paths.data_processed_dir / 'iscx' / 'features.parquet')
iscx_feats['dataset'] = 'iscx'
usbvpn_feats = pd.read_parquet(paths.data_processed_dir / 'usbvpn' / 'flows.parquet')
usbvpn_feats['dataset'] = 'usbvpn'

v_feat = set(vnat_feats.columns) - META
i_feat = set(iscx_feats.columns) - META
u_feat = set(usbvpn_feats.columns) - META
INTERSECTION = sorted(v_feat & i_feat & u_feat)

keep = sorted(META & set(vnat_feats.columns) & set(iscx_feats.columns)
              & set(usbvpn_feats.columns)) + INTERSECTION

def safe_sel(d, cols):
    return d[[c for c in cols if c in d.columns]].copy()

df_all = pd.concat([safe_sel(vnat_feats, keep), safe_sel(iscx_feats, keep),
                     safe_sel(usbvpn_feats, keep)], ignore_index=True)
if 'q_min_packets_ok' in df_all.columns:
    df_all = df_all[df_all['q_min_packets_ok'].fillna(1) == 1].copy()
df_all['split'] = df_all['split'].astype(str)
df_all['dataset'] = df_all['dataset'].astype(str)
for c in INTERSECTION:
    df_all[c] = pd.to_numeric(df_all[c], errors='coerce').fillna(0.0).astype(float)

df_train_dom = df_all[df_all['split'] == 'train']
df_val_dom = df_all[df_all['split'] == 'val']
le = LabelEncoder()
le.fit(df_all['dataset'])
y_tr_d = le.transform(df_train_dom['dataset'])
y_va_d = le.transform(df_val_dom['dataset'])

print('\n=== Domain Detector AUC per Feature Subset ===')
for i, row in enumerate(ablation_rows):
    exp_name = row['experiment']
    feat_list = ABLATIONS[exp_name]['features']
    avail = [f for f in feat_list if f in df_train_dom.columns]
    if len(avail) < 2:
        row['domain_det_auc'] = float('nan')
        continue
    try:
        _, dd_auc = train_dataset_detector(
            df_train_dom[avail].values, y_tr_d,
            df_val_dom[avail].values, y_va_d)
        row['domain_det_auc'] = float(dd_auc)
    except Exception as e:
        print(f'  {exp_name}: domain detector failed: {e}')
        row['domain_det_auc'] = float('nan')
    print(f'  {exp_name}: domain_det_auc = {row["domain_det_auc"]:.4f}')

# %% [markdown]
# # Save Ablation Results

# %%
ablation_df = pd.DataFrame(ablation_rows)

# Add deltas vs 5f baseline
baseline = ablation_df[ablation_df['experiment'] == '5f_baseline'].iloc[0]
for col in ['session_roc_auc_p90', 'block_fpr_pooled', 'block_fpr_iscx',
            'domain_det_auc', 'block_recall_p90']:
    if col in ablation_df.columns:
        ablation_df[f'{col}_delta'] = ablation_df[col] - baseline[col]

print('\n=== ABLATION RESULTS ===')
display_cols = [
    'experiment', 'n_features',
    'flow_roc_auc', 'session_roc_auc_p90', 'block_recall_p90',
    'block_fpr_pooled', 'block_fpr_iscx', 'block_fpr_vnat', 'block_fpr_usbvpn',
    'precision_p90', 'ece', 'brier', 'train_test_gap',
    'domain_det_auc', 'thr_range',
    'session_roc_auc_wt5', 'block_recall_wt5', 'block_fpr_wt5', 'wt5_beats_p90',
]
avail = [c for c in display_cols if c in ablation_df.columns]
_disp = ablation_df[avail].copy()
_num = _disp.select_dtypes('number').columns
_disp[_num] = _disp[_num].round(4)
print(_disp.to_string(index=False))

ablation_df.to_csv(EVAL_DIR / 'ablation_results.csv', index=False)
print(f'\nSaved: {EVAL_DIR / "ablation_results.csv"}')

# %% [markdown]
# # Deployment Policy Ranking

# %%
policy_rows = []

for exp_name, cfg in ABLATIONS.items():
    pred_path = EXPERIMENTS_DIR / cfg['subdir'] / 'predictions.csv'
    if not pred_path.exists():
        continue
    preds = pd.read_csv(pred_path)
    val_df = preds[preds['split'] == 'val']
    test_df = preds[preds['split'] == 'test']
    pc = 'prob_iso'

    for agg_name, agg_fn in [('p90', p90_agg), ('wt5', weighted_top5)]:
        for target_fpr in [0.0, 0.01]:
            # Val threshold
            vl = val_df.groupby('capture_id')['label'].max()
            vs = val_df.groupby('capture_id')[pc].agg(agg_fn)
            vc = vl.index.intersection(vs.index)
            if len(vc) == 0 or vl.loc[vc].nunique() < 2:
                continue
            thr = threshold_at_fpr(vl.loc[vc].values, vs.loc[vc].values,
                                   target_fpr, warn_resolution=False)

            # Test metrics
            tl = test_df.groupby('capture_id')['label'].max()
            ts = test_df.groupby('capture_id')[pc].agg(agg_fn)
            tc = tl.index.intersection(ts.index)
            y_t = tl.loc[tc].values
            s_t = ts.loc[tc].values
            cm = confusion_at_threshold(y_t, s_t, thr)

            prow = {
                'experiment': exp_name,
                'aggregation': agg_name,
                'val_target_fpr': target_fpr,
                'threshold': thr,
                'block_recall': cm['recall'],
                'block_fpr': cm['fpr'],
                'precision': cm['precision'],
            }

            # Per-dataset
            for ds in ['iscx', 'vnat', 'usbvpn']:
                ds_sub = test_df[test_df['dataset'] == ds]
                dsl = ds_sub.groupby('capture_id')['label'].max()
                dss = ds_sub.groupby('capture_id')[pc].agg(agg_fn)
                dc = dsl.index.intersection(dss.index)
                if len(dc) > 0:
                    dcm = confusion_at_threshold(dsl.loc[dc].values,
                                                 dss.loc[dc].values, thr)
                    prow[f'fpr_{ds}'] = dcm['fpr']
                    prow[f'recall_{ds}'] = dcm['recall']

            policy_rows.append(prow)

policy_df = pd.DataFrame(policy_rows)

# Rank: lower pooled FPR, lower ISCX FPR, higher recall, higher precision
policy_df = policy_df.sort_values(
    ['block_fpr', 'fpr_iscx', 'block_recall', 'precision'],
    ascending=[True, True, False, False]
)
policy_df['rank'] = range(1, len(policy_df) + 1)

print('\n=== DEPLOYMENT POLICY RANKING ===')
print(policy_df.round(4).to_string(index=False))

policy_df.to_csv(EVAL_DIR / 'deployment_policy_ranking.csv', index=False)
print(f'\nSaved: {EVAL_DIR / "deployment_policy_ranking.csv"}')

# %% [markdown]
# # Feature Distribution by Dataset

# %%
dist_rows = []
for feat in FEATS_5F:
    if feat not in df_all.columns:
        continue
    pooled_vals = df_all[df_all['split'] == 'train'][feat].dropna().values
    for ds in ['iscx', 'vnat', 'usbvpn']:
        for sp in ['train', 'val', 'test']:
            subset = df_all[(df_all['dataset'] == ds) & (df_all['split'] == sp)]
            vals = subset[feat].dropna().values
            if len(vals) == 0:
                continue
            ks_stat, ks_p = ks_2samp(vals, pooled_vals) if len(pooled_vals) > 0 else (0, 1)
            dist_rows.append({
                'feature': feat, 'dataset': ds, 'split': sp,
                'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                'median': float(np.median(vals)),
                'p5': float(np.percentile(vals, 5)),
                'p25': float(np.percentile(vals, 25)),
                'p75': float(np.percentile(vals, 75)),
                'p95': float(np.percentile(vals, 95)),
                'ks_stat_vs_pooled': float(ks_stat),
                'ks_pvalue_vs_pooled': float(ks_p),
            })

dist_df = pd.DataFrame(dist_rows)
dist_df.to_csv(EVAL_DIR / 'feature_distribution_by_dataset.csv', index=False)
print(f'\nSaved: {EVAL_DIR / "feature_distribution_by_dataset.csv"}')

# Show highest KS-stat features per dataset (train split)
print('\n=== Top Domain-Discriminative Features (KS stat vs pooled, train) ===')
train_dist = dist_df[dist_df['split'] == 'train'].copy()
for ds in ['iscx', 'vnat', 'usbvpn']:
    ds_d = train_dist[train_dist['dataset'] == ds].sort_values('ks_stat_vs_pooled',
                                                                 ascending=False)
    print(f'\n  {ds.upper()}:')
    for _, r in ds_d.head(5).iterrows():
        print(f'    {r["feature"]:30s}  KS={r["ks_stat_vs_pooled"]:.4f}  '
              f'p={r["ks_pvalue_vs_pooled"]:.2e}')

# %% [markdown]
# # Threshold Resolution Report

# %%
res_rows = []
for exp_name, cfg in ABLATIONS.items():
    pred_path = EXPERIMENTS_DIR / cfg['subdir'] / 'predictions.csv'
    if not pred_path.exists():
        continue
    preds = pd.read_csv(pred_path)
    val_df = preds[preds['split'] == 'val']
    pc = 'prob_iso'

    for agg_name, agg_fn in [('p90', p90_agg), ('wt5', weighted_top5)]:
        vl = val_df.groupby('capture_id')['label'].max()
        vs = val_df.groupby('capture_id')[pc].agg(agg_fn)
        vc = vl.index.intersection(vs.index)
        if len(vc) == 0:
            continue
        y_v = vl.loc[vc].values
        s_v = vs.loc[vc].values

        for target_fpr in [0.0, 0.01, 0.02, 0.05]:
            thr, meta = threshold_at_fpr_robust(y_v, s_v, target_fpr)
            res_rows.append({
                'experiment': exp_name,
                'split': 'val',
                'aggregation': agg_name,
                'target_fpr': target_fpr,
                'achieved_threshold': thr,
                **meta,
            })

res_df = pd.DataFrame(res_rows)
res_df.to_csv(EVAL_DIR / 'threshold_resolution_report.csv', index=False)
print(f'\nSaved: {EVAL_DIR / "threshold_resolution_report.csv"}')

# %% [markdown]
# # Domain Leakage Comparison

# %%
leakage_rows = []
for row in ablation_rows:
    exp_name = row['experiment']
    leakage_rows.append({
        'feature_set': exp_name,
        'n_features': row['n_features'],
        'domain_det_auc': row.get('domain_det_auc', float('nan')),
        'domain_det_auc_delta_vs_5f': row.get('domain_det_auc', 0) - baseline.get('domain_det_auc', 0),
        'iscx_fpr': row.get('block_fpr_iscx', float('nan')),
        'iscx_fpr_delta_vs_5f': row.get('block_fpr_iscx', 0) - baseline.get('block_fpr_iscx', 0),
        'session_auc': row.get('session_roc_auc_p90', float('nan')),
        'session_auc_delta_vs_5f': row.get('session_roc_auc_p90', 0) - baseline.get('session_roc_auc_p90', 0),
    })

leakage_df = pd.DataFrame(leakage_rows)
leakage_df.to_csv(EVAL_DIR / 'domain_leakage_comparison.csv', index=False)
print(f'\nSaved: {EVAL_DIR / "domain_leakage_comparison.csv"}')
print(leakage_df.round(4).to_string(index=False))

# %% [markdown]
# # Final Summary

# %%
print('\n' + '=' * 80)
print('  LEAKAGE ABLATION COMPLETE')
print('=' * 80)
print(f'\nExperiments evaluated: {len(ablation_rows)}')
print(f'\nOutput files:')
for f in sorted(EVAL_DIR.glob('*.csv')):
    print(f'  {f.name}')

# Quick pass/fail assessment
for row in ablation_rows:
    name = row['experiment']
    sa = row.get('session_roc_auc_p90', 0)
    dd = row.get('domain_det_auc', 1)
    ifpr = row.get('block_fpr_iscx', 1)
    br = row.get('block_recall_p90', 0)

    flags = []
    if sa >= 0.95 and dd < 0.96 and ifpr < 0.04:
        flags.append('STRONG IMPROVEMENT')
    elif sa >= 0.95 and (dd < baseline.get('domain_det_auc', 1) - 0.01):
        flags.append('ACCEPTABLE')
    elif sa < 0.93:
        flags.append('NOT WORTH IT: detector quality degraded')
    elif dd >= baseline.get('domain_det_auc', 0) - 0.005:
        flags.append('NOT WORTH IT: no domain improvement')
    else:
        flags.append('MARGINAL')

    print(f'\n  {name}:')
    print(f'    Session AUC: {sa:.4f}  Domain AUC: {dd:.4f}  ISCX FPR: {ifpr:.4f}  '
          f'Recall: {br:.4f}')
    print(f'    Assessment: {", ".join(flags)}')


