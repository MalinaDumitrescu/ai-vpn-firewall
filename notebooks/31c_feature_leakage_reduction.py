# %% [markdown]
# # 31c — Feature Leakage Reduction & Deployment-Optimal Ablation
#
# ## Purpose
#
# The 3DS primary model (5 compact non-directional features) has strong
# classifier quality (session AUC = 0.9879) but unstable deployment
# thresholds.  The root causes:
#
# 1. **Domain leakage**: domain detector AUC = 0.9769.
#    `sz_p25_median_ratio` (0.9157) and `sz_p75_median_ratio` (0.9073)
#    are the strongest domain fingerprints.
# 2. **ISCX FPR explosion**: Block FPR = 0.4706 under global threshold.
# 3. **Threshold transferability range**: 0.5367 across datasets.
#
# ## Strategy
#
# Run feature-subset ablations, retrain the ensemble for each, and
# compare deployment-relevant metrics.  Focus on improving threshold
# stability and lowering domain sensitivity, even at modest AUC cost.
#
# ## Feature Subsets
#
# | Tag | Features | Rationale |
# |-----|----------|-----------|
# | 5f  | all 5 no-dir | Baseline |
# | 4f-drop-p25 | drop sz_p25_median_ratio | Highest domain AUC feature |
# | 4f-drop-p75 | drop sz_p75_median_ratio | Second-highest domain AUC |
# | 3f-core | sz_coef_variation, sz_iqr_norm_median, dispersion_symmetry | Minimal core |
# | 4f-backup | (existing 4f reduced: drop p25) | Existing backup for comparison |
#
# ## Aggregation × Calibration Sweep
#
# For every feature subset, test:
# - p90 + isotonic (current primary)
# - weighted_top5_mean + isotonic (serious deployment candidate)
# - mean + isotonic
# - p90 + raw
# - p90 + platt

# %%
# ── Cell 0: Imports & Config ─────────────────────────────────────────────

import sys, os, json, warnings, time
warnings.filterwarnings('ignore')

import io as _io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')

_root = os.path.abspath(os.path.join(os.getcwd(), '..')) \
    if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder

from src.utils.paths import load_paths
from src.utils.logging import setup_logger
from src.pipeline.feature_pipeline import COMPACT_FEATURES, DIRECTION_FEATURES
from src.eval.metrics import threshold_at_fpr, confusion_at_threshold
from src.optimization.dataset_adversarial_feature_selection import train_dataset_detector
from src.models.train_balanced_bagging_ensemble import run_balanced_bagging
from src.features.extract import load_feature_config

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

paths = load_paths()
logger = setup_logger(level='INFO')
SEED = 42

EXPERIMENTS_DIR = paths.artifacts_dir / 'experiments'
OUTPUT_DIR = EXPERIMENTS_DIR / 'feature_leakage_reduction'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Project root: {_root}')
print(f'Output: {OUTPUT_DIR}')

# ── Feature subsets ──
FEATS_5F = [c for c in COMPACT_FEATURES if c not in DIRECTION_FEATURES]
FEATS_4F_DROP_P25 = [f for f in FEATS_5F if f != 'sz_p25_median_ratio']
FEATS_4F_DROP_P75 = [f for f in FEATS_5F if f != 'sz_p75_median_ratio']
FEATS_3F_CORE = ['sz_coef_variation', 'sz_iqr_norm_median', 'dispersion_symmetry']
# Also try dropping both percentile features
FEATS_3F_CORE_V2 = ['sz_coef_variation', 'sz_p75_median_ratio', 'dispersion_symmetry']

ABLATIONS = [
    ('5f-baseline',     FEATS_5F),
    ('4f-drop-p25',     FEATS_4F_DROP_P25),
    ('4f-drop-p75',     FEATS_4F_DROP_P75),
    ('3f-core',         FEATS_3F_CORE),
    ('3f-core-v2',      FEATS_3F_CORE_V2),
]

print(f'\nAblation subsets:')
for tag, feats in ABLATIONS:
    print(f'  {tag:18s}: {feats}')

# %% [markdown]
# ## Section 1 — Load & Prepare 3-Dataset Pool
#
# Strict feature intersection (no zero-filling of structurally missing cols).

# %%
# ── Cell 1: Load datasets ───────────────────────────────────────────────

cfg = load_feature_config(paths.configs_dir / 'features.yaml')

META = {'flow_id', 'capture_id', 'label', 'dataset', 'split',
        'source_file', 'source_capture_id', 'q_packet_count',
        'q_min_packets_ok', 'app', 'connection_str', 'file_names'}

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
print(f'Strict feature intersection: {len(INTERSECTION)} features')

keep = sorted(META & set(vnat_feats.columns) & set(iscx_feats.columns)
              & set(usbvpn_feats.columns)) + INTERSECTION

def safe_sel(d, cols):
    return d[[c for c in cols if c in d.columns]].copy()

df_all = pd.concat([safe_sel(vnat_feats, keep), safe_sel(iscx_feats, keep),
                     safe_sel(usbvpn_feats, keep)], ignore_index=True)
for id_col in ['source_file', 'source_capture_id']:
    if id_col not in df_all.columns:
        df_all[id_col] = ''
if 'q_min_packets_ok' in df_all.columns:
    df_all = df_all[df_all['q_min_packets_ok'].fillna(1) == 1].copy()
df_all[INTERSECTION] = df_all[INTERSECTION].fillna(0.0)
df_all['split'] = df_all['split'].astype(str)
df_all['dataset'] = df_all['dataset'].astype(str)
df_all['label'] = df_all['label'].astype(int)
for c in INTERSECTION:
    df_all[c] = pd.to_numeric(df_all[c], errors='coerce').fillna(0.0).astype(float)

print(f'Clean pool: {df_all.shape}')
print(pd.crosstab(df_all['split'], df_all['dataset']))

# ── Balanced training pool ──
def balance_training_pool(df, seed=SEED):
    """Undersample majority per-dataset in train split."""
    tr = df[df['split'] == 'train'].copy()
    rest = df[df['split'] != 'train'].copy()
    balanced = []
    for ds in tr['dataset'].unique():
        ds_df = tr[tr['dataset'] == ds]
        minority = ds_df[ds_df['label'] == 1]
        majority = ds_df[ds_df['label'] == 0]
        n = min(len(majority), max(len(minority) * 3, 500))
        maj_sample = majority.sample(n=n, random_state=seed) if len(majority) > n else majority
        balanced.append(pd.concat([minority, maj_sample]))
    return pd.concat([pd.concat(balanced)] + [rest], ignore_index=True)

df_balanced = balance_training_pool(df_all)
print(f'\nBalanced pool: {df_balanced.shape}')
print(pd.crosstab(df_balanced[df_balanced["split"]=="train"]["dataset"],
                   df_balanced[df_balanced["split"]=="train"]["label"]))

# %% [markdown]
# ## Section 2 — Hyperparameters

# %%
# ── Cell 2: Load hyperparameters ─────────────────────────────────────────

with open(paths.artifacts_dir / 'optuna_xgboost_firewall_best_params.json') as f:
    xgb_params = json.load(f)
xgb_params.update({'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'booster': 'gbtree', 'tree_method': 'hist',
    'n_estimators': 1000, 'random_state': SEED, 'n_jobs': 1,
    'early_stopping_rounds': 50})

with open(paths.artifacts_dir / 'optuna_catboost_firewall_best_params.json') as f:
    cat_params = json.load(f)
cat_params.update({'iterations': 1000, 'random_seed': SEED, 'thread_count': 1,
    'verbose': False, 'allow_writing_files': False, 'early_stopping_rounds': 150})

with open(paths.artifacts_dir / 'optuna_lgbm_firewall_best_params.json') as f:
    lgbm_params = json.load(f)
lgbm_params.update({'objective': 'binary', 'metric': 'binary_logloss',
    'boosting_type': 'gbdt', 'n_estimators': 1000, 'verbose': -1,
    'random_state': SEED, 'n_jobs': 1})

print('Hyperparameters loaded (Optuna-tuned).')

# %% [markdown]
# ## Section 3 — Aggregation & Evaluation Helpers

# %%
# ── Cell 3: Evaluation helpers ───────────────────────────────────────────

def p90_agg(x):
    return np.percentile(x, 90)

def weighted_top5(x):
    vals = np.sort(x)[::-1][:5]
    w = np.array([0.40, 0.25, 0.15, 0.10, 0.10])[:len(vals)]
    w = w / w.sum()
    return float(np.sum(vals * w))

def mean_agg(x):
    return np.mean(x)

AGG_RULES = {
    'p90': p90_agg,
    'wt5': weighted_top5,
    'mean': mean_agg,
}


def evaluate_predictions(pred_path, feature_tag, feature_list):
    """Comprehensive evaluation of a trained ensemble's predictions.

    Returns a list of result dicts — one per (aggregation, calibration) combo.
    All thresholds are val-derived.  No oracle/test-derived thresholds.
    """
    preds = pd.read_csv(pred_path)
    train_df = preds[preds['split'] == 'train']
    val_df   = preds[preds['split'] == 'val']
    test_df  = preds[preds['split'] == 'test']

    prob_cols = {'isotonic': 'prob_iso', 'raw': 'prob_raw', 'platt': 'prob_platt'}
    results = []

    for agg_name, agg_fn in AGG_RULES.items():
        for calib_name, pc in prob_cols.items():
            if pc not in preds.columns:
                continue

            row = {
                'features': feature_tag,
                'n_features': len(feature_list),
                'aggregation': agg_name,
                'calibration': calib_name,
            }

            # ── Flow-level (threshold-independent) ──
            y_test = test_df['label'].values
            p_test = test_df[pc].values
            row['flow_auc'] = float(roc_auc_score(y_test, p_test))
            row['flow_pr_auc'] = float(average_precision_score(y_test, p_test))

            # Train AUC for overfitting check
            if len(train_df) > 0 and train_df['label'].nunique() > 1:
                row['train_flow_auc'] = float(roc_auc_score(
                    train_df['label'], train_df[pc]))
                row['train_test_gap'] = row['train_flow_auc'] - row['flow_auc']
            else:
                row['train_flow_auc'] = float('nan')
                row['train_test_gap'] = float('nan')

            # ── Session-level (threshold-independent) ──
            # Test sessions
            tl = test_df.groupby('capture_id')['label'].max()
            ts = test_df.groupby('capture_id')[pc].agg(agg_fn)
            tc = tl.index.intersection(ts.index)
            y_s = tl.loc[tc].values
            s_s = ts.loc[tc].values

            if len(np.unique(y_s)) > 1:
                row['session_auc'] = float(roc_auc_score(y_s, s_s))
                row['session_pr_auc'] = float(average_precision_score(y_s, s_s))
            else:
                row['session_auc'] = float('nan')
                row['session_pr_auc'] = float('nan')

            # ── Val-derived threshold (FPR→0) ──
            # IMPORTANT: threshold targets FPR→0 on val. Observed test FPR
            # will generally be nonzero due to domain shift.
            vl = val_df.groupby('capture_id')['label'].max()
            vs = val_df.groupby('capture_id')[pc].agg(agg_fn)
            vc = vl.index.intersection(vs.index)
            y_v = vl.loc[vc].values
            s_v = vs.loc[vc].values

            thr = threshold_at_fpr(y_v, s_v, target_fpr=0.0)
            row['block_threshold'] = float(thr)

            # Pooled test metrics at val-derived threshold
            cm = confusion_at_threshold(y_s, s_s, thr)
            row['block_recall'] = cm['recall']
            row['block_fpr'] = cm['fpr']
            row['block_precision'] = cm['precision']

            # ── Per-dataset test metrics at val-derived threshold ──
            for ds in sorted(test_df['dataset'].unique()):
                ds_t = test_df[test_df['dataset'] == ds]
                dsl = ds_t.groupby('capture_id')['label'].max()
                dss = ds_t.groupby('capture_id')[pc].agg(agg_fn)
                dc = dsl.index.intersection(dss.index)
                dy = dsl.loc[dc].values
                ds_s = dss.loc[dc].values

                if len(np.unique(dy)) > 1:
                    row[f'{ds}_session_auc'] = float(roc_auc_score(dy, ds_s))
                else:
                    row[f'{ds}_session_auc'] = float('nan')

                dcm = confusion_at_threshold(dy, ds_s, thr)
                row[f'{ds}_block_recall'] = dcm['recall']
                row[f'{ds}_block_fpr'] = dcm['fpr']

            # ── Threshold transferability ──
            oracle_thrs = []
            for ds in sorted(test_df['dataset'].unique()):
                ds_t = test_df[test_df['dataset'] == ds]
                dsl = ds_t.groupby('capture_id')['label'].max()
                dss = ds_t.groupby('capture_id')[pc].agg(agg_fn)
                dc = dsl.index.intersection(dss.index)
                dy = dsl.loc[dc].values
                ds_s = dss.loc[dc].values
                if len(np.unique(dy)) > 1:
                    oracle_thrs.append(threshold_at_fpr(dy, ds_s, 0.0))
            row['thr_range'] = (max(oracle_thrs) - min(oracle_thrs)
                                if oracle_thrs else float('nan'))

            results.append(row)

    return results


print('Evaluation helpers defined.')

# %% [markdown]
# ## Section 4 — Train Ablation Ensembles
#
# For each feature subset, train a fresh balanced-bagging ensemble.
# Skip subsets that already have artifacts (set FORCE_RETRAIN=True to override).

# %%
# ── Cell 4: Train ablations ──────────────────────────────────────────────

FORCE_RETRAIN = False
all_results = []

for tag, feats in ABLATIONS:
    exp_dir = OUTPUT_DIR / f'exp_{tag}'
    pred_path = exp_dir / 'predictions.csv'

    # Check if features exist in pool
    missing = [f for f in feats if f not in df_balanced.columns]
    if missing:
        print(f'\n[SKIP] {tag}: missing features {missing}')
        continue

    if pred_path.exists() and not FORCE_RETRAIN:
        print(f'\n[CACHED] {tag}: predictions exist at {pred_path}')
    else:
        print(f'\n{"="*72}')
        print(f'  TRAINING: {tag}')
        print(f'  Features ({len(feats)}): {feats}')
        print(f'{"="*72}')
        t0 = time.time()

        run_balanced_bagging(
            df=df_balanced,
            label_col='label', group_col='capture_id',
            dataset_col='dataset', split_col='split',
            bags_per_family=3, majority_ratio=1.0,
            target_fprs='0.0,0.001,0.005,0.01',
            seed=SEED,
            output_dir=str(exp_dir),
            model_types=['xgb', 'lgbm', 'cat'],
            feature_cols=feats,
            weight_xgb=1.0, weight_lgbm=1.0, weight_cat=1.0,
            xgb_params=xgb_params, cat_params=cat_params, lgbm_params=lgbm_params,
        )
        elapsed = time.time() - t0
        print(f'  {tag}: trained in {elapsed:.0f}s')

    # Evaluate
    if pred_path.exists():
        tag_results = evaluate_predictions(pred_path, tag, feats)
        all_results.extend(tag_results)
        n = len(tag_results)
        print(f'  {tag}: {n} (agg × calib) evaluations')
    else:
        print(f'  WARNING: {tag} predictions not found after training')

print(f'\nTotal evaluations: {len(all_results)}')

# %% [markdown]
# ## Section 5 — Domain Detector for Each Subset

# %%
# ── Cell 5: Domain detection per feature subset ──────────────────────────

df_train = df_all[df_all['split'] == 'train']
df_val_d = df_all[df_all['split'] == 'val']
le = LabelEncoder()
le.fit(df_all['dataset'])
y_tr_d = le.transform(df_train['dataset'])
y_va_d = le.transform(df_val_d['dataset'])

domain_auc_map = {}
print('=== Domain Detector AUC per Feature Subset ===')
for tag, feats in ABLATIONS:
    avail = [f for f in feats if f in df_train.columns]
    if len(avail) < 2:
        domain_auc_map[tag] = float('nan')
        continue
    _, dd_auc = train_dataset_detector(
        df_train[avail].values, y_tr_d,
        df_val_d[avail].values, y_va_d)
    domain_auc_map[tag] = float(dd_auc)
    print(f'  {tag:18s}: domain AUC = {dd_auc:.4f}')

# Inject domain AUC into results
for r in all_results:
    r['domain_auc'] = domain_auc_map.get(r['features'], float('nan'))

# Per-feature solo domain AUC
print('\n--- Per-Feature Solo Domain AUC ---')
for feat in FEATS_5F:
    if feat not in df_train.columns:
        continue
    _, solo_auc = train_dataset_detector(
        df_train[[feat]].values, y_tr_d,
        df_val_d[[feat]].values, y_va_d)
    print(f'  {feat:30s}  {solo_auc:.4f}')

# %% [markdown]
# ## Section 6 — Deployment-Focused Ranking
#
# Rank all (feature_subset × aggregation × calibration) combos by
# deployment usefulness, using these priorities:
#
# 1. Lower pooled block FPR
# 2. Lower ISCX block FPR
# 3. Higher VPN block recall
# 4. Higher session ROC-AUC
# 5. Lower domain detector AUC

# %%
# ── Cell 6: Build ranked comparison table ────────────────────────────────

results_df = pd.DataFrame(all_results)

def deployment_score(row):
    """Composite score: higher is better for deployment."""
    # Penalize FPR heavily
    fpr_penalty = (1.0 - row.get('block_fpr', 1)) * 25
    iscx_fpr_penalty = (1.0 - row.get('iscx_block_fpr', 1)) * 20
    # Reward recall
    recall_reward = row.get('block_recall', 0) * 15
    # Reward session AUC
    auc_reward = row.get('session_auc', 0) * 10
    # Penalize domain leakage
    domain_penalty = (1.0 - row.get('domain_auc', 1)) * 10
    # Penalize threshold instability
    thr_stability = max(0, 1.0 - row.get('thr_range', 1)) * 5
    return fpr_penalty + iscx_fpr_penalty + recall_reward + auc_reward + domain_penalty + thr_stability

results_df['deploy_score'] = results_df.apply(deployment_score, axis=1)
ranked = results_df.sort_values('deploy_score', ascending=False).reset_index(drop=True)

# Display columns for the comparison table (Task I)
display_cols = [
    'features', 'aggregation', 'calibration',
    'session_auc', 'session_pr_auc',
    'block_recall', 'block_fpr',
    'iscx_block_fpr', 'usbvpn_block_recall', 'vnat_block_recall',
    'domain_auc', 'thr_range', 'deploy_score',
]
avail = [c for c in display_cols if c in ranked.columns]

print('=== Full Comparison Table (ranked by deployment score) ===')
_disp = ranked[avail].copy()
_num = _disp.select_dtypes('number').columns
_disp[_num] = _disp[_num].round(4)
print(_disp.to_string(index=False))

ranked.to_csv(OUTPUT_DIR / 'full_comparison.csv', index=False)

# %% [markdown]
# ## Section 7 — Top Candidates Analysis

# %%
# ── Cell 7: Top candidates ───────────────────────────────────────────────

print('=== Top 10 Deployment Candidates ===')
top10 = ranked.head(10)
top_cols = ['features', 'aggregation', 'calibration',
            'session_auc', 'block_recall', 'block_fpr',
            'iscx_block_fpr', 'usbvpn_block_recall', 'vnat_block_recall',
            'domain_auc', 'deploy_score']
top_avail = [c for c in top_cols if c in top10.columns]
_t = top10[top_avail].copy()
_tn = _t.select_dtypes('number').columns
_t[_tn] = _t[_tn].round(4)
print(_t.to_string(index=False))

# Best for detector quality (highest session AUC)
best_detector = ranked.loc[ranked['session_auc'].idxmax()]
print(f'\n--- Best for Detector Quality ---')
print(f'  Config: {best_detector["features"]} / {best_detector["aggregation"]} / {best_detector["calibration"]}')
print(f'  Session AUC: {best_detector["session_auc"]:.4f}')
print(f'  Block FPR: {best_detector["block_fpr"]:.4f}')
print(f'  ISCX FPR: {best_detector.get("iscx_block_fpr", float("nan")):.4f}')

# Best for deployment (highest deploy_score)
best_deploy = ranked.iloc[0]
print(f'\n--- Best for Deployment Policy ---')
print(f'  Config: {best_deploy["features"]} / {best_deploy["aggregation"]} / {best_deploy["calibration"]}')
print(f'  Session AUC: {best_deploy["session_auc"]:.4f}')
print(f'  Block Recall: {best_deploy["block_recall"]:.4f}')
print(f'  Block FPR: {best_deploy["block_fpr"]:.4f}')
print(f'  ISCX FPR: {best_deploy.get("iscx_block_fpr", float("nan")):.4f}')
print(f'  Domain AUC: {best_deploy["domain_auc"]:.4f}')
print(f'  Deploy Score: {best_deploy["deploy_score"]:.2f}')

# %%
# ── Cell 7b: Comparison chart ────────────────────────────────────────────

# Filter to isotonic calibration for cleaner visualization
iso_results = ranked[ranked['calibration'] == 'isotonic'].copy()

if len(iso_results) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Plot 1: Session AUC by feature subset and aggregation
    ax = axes[0, 0]
    for agg in ['p90', 'wt5', 'mean']:
        subset = iso_results[iso_results['aggregation'] == agg]
        if len(subset) > 0:
            ax.barh([f'{r["features"]}' for _, r in subset.iterrows()],
                    subset['session_auc'].values, alpha=0.7, label=agg)
    ax.set_xlabel('Session AUC')
    ax.set_title('Session AUC (Isotonic)')
    ax.legend()

    # Plot 2: Pooled Block FPR
    ax = axes[0, 1]
    for agg in ['p90', 'wt5', 'mean']:
        subset = iso_results[iso_results['aggregation'] == agg]
        if len(subset) > 0:
            ax.barh([f'{r["features"]}' for _, r in subset.iterrows()],
                    subset['block_fpr'].values, alpha=0.7, label=agg)
    ax.set_xlabel('Block FPR (pooled test)')
    ax.set_title('Pooled Block FPR (lower=better)')
    ax.legend()

    # Plot 3: ISCX Block FPR
    ax = axes[0, 2]
    if 'iscx_block_fpr' in iso_results.columns:
        for agg in ['p90', 'wt5', 'mean']:
            subset = iso_results[iso_results['aggregation'] == agg]
            if len(subset) > 0:
                vals = subset['iscx_block_fpr'].fillna(0).values
                ax.barh([f'{r["features"]}' for _, r in subset.iterrows()],
                        vals, alpha=0.7, label=agg)
    ax.set_xlabel('ISCX Block FPR')
    ax.set_title('ISCX FPR (lower=better)')
    ax.legend()

    # Plot 4: Block Recall
    ax = axes[1, 0]
    for agg in ['p90', 'wt5', 'mean']:
        subset = iso_results[iso_results['aggregation'] == agg]
        if len(subset) > 0:
            ax.barh([f'{r["features"]}' for _, r in subset.iterrows()],
                    subset['block_recall'].values, alpha=0.7, label=agg)
    ax.set_xlabel('Block Recall')
    ax.set_title('Block Recall (higher=better)')
    ax.legend()

    # Plot 5: Domain AUC
    ax = axes[1, 1]
    for agg in ['p90', 'wt5', 'mean']:
        subset = iso_results[iso_results['aggregation'] == agg]
        if len(subset) > 0:
            ax.barh([f'{r["features"]}' for _, r in subset.iterrows()],
                    subset['domain_auc'].values, alpha=0.7, label=agg)
    ax.set_xlabel('Domain Detector AUC')
    ax.set_title('Domain AUC (lower=better)')
    ax.legend()

    # Plot 6: Deploy Score
    ax = axes[1, 2]
    for agg in ['p90', 'wt5', 'mean']:
        subset = iso_results[iso_results['aggregation'] == agg]
        if len(subset) > 0:
            ax.barh([f'{r["features"]}' for _, r in subset.iterrows()],
                    subset['deploy_score'].values, alpha=0.7, label=agg)
    ax.set_xlabel('Deployment Score')
    ax.set_title('Deployment Score (higher=better)')
    ax.legend()

    plt.suptitle('Feature Ablation: Deployment Metrics (Isotonic calibration)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ablation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## Section 8 — Does Dropping p25/p75 Help?

# %%
# ── Cell 8: p25/p75 ablation analysis ────────────────────────────────────

print('=== Impact of Dropping sz_p25_median_ratio / sz_p75_median_ratio ===\n')

# Compare matched configs (same agg + calibration)
baseline_tag = '5f-baseline'

for compare_tag in ['4f-drop-p25', '4f-drop-p75', '3f-core']:
    print(f'--- {baseline_tag} vs {compare_tag} ---')
    for agg in ['p90', 'wt5']:
        for calib in ['isotonic']:
            bl = ranked[(ranked['features'] == baseline_tag) &
                        (ranked['aggregation'] == agg) &
                        (ranked['calibration'] == calib)]
            cp = ranked[(ranked['features'] == compare_tag) &
                        (ranked['aggregation'] == agg) &
                        (ranked['calibration'] == calib)]
            if len(bl) == 0 or len(cp) == 0:
                continue
            bl = bl.iloc[0]
            cp = cp.iloc[0]

            delta_auc = cp['session_auc'] - bl['session_auc']
            delta_fpr = cp['block_fpr'] - bl['block_fpr']
            delta_iscx = cp.get('iscx_block_fpr', 0) - bl.get('iscx_block_fpr', 0)
            delta_recall = cp['block_recall'] - bl['block_recall']
            delta_domain = cp['domain_auc'] - bl['domain_auc']

            print(f'  [{agg}/{calib}]')
            print(f'    Session AUC:  {bl["session_auc"]:.4f} → {cp["session_auc"]:.4f}  ({delta_auc:+.4f})')
            print(f'    Block FPR:    {bl["block_fpr"]:.4f} → {cp["block_fpr"]:.4f}  ({delta_fpr:+.4f})')
            print(f'    ISCX FPR:     {bl.get("iscx_block_fpr",0):.4f} → {cp.get("iscx_block_fpr",0):.4f}  ({delta_iscx:+.4f})')
            print(f'    Block Recall: {bl["block_recall"]:.4f} → {cp["block_recall"]:.4f}  ({delta_recall:+.4f})')
            print(f'    Domain AUC:   {bl["domain_auc"]:.4f} → {cp["domain_auc"]:.4f}  ({delta_domain:+.4f})')
    print()

# %% [markdown]
# ## Section 9 — wt5 as Serious Deployment Candidate

# %%
# ── Cell 9: wt5 vs p90 head-to-head ─────────────────────────────────────

print('=== weighted_top5_mean vs p90: Head-to-Head (Isotonic) ===\n')

for tag, _ in ABLATIONS:
    p90_row = ranked[(ranked['features'] == tag) &
                     (ranked['aggregation'] == 'p90') &
                     (ranked['calibration'] == 'isotonic')]
    wt5_row = ranked[(ranked['features'] == tag) &
                     (ranked['aggregation'] == 'wt5') &
                     (ranked['calibration'] == 'isotonic')]
    if len(p90_row) == 0 or len(wt5_row) == 0:
        continue
    p = p90_row.iloc[0]
    w = wt5_row.iloc[0]

    better_fpr = '✓ wt5' if w['block_fpr'] < p['block_fpr'] else '✓ p90'
    better_recall = '✓ wt5' if w['block_recall'] > p['block_recall'] else '✓ p90'
    better_iscx = ('✓ wt5' if w.get('iscx_block_fpr', 1) < p.get('iscx_block_fpr', 1)
                   else '✓ p90')

    print(f'{tag}:')
    print(f'  Session AUC:  p90={p["session_auc"]:.4f}  wt5={w["session_auc"]:.4f}')
    print(f'  Block FPR:    p90={p["block_fpr"]:.4f}  wt5={w["block_fpr"]:.4f}  {better_fpr}')
    print(f'  Block Recall: p90={p["block_recall"]:.4f}  wt5={w["block_recall"]:.4f}  {better_recall}')
    print(f'  ISCX FPR:     p90={p.get("iscx_block_fpr",0):.4f}  wt5={w.get("iscx_block_fpr",0):.4f}  {better_iscx}')
    print(f'  Deploy Score: p90={p["deploy_score"]:.2f}  wt5={w["deploy_score"]:.2f}')
    print()

# %% [markdown]
# ## Section 10 — Final Verdict & Recommendations

# %%
# ── Cell 10: Final recommendations ───────────────────────────────────────

print('=' * 80)
print('  FEATURE LEAKAGE REDUCTION — FINAL RECOMMENDATIONS')
print('=' * 80)

# Best detector quality
best_det_idx = ranked['session_auc'].idxmax()
bd = ranked.loc[best_det_idx]
print(f'''
Q1: Which feature subset is best for DETECTOR QUALITY?
  → {bd["features"]} / {bd["aggregation"]} / {bd["calibration"]}
    Session AUC = {bd["session_auc"]:.4f}
    NOTE: Raw discrimination ability. Does not consider deployment stability.
''')

# Best deployment
best_dep = ranked.iloc[0]
print(f'''Q2: Which feature subset is best for DEPLOYMENT POLICY?
  → {best_dep["features"]} / {best_dep["aggregation"]} / {best_dep["calibration"]}
    Session AUC    = {best_dep["session_auc"]:.4f}
    Block Recall   = {best_dep["block_recall"]:.4f}
    Block FPR      = {best_dep["block_fpr"]:.4f}
    ISCX FPR       = {best_dep.get("iscx_block_fpr", float("nan")):.4f}
    Domain AUC     = {best_dep["domain_auc"]:.4f}
    Thr Range      = {best_dep.get("thr_range", float("nan")):.4f}
    Deploy Score   = {best_dep["deploy_score"]:.2f}
''')

# Q3: Does dropping p25/p75 help domain robustness?
d5f = domain_auc_map.get('5f-baseline', float('nan'))
d4f_p25 = domain_auc_map.get('4f-drop-p25', float('nan'))
d4f_p75 = domain_auc_map.get('4f-drop-p75', float('nan'))
d3f = domain_auc_map.get('3f-core', float('nan'))

print(f'Q3: Does dropping sz_p25_median_ratio / sz_p75_median_ratio improve domain robustness?')
print(f'  5f baseline   domain AUC: {d5f:.4f}')
print(f'  4f drop p25   domain AUC: {d4f_p25:.4f}  (delta = {d4f_p25 - d5f:+.4f})')
print(f'  4f drop p75   domain AUC: {d4f_p75:.4f}  (delta = {d4f_p75 - d5f:+.4f})')
print(f'  3f core       domain AUC: {d3f:.4f}  (delta = {d3f - d5f:+.4f})')

if d4f_p25 < d5f - 0.01:
    print(f'  → YES: Dropping sz_p25_median_ratio reduces domain leakage by {d5f - d4f_p25:.4f}')
else:
    print(f'  → NO: Dropping sz_p25_median_ratio does not materially reduce domain leakage')

if d4f_p75 < d5f - 0.01:
    print(f'  → YES: Dropping sz_p75_median_ratio reduces domain leakage by {d5f - d4f_p75:.4f}')
else:
    print(f'  → NO: Dropping sz_p75_median_ratio does not materially reduce domain leakage')

if d3f < d5f - 0.03:
    print(f'  → YES: 3f-core substantially reduces domain leakage by {d5f - d3f:.4f}')
else:
    print(f'  → MARGINAL: 3f-core offers limited domain improvement ({d5f - d3f:.4f})')
print()

# ── Summary verdict ──
print('=' * 80)
print('  LAYERED VERDICT')
print('=' * 80)

# Find zero-FPR candidates
zero_fpr = ranked[ranked['block_fpr'] == 0.0]
if len(zero_fpr) > 0:
    best_zero = zero_fpr.loc[zero_fpr['block_recall'].idxmax()]
    print(f'''
  ACTUAL ZERO-FPR CANDIDATE (observed test FPR = 0.0):
    Config: {best_zero["features"]} / {best_zero["aggregation"]} / {best_zero["calibration"]}
    Block Recall = {best_zero["block_recall"]:.4f}
    Session AUC  = {best_zero["session_auc"]:.4f}
''')
else:
    print('\n  No configuration achieved actual zero test FPR.')

print(f'''  RECOMMENDED DEPLOYMENT CONFIGURATION:
    {best_dep["features"]} / {best_dep["aggregation"]} / {best_dep["calibration"]}
    This balances detection quality with threshold stability.
''')

# %% [markdown]
# ## Section 11 — Save Outputs

# %%
# ── Cell 11: Save everything ─────────────────────────────────────────────

# Full results
ranked.to_csv(OUTPUT_DIR / 'full_comparison.csv', index=False)
ranked.to_json(OUTPUT_DIR / 'full_comparison.json', orient='records', indent=2)

# Domain AUC map
with open(OUTPUT_DIR / 'domain_auc_by_subset.json', 'w') as f:
    json.dump(domain_auc_map, f, indent=2)

# Top-10
ranked.head(10).to_csv(OUTPUT_DIR / 'top10_deployment.csv', index=False)

# Summary
summary = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'notebook': '31c_feature_leakage_reduction',
    'best_detector_quality': {
        'config': f'{bd["features"]}/{bd["aggregation"]}/{bd["calibration"]}',
        'session_auc': float(bd['session_auc']),
    },
    'best_deployment': {
        'config': f'{best_dep["features"]}/{best_dep["aggregation"]}/{best_dep["calibration"]}',
        'session_auc': float(best_dep['session_auc']),
        'block_recall': float(best_dep['block_recall']),
        'block_fpr': float(best_dep['block_fpr']),
        'iscx_block_fpr': float(best_dep.get('iscx_block_fpr', float('nan'))),
        'domain_auc': float(best_dep['domain_auc']),
        'deploy_score': float(best_dep['deploy_score']),
    },
    'domain_auc_by_subset': domain_auc_map,
    'n_total_configs': len(ranked),
    'ablation_subsets': {tag: feats for tag, feats in ABLATIONS},
}
with open(OUTPUT_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f'All outputs saved to: {OUTPUT_DIR}')
for f in sorted(OUTPUT_DIR.glob('*')):
    if f.is_file():
        print(f'  {f.name}')

# %%
print('\n' + '#' * 80)
print('  31c — FEATURE LEAKAGE REDUCTION COMPLETE')
print('#' * 80)


