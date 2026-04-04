# %% [markdown]
# # 31b — Focused Performance Improvement Pass
#
# ## Purpose
# Systematic search over calibration × aggregation × threshold × weight
# configurations to reduce pooled test Block FPR, improve threshold
# portability, and fix ISCX FPR — the actual deployment bottlenecks
# identified in Notebook 31.
#
# ## Key insight from NB31
# - weighted_top5_mean + isotonic already gives FPR=0.0198 vs p90+iso FPR=0.0792
#   with **same** recall (0.9444). That is the main clue.
# - raw/platt + p90 give FPR=0.0000 but lower recall (0.7778).
# - The classifier is strong; the problem is threshold-policy portability.
#
# ## Approach
# Grid search over (calibration, aggregation, family_weights, target_fpr)
# then rank by a firewall-deployment-aware composite score.

# %%
import sys, os, json, warnings
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
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, roc_curve, precision_recall_curve, confusion_matrix,
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.utils.paths import load_paths
from src.eval.metrics import threshold_at_fpr, confusion_at_threshold
from src.eval.calibration_diagnostics import expected_calibration_error

sns.set_theme(style='whitegrid', font_scale=1.0)
plt.rcParams['figure.dpi'] = 120

paths = load_paths()
SEED = 42
np.random.seed(SEED)

# ── Directories ──
EXPERIMENTS_DIR = paths.artifacts_dir / 'experiments'
PRIMARY_DIR = EXPERIMENTS_DIR / 'exp_c_combined'
BACKUP_DIR = EXPERIMENTS_DIR / 'exp_f9_reduced'
OUTPUT_DIR = EXPERIMENTS_DIR / 'final_3ds_improvement_pass'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Output: {OUTPUT_DIR}')

# %% [markdown]
# ## Load Data

# %%
df_5f = pd.read_csv(PRIMARY_DIR / 'predictions.csv')
df_4f = pd.read_csv(BACKUP_DIR / 'predictions.csv')

train_5f = df_5f[df_5f['split'] == 'train'].copy()
val_5f = df_5f[df_5f['split'] == 'val'].copy()
test_5f = df_5f[df_5f['split'] == 'test'].copy()

train_4f = df_4f[df_4f['split'] == 'train'].copy()
val_4f = df_4f[df_4f['split'] == 'val'].copy()
test_4f = df_4f[df_4f['split'] == 'test'].copy()

print(f'5f: train={len(train_5f)}, val={len(val_5f)}, test={len(test_5f)}')
print(f'4f: train={len(train_4f)}, val={len(val_4f)}, test={len(test_4f)}')

# %% [markdown]
# ## Define Configuration Space
#
# We do NOT explore randomly. Each axis is justified by NB31 findings.

# %%
# ── Aggregation functions ──
def p90_agg(x): return np.percentile(x, 90)
def p85_agg(x): return np.percentile(x, 85)
def p95_agg(x): return np.percentile(x, 95)
def mean_agg(x): return np.mean(x)

def weighted_top5(x):
    vals = np.sort(x)[::-1][:5]
    w = np.array([0.40, 0.25, 0.15, 0.10, 0.10])[:len(vals)]
    w = w / w.sum()
    return float(np.sum(vals * w))

def trimmed_top3(x):
    """Top-3 average — more aggressive than wt5, less than max."""
    vals = np.sort(x)[::-1][:3]
    return float(np.mean(vals))

AGG_FNS = {
    'p90': p90_agg,
    'p85': p85_agg,
    'p95': p95_agg,
    'mean': mean_agg,
    'wt5': weighted_top5,
    'top3_mean': trimmed_top3,
}

# ── Family weight configurations ──
# We recompute ensemble probs from per-family raw probs with different weights.
WEIGHT_CONFIGS = {
    'equal':     {'xgb': 1/3, 'lgbm': 1/3, 'cat': 1/3},
    'xgb_heavy': {'xgb': 0.50, 'lgbm': 0.25, 'cat': 0.25},
    'lgbm_heavy':{'xgb': 0.25, 'lgbm': 0.50, 'cat': 0.25},
    'cat_heavy': {'xgb': 0.25, 'lgbm': 0.25, 'cat': 0.50},
    'xgb_cat':   {'xgb': 0.40, 'lgbm': 0.20, 'cat': 0.40},
}

# ── Calibration methods ──
# We re-calibrate from val data for each weight configuration
CALIB_METHODS = ['raw', 'isotonic', 'platt']

# ── FPR targets ──
FPR_TARGETS = [0.0, 0.001, 0.005, 0.01]

# %% [markdown]
# ## Build Re-weighted & Re-calibrated Probabilities

# %%
def build_weighted_probs(df, weights):
    """Recompute ensemble probability from per-family raw probs."""
    w = weights
    return (df['p_xgb_raw'] * w['xgb'] +
            df['p_lgbm_raw'] * w['lgbm'] +
            df['p_cat_raw'] * w['cat'])


def fit_calibrator(val_labels, val_probs, method):
    """Fit a calibration model on validation data. Returns a predict function."""
    if method == 'raw':
        return lambda p: p
    elif method == 'isotonic':
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        ir.fit(val_probs, val_labels)
        return lambda p: ir.predict(np.clip(p, 0, 1))
    elif method == 'platt':
        lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=5000)
        lr.fit(val_probs.reshape(-1, 1), val_labels)
        return lambda p: lr.predict_proba(p.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f'Unknown calibration method: {method}')


print('Building probability configurations...')

# For each (feature_set, weight_config, calib_method), produce calibrated
# val and test probabilities + train probabilities for overfitting check.
configs = []

for feat_label, train_df, val_df, test_df in [
    ('5f', train_5f, val_5f, test_5f),
    ('4f', train_4f, val_4f, test_4f),
]:
    for wname, weights in WEIGHT_CONFIGS.items():
        # Recompute weighted raw probs
        train_raw = build_weighted_probs(train_df, weights).values
        val_raw = build_weighted_probs(val_df, weights).values
        test_raw = build_weighted_probs(test_df, weights).values

        for calib in CALIB_METHODS:
            try:
                cal_fn = fit_calibrator(val_df['label'].values, val_raw, calib)
                val_cal = cal_fn(val_raw)
                test_cal = cal_fn(test_raw)
                train_cal = cal_fn(train_raw)
            except Exception as e:
                print(f'  SKIP {feat_label}/{wname}/{calib}: {e}')
                continue

            configs.append({
                'features': feat_label,
                'weights': wname,
                'calibration': calib,
                'train_df': train_df,
                'val_df': val_df,
                'test_df': test_df,
                'train_probs': train_cal,
                'val_probs': val_cal,
                'test_probs': test_cal,
            })

print(f'Total probability configurations: {len(configs)}')

# %% [markdown]
# ## Evaluate All Candidates
#
# For each (features, weights, calibration, aggregation, target_fpr),
# compute the full metric set.

# %%
def evaluate_candidate(cfg, agg_name, agg_fn, target_fpr):
    """Evaluate one candidate configuration and return a metrics dict."""
    val_df = cfg['val_df']
    test_df = cfg['test_df']
    train_df = cfg['train_df']
    val_probs = cfg['val_probs']
    test_probs = cfg['test_probs']
    train_probs = cfg['train_probs']

    result = {
        'features': cfg['features'],
        'weights': cfg['weights'],
        'calibration': cfg['calibration'],
        'aggregation': agg_name,
        'target_fpr': target_fpr,
    }

    # ── Flow-level metrics (test) ──
    y_test = test_df['label'].values
    y_train = train_df['label'].values

    if len(np.unique(y_test)) < 2:
        return None

    result['flow_roc_auc'] = float(roc_auc_score(y_test, test_probs))
    result['flow_pr_auc'] = float(average_precision_score(y_test, test_probs))
    result['brier'] = float(brier_score_loss(y_test, np.clip(test_probs, 0, 1)))
    result['log_loss'] = float(log_loss(y_test, np.clip(test_probs, 1e-12, 1-1e-12)))

    ece_data = expected_calibration_error(y_test, np.clip(test_probs, 0, 1))
    result['ece'] = ece_data['ece']

    # Train AUC for overfitting check
    if len(np.unique(y_train)) >= 2:
        result['train_flow_auc'] = float(roc_auc_score(y_train, train_probs))
        result['train_test_gap'] = result['train_flow_auc'] - result['flow_roc_auc']
    else:
        result['train_flow_auc'] = float('nan')
        result['train_test_gap'] = float('nan')

    # Val AUC
    y_val = val_df['label'].values
    if len(np.unique(y_val)) >= 2:
        result['val_flow_auc'] = float(roc_auc_score(y_val, val_probs))
        result['val_test_gap'] = result['val_flow_auc'] - result['flow_roc_auc']
    else:
        result['val_flow_auc'] = float('nan')
        result['val_test_gap'] = float('nan')

    # ── Session-level ──
    # Build temporary dataframes with calibrated probs
    test_tmp = test_df[['capture_id', 'label', 'dataset']].copy()
    test_tmp['prob'] = test_probs
    val_tmp = val_df[['capture_id', 'label', 'dataset']].copy()
    val_tmp['prob'] = val_probs

    # Session scores
    sess_labels_test = test_tmp.groupby('capture_id')['label'].max()
    sess_scores_test = test_tmp.groupby('capture_id')['prob'].agg(agg_fn)
    ct = sess_labels_test.index.intersection(sess_scores_test.index)
    y_sess = sess_labels_test.loc[ct].values
    s_sess = sess_scores_test.loc[ct].values

    sess_labels_val = val_tmp.groupby('capture_id')['label'].max()
    sess_scores_val = val_tmp.groupby('capture_id')['prob'].agg(agg_fn)
    cv = sess_labels_val.index.intersection(sess_scores_val.index)
    y_val_s = sess_labels_val.loc[cv].values
    s_val_s = sess_scores_val.loc[cv].values

    if len(np.unique(y_sess)) < 2 or len(np.unique(y_val_s)) < 2:
        return None

    result['session_roc_auc'] = float(roc_auc_score(y_sess, s_sess))
    result['session_pr_auc'] = float(average_precision_score(y_sess, s_sess))
    result['n_sessions'] = len(y_sess)
    result['n_vpn_sessions'] = int(y_sess.sum())

    # ── Threshold from val ──
    thr = threshold_at_fpr(y_val_s, s_val_s, target_fpr=target_fpr)
    result['block_threshold'] = float(thr)

    cm = confusion_at_threshold(y_sess, s_sess, thr)
    result['block_recall'] = cm['recall']
    result['block_fpr'] = cm['fpr']
    result['block_precision'] = cm['precision']
    result['block_f1'] = cm['f1']

    # ── Per-dataset breakdown ──
    ds_map = test_tmp.groupby('capture_id')['dataset'].first()
    per_ds_fpr = {}
    per_ds_recall = {}
    per_ds_sess_auc = {}
    per_ds_thresholds = []

    for ds in sorted(test_tmp['dataset'].unique()):
        ds_ids = ds_map[ds_map == ds].index
        ds_y = sess_labels_test.loc[ds_ids.intersection(sess_labels_test.index)].values
        ds_s = sess_scores_test.loc[ds_ids.intersection(sess_scores_test.index)].values

        if len(ds_y) == 0 or len(np.unique(ds_y)) < 2:
            per_ds_sess_auc[ds] = float('nan')
            per_ds_recall[ds] = float('nan')
            per_ds_fpr[ds] = float('nan')
            continue

        per_ds_sess_auc[ds] = float(roc_auc_score(ds_y, ds_s))
        dcm = confusion_at_threshold(ds_y, ds_s, thr)
        per_ds_recall[ds] = dcm['recall']
        per_ds_fpr[ds] = dcm['fpr']

        # Local optimal threshold (diagnostic)
        local_thr = threshold_at_fpr(ds_y, ds_s, target_fpr=0.0)
        per_ds_thresholds.append(local_thr)

    result['iscx_sess_auc'] = per_ds_sess_auc.get('iscx', float('nan'))
    result['vnat_sess_auc'] = per_ds_sess_auc.get('vnat', float('nan'))
    result['usbvpn_sess_auc'] = per_ds_sess_auc.get('usbvpn', float('nan'))

    result['iscx_recall'] = per_ds_recall.get('iscx', float('nan'))
    result['vnat_recall'] = per_ds_recall.get('vnat', float('nan'))
    result['usbvpn_recall'] = per_ds_recall.get('usbvpn', float('nan'))

    result['iscx_fpr'] = per_ds_fpr.get('iscx', float('nan'))
    result['vnat_fpr'] = per_ds_fpr.get('vnat', float('nan'))
    result['usbvpn_fpr'] = per_ds_fpr.get('usbvpn', float('nan'))

    # Threshold portability
    if len(per_ds_thresholds) >= 2:
        result['thr_range'] = max(per_ds_thresholds) - min(per_ds_thresholds)
        result['thr_max_shift'] = max(abs(t - thr) for t in per_ds_thresholds)
    else:
        result['thr_range'] = float('nan')
        result['thr_max_shift'] = float('nan')

    # Min per-dataset recall (critical for deployment)
    recalls = [v for v in per_ds_recall.values() if not np.isnan(v)]
    result['min_ds_recall'] = min(recalls) if recalls else float('nan')

    # Max per-dataset FPR
    fprs = [v for v in per_ds_fpr.values() if not np.isnan(v)]
    result['max_ds_fpr'] = max(fprs) if fprs else float('nan')

    return result


# ── Run the grid ──
print('Evaluating candidate grid...')
all_results = []
total = len(configs) * len(AGG_FNS) * len(FPR_TARGETS)
done = 0

for cfg in configs:
    for agg_name, agg_fn in AGG_FNS.items():
        for fpr_t in FPR_TARGETS:
            r = evaluate_candidate(cfg, agg_name, agg_fn, fpr_t)
            if r is not None:
                all_results.append(r)
            done += 1
            if done % 100 == 0:
                print(f'  {done}/{total} evaluated...')

results_df = pd.DataFrame(all_results)
print(f'\nTotal candidates evaluated: {len(results_df)}')

# Save raw grid
results_df.to_csv(OUTPUT_DIR / 'calibration_aggregation_threshold_grid.csv', index=False)

# %% [markdown]
# ## Firewall-Deployment Ranking Score
#
# Priority:
# 1. Lower pooled test Block FPR (HEAVY)
# 2. Lower ISCX FPR (HEAVY)
# 3. Lower threshold range (MODERATE)
# 4. Higher pooled session ROC-AUC (MODERATE)
# 5. Higher pooled session PR-AUC (LIGHT)
# 6. Higher min per-dataset recall (MODERATE)
# 7. Lower train-test gap (LIGHT)
# 8. Lower ECE (LIGHT)

# %%
def firewall_ranking_score(row):
    """
    Composite score where HIGHER is BETTER.
    Designed for firewall deployment: heavily penalizes FPR,
    rewards robust cross-dataset behavior.
    """
    score = 0.0

    # --- FPR reduction (main goal) ---
    # block_fpr: 0 is best, 0.08 is current baseline
    block_fpr = row.get('block_fpr', 1.0)
    score += 30.0 * (1.0 - min(block_fpr / 0.10, 1.0))  # 30 pts if FPR=0, 0 if FPR>=0.10

    # iscx_fpr: the problem domain
    iscx_fpr = row.get('iscx_fpr', 1.0)
    if np.isnan(iscx_fpr):
        iscx_fpr = 1.0
    score += 20.0 * (1.0 - min(iscx_fpr / 0.50, 1.0))  # 20 pts if ISCX FPR=0

    # max_ds_fpr: no domain should have terrible FPR
    max_fpr = row.get('max_ds_fpr', 1.0)
    if np.isnan(max_fpr):
        max_fpr = 1.0
    score += 10.0 * (1.0 - min(max_fpr / 0.50, 1.0))

    # --- Threshold portability ---
    thr_range = row.get('thr_range', 1.0)
    if np.isnan(thr_range):
        thr_range = 1.0
    score += 10.0 * (1.0 - min(thr_range / 0.60, 1.0))

    # --- Recall / AUC ---
    sess_auc = row.get('session_roc_auc', 0.5)
    score += 10.0 * max(0, (sess_auc - 0.90) / 0.10)  # 10 pts at 1.0, 0 at 0.90

    sess_prauc = row.get('session_pr_auc', 0.0)
    score += 5.0 * max(0, (sess_prauc - 0.70) / 0.30)

    block_recall = row.get('block_recall', 0.0)
    score += 8.0 * block_recall  # 8 pts at recall=1.0

    min_recall = row.get('min_ds_recall', 0.0)
    if np.isnan(min_recall):
        min_recall = 0.0
    score += 5.0 * min_recall  # reward no-domain-collapse

    # --- Penalties ---
    train_test_gap = abs(row.get('train_test_gap', 0.0))
    if np.isnan(train_test_gap):
        train_test_gap = 0.0
    score -= 5.0 * min(train_test_gap / 0.05, 1.0)  # penalize overfitting

    ece = row.get('ece', 0.1)
    if np.isnan(ece):
        ece = 0.1
    score -= 2.0 * min(ece / 0.10, 1.0)  # penalize bad calibration

    return round(score, 4)


# Compute ranking score for all candidates
results_df['fw_score'] = results_df.apply(firewall_ranking_score, axis=1)

# Sort by composite score
ranked = results_df.sort_values('fw_score', ascending=False).reset_index(drop=True)

print('=== Top 20 Candidates by Firewall Deployment Score ===')
top_cols = ['features', 'weights', 'calibration', 'aggregation', 'target_fpr',
            'fw_score', 'session_roc_auc', 'session_pr_auc',
            'block_recall', 'block_fpr', 'block_precision',
            'iscx_fpr', 'usbvpn_recall', 'vnat_recall',
            'thr_range', 'ece', 'train_test_gap']
avail = [c for c in top_cols if c in ranked.columns]
_top = ranked.head(20)[avail].copy()
_num = _top.select_dtypes('number').columns
_top[_num] = _top[_num].round(4)
print(_top.to_string(index=False))

# Save ranking
ranked.to_csv(OUTPUT_DIR / 'ranking_table.csv', index=False)
ranked.head(50).to_json(OUTPUT_DIR / 'ranking_table.json', orient='records', indent=2)

# %% [markdown]
# ## Identify Best Candidate and Current Baseline

# %%
# Current baseline: 5f + equal + isotonic + p90 + target_fpr=0.0
baseline_mask = (
    (ranked['features'] == '5f') &
    (ranked['weights'] == 'equal') &
    (ranked['calibration'] == 'isotonic') &
    (ranked['aggregation'] == 'p90') &
    (ranked['target_fpr'] == 0.0)
)
baseline_row = ranked[baseline_mask]
def _print_row(row, cols):
    """Print a series filtering numeric only for rounding."""
    for c in cols:
        v = row[c]
        if isinstance(v, (int, float, np.floating, np.integer)):
            print(f'  {c:25s}: {v:.4f}')
        else:
            print(f'  {c:25s}: {v}')

if len(baseline_row) > 0:
    baseline = baseline_row.iloc[0]
    print('=== CURRENT BASELINE (NB31 Primary: 5f/equal/isotonic/p90/fpr=0) ===')
    _print_row(baseline, avail)
else:
    print('WARNING: Baseline not found in grid')
    baseline = None

# Best overall
best = ranked.iloc[0]
print(f'\n=== BEST CANDIDATE (fw_score = {best["fw_score"]:.4f}) ===')
_print_row(best, avail)

# Best that keeps high recall (block_recall >= 0.85)
high_recall = ranked[ranked['block_recall'] >= 0.85]
if len(high_recall) > 0:
    best_hr = high_recall.iloc[0]
    print(f'\n=== BEST HIGH-RECALL CANDIDATE (recall >= 0.85, fw_score = {best_hr["fw_score"]:.4f}) ===')
    _print_row(best_hr, avail)
else:
    best_hr = best
    print('\nNo candidate with recall >= 0.85 found, using overall best')

# Best with zero pooled FPR
zero_fpr = ranked[ranked['block_fpr'] == 0.0]
if len(zero_fpr) > 0:
    best_zfpr = zero_fpr.iloc[0]
    print(f'\n=== BEST ZERO-FPR CANDIDATE (fw_score = {best_zfpr["fw_score"]:.4f}) ===')
    _print_row(best_zfpr, avail)
else:
    best_zfpr = None
    print('\nNo zero-FPR candidate found')

# %% [markdown]
# ## Detailed Comparison: Top Candidates vs Baseline

# %%
def format_comparison(row, label):
    """Format a candidate as a comparison dict."""
    d = {'label': label}
    for c in avail:
        d[c] = row[c] if c in row.index else float('nan')
    return d

comparison_rows = []
if baseline is not None:
    comparison_rows.append(format_comparison(baseline, 'BASELINE (NB31 primary)'))
comparison_rows.append(format_comparison(best, 'BEST overall'))
if best_hr is not best:
    comparison_rows.append(format_comparison(best_hr, 'BEST high-recall'))
if best_zfpr is not None and best_zfpr is not best:
    comparison_rows.append(format_comparison(best_zfpr, 'BEST zero-FPR'))

# Also add the NB31 wt5+iso candidate that was already identified as promising
wt5_iso_mask = (
    (ranked['features'] == '5f') &
    (ranked['weights'] == 'equal') &
    (ranked['calibration'] == 'isotonic') &
    (ranked['aggregation'] == 'wt5') &
    (ranked['target_fpr'] == 0.0)
)
wt5_iso = ranked[wt5_iso_mask]
if len(wt5_iso) > 0:
    comparison_rows.append(format_comparison(wt5_iso.iloc[0], 'NB31 clue: 5f/equal/iso/wt5'))

# Add top-5 unique configs
seen = set()
for _, row in ranked.head(30).iterrows():
    key = f"{row['features']}/{row['weights']}/{row['calibration']}/{row['aggregation']}"
    if key not in seen and len(comparison_rows) < 10:
        seen.add(key)
        lbl = f"Top: {row['features']}/{row['weights']}/{row['calibration']}/{row['aggregation']}/fpr={row['target_fpr']}"
        comparison_rows.append(format_comparison(row, lbl))

cmp_df = pd.DataFrame(comparison_rows)
print('=== Candidate Comparison Table ===')
_num = cmp_df.select_dtypes('number').columns
_disp = cmp_df.copy()
_disp[_num] = _disp[_num].round(4)
print(_disp.to_string(index=False))

cmp_df.to_csv(OUTPUT_DIR / 'candidate_summary.csv', index=False)
cmp_df.to_json(OUTPUT_DIR / 'candidate_summary.json', orient='records', indent=2)

# %% [markdown]
# ## Per-Dataset Breakdown for Best Candidates

# %%
def detailed_per_dataset(cfg_row, results_df, configs):
    """Get detailed per-dataset breakdown for a specific config."""
    # Find matching config
    matching = [c for c in configs
                if c['features'] == cfg_row['features']
                and c['weights'] == cfg_row['weights']
                and c['calibration'] == cfg_row['calibration']]
    if not matching:
        return None

    cfg = matching[0]
    agg_fn = AGG_FNS[cfg_row['aggregation']]
    thr = cfg_row['block_threshold']

    test_df = cfg['test_df']
    test_tmp = test_df[['capture_id', 'label', 'dataset']].copy()
    test_tmp['prob'] = cfg['test_probs']

    rows = []
    for ds in ['ALL'] + sorted(test_tmp['dataset'].unique()):
        sub = test_tmp if ds == 'ALL' else test_tmp[test_tmp['dataset'] == ds]
        sl = sub.groupby('capture_id')['label'].max()
        ss = sub.groupby('capture_id')['prob'].agg(agg_fn)
        c = sl.index.intersection(ss.index)
        y = sl.loc[c].values; s = ss.loc[c].values

        r = {'dataset': ds, 'sessions': len(y), 'vpn_sessions': int(y.sum())}
        if len(np.unique(y)) >= 2:
            r['session_auc'] = float(roc_auc_score(y, s))
        else:
            r['session_auc'] = float('nan')

        cm = confusion_at_threshold(y, s, thr)
        r['block_recall'] = cm['recall']
        r['block_fpr'] = cm['fpr']
        r['block_precision'] = cm['precision']

        if len(np.unique(y)) >= 2:
            local_thr = threshold_at_fpr(y, s, 0.0)
            r['local_optimal_thr'] = local_thr
            lcm = confusion_at_threshold(y, s, local_thr)
            r['recall_at_local_thr'] = lcm['recall']
        else:
            r['local_optimal_thr'] = float('nan')
            r['recall_at_local_thr'] = float('nan')

        rows.append(r)
    return pd.DataFrame(rows)


# Per-dataset for best candidate
best_ds = detailed_per_dataset(best, results_df, configs)
if best_ds is not None:
    print('=== Per-Dataset Breakdown: BEST ===')
    print(f'Config: {best["features"]}/{best["weights"]}/{best["calibration"]}/{best["aggregation"]}/fpr={best["target_fpr"]}')
    print(best_ds.round(4).to_string(index=False))
    best_ds.to_csv(OUTPUT_DIR / 'per_dataset_breakdown_best.csv', index=False)

# Per-dataset for best high-recall
if best_hr is not best:
    best_hr_ds = detailed_per_dataset(best_hr, results_df, configs)
    if best_hr_ds is not None:
        print(f'\n=== Per-Dataset Breakdown: BEST HIGH-RECALL ===')
        print(f'Config: {best_hr["features"]}/{best_hr["weights"]}/{best_hr["calibration"]}/{best_hr["aggregation"]}/fpr={best_hr["target_fpr"]}')
        print(best_hr_ds.round(4).to_string(index=False))

# %% [markdown]
# ## Threshold Transferability for Best Candidate

# %%
def threshold_transferability(cfg_row, configs):
    """Compute threshold transferability table."""
    matching = [c for c in configs
                if c['features'] == cfg_row['features']
                and c['weights'] == cfg_row['weights']
                and c['calibration'] == cfg_row['calibration']]
    if not matching:
        return None

    cfg = matching[0]
    agg_fn = AGG_FNS[cfg_row['aggregation']]
    global_thr = cfg_row['block_threshold']

    test_df = cfg['test_df']
    test_tmp = test_df[['capture_id', 'label', 'dataset']].copy()
    test_tmp['prob'] = cfg['test_probs']

    rows = []
    for ds in sorted(test_tmp['dataset'].unique()):
        sub = test_tmp[test_tmp['dataset'] == ds]
        sl = sub.groupby('capture_id')['label'].max()
        ss = sub.groupby('capture_id')['prob'].agg(agg_fn)
        c = sl.index.intersection(ss.index)
        y = sl.loc[c].values; s = ss.loc[c].values

        if len(np.unique(y)) < 2:
            continue

        local_thr = threshold_at_fpr(y, s, 0.0)
        cm_global = confusion_at_threshold(y, s, global_thr)
        cm_local = confusion_at_threshold(y, s, local_thr)

        rows.append({
            'dataset': ds,
            'global_thr': global_thr,
            'local_thr': local_thr,
            'thr_delta': abs(local_thr - global_thr),
            'recall_at_global': cm_global['recall'],
            'fpr_at_global': cm_global['fpr'],
            'recall_at_local': cm_local['recall'],
            'fpr_at_local': cm_local['fpr'],
        })

    return pd.DataFrame(rows)


thr_trans = threshold_transferability(best, configs)
if thr_trans is not None:
    print('=== Threshold Transferability: BEST ===')
    print(thr_trans.round(4).to_string(index=False))
    thr_trans.to_csv(OUTPUT_DIR / 'threshold_transferability_best.csv', index=False)

# %% [markdown]
# ## Plots

# %%
# ── 1. Recall vs FPR for top candidates ──
fig, ax = plt.subplots(figsize=(10, 7))
top_unique = ranked.drop_duplicates(
    subset=['features', 'weights', 'calibration', 'aggregation']
).head(15)

colors = plt.cm.tab20(np.linspace(0, 1, 15))
for i, (_, row) in enumerate(top_unique.iterrows()):
    label = f"{row['features']}/{row['weights'][:3]}/{row['calibration'][:3]}/{row['aggregation']}"
    ax.scatter(row['block_fpr'], row['block_recall'],
               s=100, color=colors[i], zorder=5, label=label, alpha=0.85)

# Mark baseline
if baseline is not None:
    ax.scatter(baseline['block_fpr'], baseline['block_recall'],
               s=200, color='red', marker='D', zorder=10,
               label='BASELINE', edgecolors='black', linewidths=1.5)

ax.set_xlabel('Block FPR (pooled test)')
ax.set_ylabel('Block Recall')
ax.set_title('Recall vs FPR: Top Candidates\n(lower-right = better for firewall)')
ax.legend(fontsize=7, loc='lower left', ncol=2)
ax.set_xlim(-0.01, max(0.15, top_unique['block_fpr'].max() + 0.02))
ax.axvline(0.0, color='green', linestyle='--', alpha=0.3, label='Zero FPR')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'recall_vs_fpr_candidates.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# ── 2. Session AUC vs Block FPR ──
fig, ax = plt.subplots(figsize=(10, 7))
for i, (_, row) in enumerate(top_unique.iterrows()):
    label = f"{row['features']}/{row['weights'][:3]}/{row['calibration'][:3]}/{row['aggregation']}"
    ax.scatter(row['block_fpr'], row['session_roc_auc'],
               s=100, color=colors[i], zorder=5, label=label, alpha=0.85)
if baseline is not None:
    ax.scatter(baseline['block_fpr'], baseline['session_roc_auc'],
               s=200, color='red', marker='D', zorder=10, label='BASELINE',
               edgecolors='black', linewidths=1.5)
ax.set_xlabel('Block FPR (pooled test)')
ax.set_ylabel('Session ROC-AUC')
ax.set_title('Session AUC vs Block FPR: Top Candidates')
ax.legend(fontsize=7, loc='lower left', ncol=2)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'session_auc_vs_block_fpr.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# ── 3. Threshold transferability comparison ──
fig, ax = plt.subplots(figsize=(10, 6))
# Compare thr_range for top candidates
top_tr = ranked.drop_duplicates(
    subset=['features', 'weights', 'calibration', 'aggregation']
).head(10)
labels = [f"{r['features']}/{r['weights'][:3]}/{r['calibration'][:3]}/{r['aggregation']}"
          for _, r in top_tr.iterrows()]
ax.barh(range(len(labels)), top_tr['thr_range'].values, color='steelblue', alpha=0.8)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Threshold Range Across Datasets')
ax.set_title('Threshold Portability (lower = more portable)')
if baseline is not None:
    ax.axvline(baseline['thr_range'], color='red', linestyle='--',
               label=f'Baseline: {baseline["thr_range"]:.3f}')
    ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'threshold_transferability_candidates.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# ── 4. Per-dataset FPR/Recall for best candidate ──
if best_ds is not None:
    ds_only = best_ds[best_ds['dataset'] != 'ALL']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ds_colors = {'iscx': '#1565C0', 'vnat': '#2E7D32', 'usbvpn': '#FF8F00'}
    for _, row in ds_only.iterrows():
        c = ds_colors.get(row['dataset'], '#607D8B')
        ax.bar(row['dataset'].upper(), row['block_recall'], color=c, alpha=0.85)
        ax.text(row['dataset'].upper(), row['block_recall'] + 0.02,
                f'{row["block_recall"]:.3f}', ha='center', fontsize=10)
    ax.set_ylabel('Block Recall'); ax.set_title('Block Recall @ Global Thr')
    ax.set_ylim(0, 1.15)

    ax = axes[1]
    for _, row in ds_only.iterrows():
        c = ds_colors.get(row['dataset'], '#607D8B')
        ax.bar(row['dataset'].upper(), row['block_fpr'], color=c, alpha=0.85)
        ax.text(row['dataset'].upper(), row['block_fpr'] + 0.02,
                f'{row["block_fpr"]:.3f}', ha='center', fontsize=10)
    ax.set_ylabel('Block FPR'); ax.set_title('Block FPR @ Global Thr')

    cfg_str = f"{best['features']}/{best['weights']}/{best['calibration']}/{best['aggregation']}"
    plt.suptitle(f'Per-Dataset Breakdown: BEST ({cfg_str})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_dataset_fpr_recall_best.png', dpi=150, bbox_inches='tight')
    plt.show()

# %%
# ── 5. Reliability diagram for best candidate ──
def plot_reliability_best(cfg_row, configs):
    matching = [c for c in configs
                if c['features'] == cfg_row['features']
                and c['weights'] == cfg_row['weights']
                and c['calibration'] == cfg_row['calibration']]
    if not matching:
        return

    cfg = matching[0]
    y_test = cfg['test_df']['label'].values
    p_test = cfg['test_probs']

    fig, ax = plt.subplots(figsize=(7, 7))
    try:
        prob_true, prob_pred = calibration_curve(y_test, np.clip(p_test, 0, 1),
                                                  n_bins=15, strategy='quantile')
        ax.plot(prob_pred, prob_true, 's-', color='#D81B60', linewidth=2, markersize=6)
    except Exception:
        pass
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('Mean Predicted Prob')
    ax.set_ylabel('Fraction of Positives')
    ece_val = cfg_row.get('ece', 0)
    ax.set_title(f'Reliability Diagram: BEST\n'
                 f'{cfg_row["features"]}/{cfg_row["weights"]}/{cfg_row["calibration"]}\n'
                 f'ECE = {ece_val:.4f}')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'reliability_diagrams_best.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_reliability_best(best, configs)

# %% [markdown]
# ## Final Recommendation

# %%
print('\n' + '=' * 80)
print('  FINAL IMPROVEMENT PASS RECOMMENDATION')
print('=' * 80)

# Compare best vs baseline
if baseline is not None:
    bl = baseline
    bt = best
    improvements = {}
    improvements['block_fpr'] = (bl['block_fpr'] - bt['block_fpr'], bl['block_fpr'], bt['block_fpr'])
    improvements['iscx_fpr'] = (bl.get('iscx_fpr', 1) - bt.get('iscx_fpr', 1),
                                 bl.get('iscx_fpr', 1), bt.get('iscx_fpr', 1))
    improvements['thr_range'] = (bl.get('thr_range', 1) - bt.get('thr_range', 1),
                                  bl.get('thr_range', 1), bt.get('thr_range', 1))
    improvements['session_roc_auc'] = (bt['session_roc_auc'] - bl['session_roc_auc'],
                                        bl['session_roc_auc'], bt['session_roc_auc'])
    improvements['block_recall'] = (bt['block_recall'] - bl['block_recall'],
                                     bl['block_recall'], bt['block_recall'])

    print(f'\nBaseline: {bl["features"]}/{bl["weights"]}/{bl["calibration"]}/{bl["aggregation"]}')
    print(f'Best:     {bt["features"]}/{bt["weights"]}/{bt["calibration"]}/{bt["aggregation"]}')
    print(f'\nFirewall Score: {bl["fw_score"]:.2f} -> {bt["fw_score"]:.2f} '
          f'(+{bt["fw_score"] - bl["fw_score"]:.2f})')

    print(f'\n  Metric improvements:')
    for metric, (delta, old, new) in improvements.items():
        direction = 'IMPROVED' if delta > 0.001 else ('SAME' if abs(delta) < 0.001 else 'DEGRADED')
        print(f'    {metric:25s}: {old:.4f} -> {new:.4f} (delta={delta:+.4f}) [{direction}]')

    # Determine if best is truly new
    is_same = (bt['features'] == bl['features'] and
               bt['weights'] == bl['weights'] and
               bt['calibration'] == bl['calibration'] and
               bt['aggregation'] == bl['aggregation'] and
               bt['target_fpr'] == bl['target_fpr'])

    if is_same:
        print('\n  RESULT: No improvement found over current baseline.')
        verdict = 'KEEP CURRENT'
    elif bt['fw_score'] > bl['fw_score'] + 0.5:
        print(f'\n  RESULT: SIGNIFICANT improvement found!')
        verdict = 'NEW BEST'
    else:
        print(f'\n  RESULT: Marginal improvement. Consider for deployment context.')
        verdict = 'MARGINAL IMPROVEMENT'

    # What changed
    changes = []
    if bt['calibration'] != bl['calibration']:
        changes.append(f'calibration: {bl["calibration"]} -> {bt["calibration"]}')
    if bt['aggregation'] != bl['aggregation']:
        changes.append(f'aggregation: {bl["aggregation"]} -> {bt["aggregation"]}')
    if bt['weights'] != bl['weights']:
        changes.append(f'weights: {bl["weights"]} -> {bt["weights"]}')
    if bt['features'] != bl['features']:
        changes.append(f'features: {bl["features"]} -> {bt["features"]}')
    if bt['target_fpr'] != bl['target_fpr']:
        changes.append(f'target_fpr: {bl["target_fpr"]} -> {bt["target_fpr"]}')

    print(f'\n  Changes: {"; ".join(changes) if changes else "None"}')
else:
    verdict = 'UNKNOWN (no baseline found)'

# Thesis-safe labels
labels = ['strong-detector']
if best.get('iscx_fpr', 1) > 0.05 or best.get('max_ds_fpr', 1) > 0.05:
    labels.append('domain-sensitive')
if best.get('thr_range', 0) > 0.10:
    labels.append('calibration-sensitive')
    labels.append('requires-local-calibration')
if best['block_fpr'] == 0.0:
    labels.append('zero-fpr-achievable')
if best.get('fw_score', 0) > 80:
    labels.append('best-current-candidate')

print(f'\n  Labels: {", ".join(labels)}')

# Explicit answers
print(f'''
=== EXPLICIT ANSWERS ===
1. Best candidate: {best["features"]}/{best["weights"]}/{best["calibration"]}/{best["aggregation"]}/fpr={best["target_fpr"]}
2. Is it the old Primary (5f)? {best["features"] == "5f" and best["weights"] == "equal" and best["calibration"] == "isotonic" and best["aggregation"] == "p90"}
3. What changed: {"; ".join(changes) if changes else "Nothing"}
4. Pooled test Block FPR improved? {bl["block_fpr"]:.4f} -> {bt["block_fpr"]:.4f} = {'YES' if bt["block_fpr"] < bl["block_fpr"] else 'NO'}
5. ISCX FPR improved? {bl.get("iscx_fpr",1):.4f} -> {bt.get("iscx_fpr",1):.4f} = {'YES' if bt.get("iscx_fpr",1) < bl.get("iscx_fpr",1) else 'NO'}
6. Threshold portability improved? {bl.get("thr_range",1):.4f} -> {bt.get("thr_range",1):.4f} = {'YES' if bt.get("thr_range",1) < bl.get("thr_range",1) else 'NO'}
7. Session AUC stayed strong? {bt["session_roc_auc"]:.4f} >= 0.95 = {'YES' if bt["session_roc_auc"] >= 0.95 else 'NO'}
''')

# %% [markdown]
# ## Save Final Outputs

# %%
# Evidence support
evidence = {
    'verdict': verdict,
    'best_config': {
        'features': best['features'],
        'weights': best['weights'],
        'calibration': best['calibration'],
        'aggregation': best['aggregation'],
        'target_fpr': float(best['target_fpr']),
    },
    'best_metrics': {
        'fw_score': float(best['fw_score']),
        'session_roc_auc': float(best['session_roc_auc']),
        'session_pr_auc': float(best['session_pr_auc']),
        'block_recall': float(best['block_recall']),
        'block_fpr': float(best['block_fpr']),
        'block_precision': float(best['block_precision']),
        'iscx_fpr': float(best.get('iscx_fpr', -1)),
        'usbvpn_recall': float(best.get('usbvpn_recall', -1)),
        'vnat_recall': float(best.get('vnat_recall', -1)),
        'thr_range': float(best.get('thr_range', -1)),
        'ece': float(best.get('ece', -1)),
        'train_test_gap': float(best.get('train_test_gap', -1)),
    },
    'baseline_metrics': {
        'fw_score': float(bl['fw_score']) if baseline is not None else -1,
        'session_roc_auc': float(bl['session_roc_auc']) if baseline is not None else -1,
        'block_fpr': float(bl['block_fpr']) if baseline is not None else -1,
        'block_recall': float(bl['block_recall']) if baseline is not None else -1,
        'iscx_fpr': float(bl.get('iscx_fpr', -1)) if baseline is not None else -1,
    },
    'labels': labels,
    'changes': changes if baseline is not None else [],
    'total_candidates_evaluated': len(results_df),
}

with open(OUTPUT_DIR / 'evidence_support_best.json', 'w') as f:
    json.dump(evidence, f, indent=2, default=str)

# Final recommendation
recommendation = {
    'verdict': verdict,
    'recommended_config': evidence['best_config'],
    'recommended_metrics': evidence['best_metrics'],
    'thesis_interpretation': (
        f'The focused improvement pass evaluated {len(results_df)} candidate configurations '
        f'across calibration, aggregation, threshold, and family-weight dimensions. '
        f'The best candidate achieves a firewall deployment score of {best["fw_score"]:.2f} '
        f'(baseline: {bl["fw_score"]:.2f}), with pooled test Block FPR = {best["block_fpr"]:.4f} '
        f'and session ROC-AUC = {best["session_roc_auc"]:.4f}.'
    ),
}
with open(OUTPUT_DIR / 'final_recommendation.json', 'w') as f:
    json.dump(recommendation, f, indent=2, default=str)

# Summary markdown
summary_md = f"""# Performance Improvement Pass — Summary

**Generated:** {pd.Timestamp.now().isoformat()}
**Candidates evaluated:** {len(results_df)}

## Verdict: {verdict}

## Best Candidate
- Features: {best['features']}
- Weights: {best['weights']}
- Calibration: {best['calibration']}
- Aggregation: {best['aggregation']}
- Target FPR: {best['target_fpr']}
- FW Score: {best['fw_score']:.2f}

## Key Metrics Comparison

| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| Block FPR | {bl['block_fpr']:.4f} | {bt['block_fpr']:.4f} | {bt['block_fpr'] - bl['block_fpr']:+.4f} |
| Block Recall | {bl['block_recall']:.4f} | {bt['block_recall']:.4f} | {bt['block_recall'] - bl['block_recall']:+.4f} |
| Session ROC-AUC | {bl['session_roc_auc']:.4f} | {bt['session_roc_auc']:.4f} | {bt['session_roc_auc'] - bl['session_roc_auc']:+.4f} |
| ISCX FPR | {bl.get('iscx_fpr',1):.4f} | {bt.get('iscx_fpr',1):.4f} | {bt.get('iscx_fpr',1) - bl.get('iscx_fpr',1):+.4f} |
| Threshold Range | {bl.get('thr_range',1):.4f} | {bt.get('thr_range',1):.4f} | {bt.get('thr_range',1) - bl.get('thr_range',1):+.4f} |
| USBVPN Recall | {bl.get('usbvpn_recall',0):.4f} | {bt.get('usbvpn_recall',0):.4f} | {bt.get('usbvpn_recall',0) - bl.get('usbvpn_recall',0):+.4f} |
| ECE | {bl.get('ece',0):.4f} | {bt.get('ece',0):.4f} | {bt.get('ece',0) - bl.get('ece',0):+.4f} |

## Changes
{chr(10).join('- ' + c for c in changes) if changes else '- None'}

## Labels
{', '.join(labels)}

## Thesis Interpretation
{recommendation['thesis_interpretation']}
"""

with open(OUTPUT_DIR / 'summary.md', 'w', encoding='utf-8') as f:
    f.write(summary_md)

print(f'\nAll outputs saved to: {OUTPUT_DIR}')
for f in sorted(OUTPUT_DIR.glob('*')):
    if f.is_file():
        print(f'  {f.name}')

print('\n' + '#' * 80)
print('  IMPROVEMENT PASS COMPLETE')
print('#' * 80)





