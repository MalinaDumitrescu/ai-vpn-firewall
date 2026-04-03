"""Deep audit: check for missing features, leakage, and domain fingerprinting."""
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(line_buffering=True)

df = pd.read_parquet('artifacts/clean_pipeline/features.parquet')
datasets = sorted(df.dataset.unique())
print(f'Total flows: {len(df)}')
print(f'Datasets: {datasets}')
for ds in datasets:
    print(f'  {ds}: {len(df[df.dataset==ds])} flows')

meta = ['flow_id','capture_id','dataset','label','app','source_file','split']
feat_cols = [c for c in df.columns if c not in meta and df[c].dtype in ['float64','float32','int64','int32']]
print(f'\nNumeric features ({len(feat_cols)}): {feat_cols}')

print('\n' + '='*80)
print('AUDIT 1: NaN/NULL/ZERO/CONSTANT per dataset per feature')
print('='*80)
issues = []
for ds in datasets:
    sub = df[df.dataset == ds]
    n = len(sub)
    print(f'\n--- {ds} ({n} flows) ---')
    for f in feat_cols:
        col = sub[f]
        n_null = int(col.isnull().sum())
        n_zero = int((col == 0).sum())
        n_neg999 = int((col == -999).sum())
        pct_null = n_null / n * 100
        pct_zero = n_zero / n * 100
        first_val = col.iloc[0] if n > 0 else None
        n_const = int((col == first_val).sum()) if first_val is not None else 0
        pct_const = n_const / n * 100
        flag = False
        if pct_null > 0.5:
            print(f'  *** NULL: {f} has {n_null} nulls ({pct_null:.1f}%)')
            issues.append((ds, f, 'NULL', pct_null))
            flag = True
        if pct_zero > 85:
            print(f'  *** ZERO: {f} has {n_zero} zeros ({pct_zero:.1f}%)')
            issues.append((ds, f, 'ZERO', pct_zero))
            flag = True
        if pct_const > 95 and n > 50:
            val = first_val
            print(f'  *** CONST: {f} is constant={val} for {pct_const:.1f}% of flows')
            issues.append((ds, f, 'CONST', pct_const))
            flag = True
        if n_neg999 > 0:
            print(f'  *** SENTINEL: {f} has {n_neg999} values == -999')
            issues.append((ds, f, 'SENTINEL', n_neg999))

print('\n' + '='*80)
print('AUDIT 2: Zero-imbalance across datasets')
print('='*80)
for f in feat_cols:
    zeros = {}
    for ds in datasets:
        sub = df[df.dataset == ds][f]
        zeros[ds] = (sub == 0).mean() * 100
    max_z, min_z = max(zeros.values()), min(zeros.values())
    if max_z - min_z > 30:
        parts = ', '.join(f'{ds}={zeros[ds]:.1f}%' for ds in datasets)
        print(f'  *** {f}: zero-rate gap={max_z-min_z:.1f}% ({parts})')

print('\n' + '='*80)
print('AUDIT 3: Between-dataset mean ratio (domain fingerprint signal)')
print('='*80)
domain_signals = []
for f in feat_cols:
    means = []
    for ds in datasets:
        means.append(df[df.dataset == ds][f].dropna().mean())
    overall_std = df[f].dropna().std()
    between_std = np.std(means)
    ratio = between_std / (overall_std + 1e-12)
    domain_signals.append((f, ratio, means))
    if ratio > 0.25:
        parts = ', '.join(f'{ds}={m:.4f}' for ds, m in zip(datasets, means))
        print(f'  *** {f}: ratio={ratio:.3f} ({parts})')

# Sort by domain signal strength
domain_signals.sort(key=lambda x: x[1], reverse=True)
print('\n  Top 10 domain-fingerprinting features:')
for f, ratio, means in domain_signals[:10]:
    parts = ', '.join(f'{ds}={m:.2f}' for ds, m in zip(datasets, means))
    print(f'    {f:25s} ratio={ratio:.3f}  ({parts})')

print('\n' + '='*80)
print('AUDIT 4: Check if features were extracted from real data or are stubs')
print('='*80)
for f in feat_cols:
    for ds in datasets:
        sub = df[df.dataset == ds][f].dropna()
        if len(sub) == 0:
            print(f'  *** EMPTY: {f} has no values in {ds}')
            continue
        nunique = sub.nunique()
        if nunique <= 3 and len(sub) > 100:
            vals = sub.value_counts().head(5)
            print(f'  *** LOW CARDINALITY: {f} in {ds} has only {nunique} unique values: {dict(vals)}')

print('\n' + '='*80)
print('AUDIT 5: Directional features — are they real or stub?')
print('='*80)
dir_feats = [f for f in feat_cols if 'dir_' in f]
print(f'  Directional features: {dir_feats}')
for f in dir_feats:
    for ds in datasets:
        sub = df[df.dataset == ds][f].dropna()
        print(f'  {f} in {ds}: mean={sub.mean():.4f}, std={sub.std():.4f}, '
              f'nunique={sub.nunique()}, zeros={int((sub==0).sum())} '
              f'({(sub==0).mean()*100:.1f}%)')

print('\n' + '='*80)
print('SUMMARY: Total issues found')
print('='*80)
if issues:
    print(f'  {len(issues)} issues found:')
    for ds, f, kind, val in issues:
        print(f'    [{ds}] {f}: {kind} ({val:.1f})')
else:
    print('  No NULL/ZERO/CONST/SENTINEL issues found.')

# Final: try a 1-feature domain classifier on each feature
print('\n' + '='*80)
print('AUDIT 6: Single-feature domain AUC (can ONE feature identify dataset?)')
print('='*80)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_auc_score

le = LabelEncoder()
y_ds = le.fit_transform(df['dataset'].values)
rng = np.random.default_rng(42)
idx = rng.permutation(len(df))
sp = int(0.7 * len(df))

single_aucs = []
for f in feat_cols:
    x = df[f].fillna(0).values.reshape(-1, 1)
    rf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42, n_jobs=-1)
    rf.fit(x[idx[:sp]], y_ds[idx[:sp]])
    proba = rf.predict_proba(x[idx[sp:]])
    y_bin = label_binarize(y_ds[idx[sp:]], classes=list(range(len(le.classes_))))
    try:
        auc = float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro"))
    except:
        auc = 0.5
    single_aucs.append((f, auc))

single_aucs.sort(key=lambda x: x[1], reverse=True)
print('\n  Single-feature domain AUC ranking (higher = more fingerprinting):')
for f, auc in single_aucs:
    flag = ' *** LEAKY' if auc > 0.80 else ' ** moderate' if auc > 0.70 else ''
    print(f'    {f:25s} AUC={auc:.4f}{flag}')

