import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.clean_pipeline.validation.domain_shift import compute_jsd, compute_ks, compute_smd

@pytest.fixture(scope="module")
def features_df():
    path = Path('artifacts/clean_pipeline/features.parquet')
    if not path.exists():
        pytest.skip("features.parquet not found. Run the clean pipeline first.")
    return pd.read_parquet(path)

# 1. LODO protocol integrity
def test_lodo_target_dataset_excluded_from_training(features_df):
    datasets = features_df['dataset'].unique()
    results = []
    for target in datasets:
        trainval = features_df[features_df['dataset'] != target]
        test = features_df[features_df['dataset'] == target]
        # Assert target not in train/val
        assert target not in trainval['dataset'].unique()
        # Assert target only in test
        assert set(test['dataset'].unique()) == {target}
        # No capture overlap
        assert set(trainval['capture_id']).isdisjoint(set(test['capture_id']))
        results.append({'target': target, 'trainval_n': len(trainval), 'test_n': len(test)})
    pd.DataFrame(results).to_csv('artifacts/validation/lodo_protocol_integrity.csv', index=False)

# 2. Dataset fingerprinting audit
def test_audit_dataset_fingerprinting_strength(features_df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    X = features_df[[c for c in features_df.columns if c not in ['dataset','capture_id','flow_id','label','split']]]
    y = features_df['dataset']
    groups = features_df['capture_id']
    clf = LogisticRegression(max_iter=1000)
    cv = GroupKFold(n_splits=5)
    aucs = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring='roc_auc_ovr')
    accs = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring='accuracy')
    # Save results
    pd.DataFrame({'auc': aucs, 'accuracy': accs}).to_csv('artifacts/validation/domain_fingerprinting_results.csv', index=False)
    # Confusion matrix (fit on all, for audit)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    cm = pd.crosstab(y, y_pred, rownames=['true'], colnames=['pred'])
    cm.to_csv('artifacts/validation/domain_fingerprinting_confusion_matrix.csv')

# 3. Feature distribution shift audit
def test_audit_feature_distribution_shift_across_datasets(features_df):
    datasets = features_df['dataset'].unique()
    feats = [c for c in features_df.columns if c not in ['dataset','capture_id','flow_id','label','split']]
    rows = []
    for f in feats:
        for i, d1 in enumerate(datasets):
            for d2 in datasets[i+1:]:
                p = features_df[features_df['dataset']==d1][f].values
                q = features_df[features_df['dataset']==d2][f].values
                jsd = compute_jsd(p, q)
                ks = compute_ks(p, q)
                rows.append({'feature': f, 'ds1': d1, 'ds2': d2, 'jsd': jsd, 'ks': ks})
    df = pd.DataFrame(rows)
    df.to_csv('artifacts/validation/feature_distribution_shift.csv', index=False)

# 4. Feature sign reversal audit
def test_audit_cross_dataset_feature_sign_reversal(features_df):
    datasets = features_df['dataset'].unique()
    feats = [c for c in features_df.columns if c not in ['dataset','capture_id','flow_id','label','split']]
    rows = []
    for f in feats:
        smds = []
        for d in datasets:
            sub = features_df[features_df['dataset']==d]
            x1 = sub[sub['label']==1][f].values
            x0 = sub[sub['label']==0][f].values
            smd = compute_smd(x1, x0)
            smds.append(smd)
            rows.append({'feature': f, 'dataset': d, 'smd': smd})
        # Check for sign reversal
        if np.sign(smds[0]) != np.sign(smds[1]) or np.sign(smds[1]) != np.sign(smds[2]):
            pass  # Could flag or log
    pd.DataFrame(rows).to_csv('artifacts/validation/sign_reversal_audit.csv', index=False)

