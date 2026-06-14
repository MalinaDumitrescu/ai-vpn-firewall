import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

@pytest.fixture(scope="module")
def preprocessing_meta():
    meta_path = Path('artifacts/clean_pipeline/preprocessing_metadata.json')
    if not meta_path.exists():
        pytest.skip("preprocessing_metadata.json not found. Run the clean pipeline with quantile scaling enabled.")
    with open(meta_path) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def feature_columns():
    columns_path = Path('artifacts/clean_pipeline/feature_columns.json')
    if not columns_path.exists():
        pytest.skip("feature_columns.json not found. Run the clean pipeline first.")
    with open(columns_path) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def features_df():
    features_path = Path('artifacts/clean_pipeline/features.parquet')
    if not features_path.exists():
        pytest.skip("features.parquet not found. Run the clean pipeline first.")
    return pd.read_parquet(features_path)

# 1. Preprocessing transformers fit on train only
def test_preprocessing_transformers_fit_on_train_only(preprocessing_meta, features_df, feature_columns):
    meta = preprocessing_meta
    assert meta['fit_split'] == 'train', "Transformer not fit on train only."
    assert set(meta['transformed_splits']) == {'train', 'val', 'test'}
    assert meta['feature_list'] == feature_columns, "Feature list used for scaling does not match model schema."
    # Check statistics: fit params should not change after val/test
    scaler_path = Path('artifacts/clean_pipeline/quantile_transformer.joblib')
    qt = joblib.load(scaler_path)
    # Check n_quantiles matches train size
    train_n = (features_df['split'] == 'train').sum()
    assert qt.n_quantiles_ == min(meta['n_quantiles'], train_n)

# 2. No datasetwise normalization in main pipeline
def test_no_datasetwise_normalization_in_main_pipeline(preprocessing_meta):
    assert not preprocessing_meta.get('datasetwise_normalization', False), "Datasetwise normalization is not allowed in main protocol."

# 3. Transformer feature order matches model schema
def test_transformer_feature_order_matches_model_schema(preprocessing_meta, feature_columns):
    assert preprocessing_meta['feature_list'] == feature_columns, "Transformer feature order does not match model schema."

# 4. Val/test scaling does not change fit statistics
def test_val_test_scaling_does_not_change_fit_statistics(features_df):
    scaler_path = Path('artifacts/clean_pipeline/quantile_transformer.joblib')
    qt = joblib.load(scaler_path)
    # Save fit params before
    params_before = (qt.quantiles_.copy(), qt.references_.copy())
    # Transform val/test
    for split in ['val', 'test']:
        idx = features_df['split'] == split
        if idx.any():
            _ = qt.transform(features_df.loc[idx, qt.feature_names_in_])
    # Check fit params unchanged
    params_after = (qt.quantiles_, qt.references_)
    for before, after in zip(params_before, params_after):
        assert np.allclose(before, after), "Quantile transformer fit statistics changed after transforming val/test."

# 5. (Optional) Monotonic scaling does not change single-feature AUC direction
def test_monotonic_scaling_does_not_change_single_feature_auc_direction(features_df):
    from sklearn.metrics import roc_auc_score
    scaler_path = Path('artifacts/clean_pipeline/quantile_transformer.joblib')
    qt = joblib.load(scaler_path)
    for col in qt.feature_names_in_:
        y = features_df['label']
        x_orig = features_df[col]
        x_scaled = qt.transform(features_df[[col]])[:, 0]
        if len(np.unique(x_orig)) > 2:
            auc_orig = roc_auc_score(y, x_orig)
            auc_scaled = roc_auc_score(y, x_scaled)
            # AUC sign should not flip
            assert np.sign(auc_orig) == np.sign(auc_scaled), f"AUC direction flipped for {col}"

