"""
Audit and evaluation utilities for benign persona false-positive scenarios in the clean VPN firewall pipeline.
"""
import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

BENIGN_PERSONA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', 'benign_personas')


def list_benign_personas() -> List[str]:
    """Return list of available persona scenario folder names."""
    if not os.path.exists(BENIGN_PERSONA_ROOT):
        return []
    return [d for d in os.listdir(BENIGN_PERSONA_ROOT) if os.path.isdir(os.path.join(BENIGN_PERSONA_ROOT, d))]


def load_persona_metadata(persona: str) -> Dict:
    """Load metadata.json for a given persona."""
    meta_path = os.path.join(BENIGN_PERSONA_ROOT, persona, 'metadata.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata.json for persona: {persona}")
    with open(meta_path, 'r') as f:
        return json.load(f)


def find_feature_file(persona: str) -> Optional[str]:
    """Find a feature CSV or Parquet file for the persona, or return None if missing."""
    folder = os.path.join(BENIGN_PERSONA_ROOT, persona)
    for ext in ('.csv', '.parquet'):
        for fname in os.listdir(folder):
            if fname.endswith(ext):
                return os.path.join(folder, fname)
    return None


def load_persona_features(persona: str) -> pd.DataFrame:
    """Load features for a persona, or raise if missing."""
    fpath = find_feature_file(persona)
    if fpath is None:
        raise FileNotFoundError(f"No feature file found for persona: {persona}")
    if fpath.endswith('.csv'):
        return pd.read_csv(fpath)
    elif fpath.endswith('.parquet'):
        return pd.read_parquet(fpath)
    else:
        raise ValueError(f"Unknown feature file type: {fpath}")


def check_schema_matches(features: pd.DataFrame, model_columns: List[str]) -> bool:
    """Check that features match model input columns exactly."""
    return list(features.columns) == model_columns and not features.isnull().any().any() and np.isfinite(features.values).all()


def compute_block_monitor_pass(scores: np.ndarray, block_thresh: float, monitor_thresh: float) -> Dict[str, float]:
    """Compute block, monitor, and pass rates given scores and thresholds."""
    block = (scores >= block_thresh).mean()
    monitor = ((scores >= monitor_thresh) & (scores < block_thresh)).mean()
    passed = (scores < monitor_thresh).mean()
    return {'block_rate': block, 'monitor_rate': monitor, 'pass_rate': passed}

# Placeholder for model scoring logic

def score_persona_features(features: pd.DataFrame) -> np.ndarray:
    """
    Synthetic scoring: Simulate robust ensemble model output for demo/testing.
    Produces low VPN risk scores for benign synthetic data.
    """
    # Simulate a robust ensemble: benign flows get low scores, some random noise
    np.random.seed(42)
    base_score = 0.01 + 0.01 * np.random.rand(len(features))
    # Add a small scenario-dependent offset for variety
    if 'feature1' in features.columns:
        base_score += 0.01 * features['feature1'].values
    return np.clip(base_score, 0, 1)

# Placeholder for session aggregation

def aggregate_session_scores(features: pd.DataFrame, scores: np.ndarray, session_col: str = 'capture_id') -> pd.DataFrame:
    """
    Aggregate flow scores to session/capture level using mean, p90, max.
    Returns a DataFrame with session_id and aggregated scores.
    """
    if session_col not in features.columns:
        raise ValueError(f"Session column '{session_col}' not found in features.")
    df = features.copy()
    df['score'] = scores
    grouped = df.groupby(session_col)['score']
    return grouped.agg(['mean', lambda x: np.percentile(x, 90), 'max']).rename(columns={'<lambda_0>': 'p90'})