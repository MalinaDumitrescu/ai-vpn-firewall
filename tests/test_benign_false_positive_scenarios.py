import os
import pytest
import pandas as pd
import numpy as np
from src.clean_pipeline.validation import benign_scenarios

# Configurable thresholds (replace with config values if available)
BLOCK_THRESH = 0.99  # strict
MONITOR_THRESH = 0.90
BLOCK_BUDGET = 0.01
MONITOR_BUDGET = 0.10

PERSONAS = benign_scenarios.list_benign_personas()

@pytest.mark.parametrize("persona", PERSONAS)
def test_benign_persona_false_positive_rate(persona):
    """
    Verify that realistic benign personas are not blocked at an unacceptable rate.
    """
    try:
        features = benign_scenarios.load_persona_features(persona)
    except FileNotFoundError:
        pytest.xfail(f"Real benign persona fixture missing: add data under data/benign_personas/{persona}")
    scores = benign_scenarios.score_persona_features(features)
    rates = benign_scenarios.compute_block_monitor_pass(scores, BLOCK_THRESH, MONITOR_THRESH)
    assert rates['block_rate'] <= BLOCK_BUDGET, f"Block rate too high for {persona}: {rates['block_rate']}"
    assert rates['monitor_rate'] <= MONITOR_BUDGET, f"Monitor rate too high for {persona}: {rates['monitor_rate']}"


def test_benign_persona_schema_matches_model_features():
    """
    Ensure benign persona features match the model schema before scoring.
    """
    import json
    with open(os.path.join(os.path.dirname(__file__), '../src/clean_pipeline/feature_columns.json')) as f:
        model_columns = json.load(f)
    for persona in PERSONAS:
        features = benign_scenarios.load_persona_features(persona)
        assert benign_scenarios.check_schema_matches(features, model_columns), f"Schema mismatch for {persona}"


def test_threshold_safety_tradeoff_for_benign_personas():
    """
    Evaluate how threshold choice changes benign false blocking.
    """
    thresholds = np.linspace(0, 1, 11)
    results = []
    for persona in PERSONAS:
        features = benign_scenarios.load_persona_features(persona)
        scores = benign_scenarios.score_persona_features(features)
        for thresh in thresholds:
            block_rate = (scores >= thresh).mean()
            results.append({"persona": persona, "threshold": thresh, "block_rate": block_rate})
    # Simple assertion: block rate decreases as threshold increases
    for persona in PERSONAS:
        persona_rates = [r["block_rate"] for r in results if r["persona"] == persona]
        assert all(x >= y for x, y in zip(persona_rates, persona_rates[1:])), f"Block rate not decreasing for {persona}"


def test_session_level_false_positive_rate_for_benign_personas():
    """
    Check whether flow-level false positives become worse after session/capture aggregation.
    """
    # Add a fake session column for synthetic data
    for persona in PERSONAS:
        features = benign_scenarios.load_persona_features(persona)
        features["capture_id"] = [1, 1, 2][:len(features)]
        scores = benign_scenarios.score_persona_features(features)
        flow_block_rate = (scores >= BLOCK_THRESH).mean()
        session_scores = benign_scenarios.aggregate_session_scores(features, scores, session_col="capture_id")
        session_block_rate = (session_scores["mean"] >= BLOCK_THRESH).mean()
        assert session_block_rate <= flow_block_rate + 0.05, f"Session block rate much higher than flow for {persona}"


def audit_benign_persona_score_distributions():
    """
    Analyze VPN risk score distributions by benign persona.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    all_scores = []
    for persona in PERSONAS:
        features = benign_scenarios.load_persona_features(persona)
        scores = benign_scenarios.score_persona_features(features)
        for s in scores:
            all_scores.append({"persona": persona, "score": s})
    df = pd.DataFrame(all_scores)
    plt.figure(figsize=(8,4))
    df.boxplot(by="persona", column="score")
    plt.axhline(BLOCK_THRESH, color="red", linestyle="--", label="Block threshold")
    plt.axhline(MONITOR_THRESH, color="orange", linestyle=":", label="Monitor threshold")
    plt.title("Benign Persona Score Distributions (Synthetic)")
    plt.suptitle("")
    plt.ylabel("VPN Risk Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/validation/benign_persona_score_distributions.png")
    plt.close()