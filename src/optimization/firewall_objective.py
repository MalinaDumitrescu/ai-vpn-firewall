import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_firewall_score(
    df_val: pd.DataFrame,
    prob_col: str = "prob",
    label_col: str = "label",
    group_col: str = "capture_id",
    block_fpr_target: float = 0.0,
    flagged_fpr_target: float = 0.05,
) -> float:
    """
    Computes the custom firewall score based on session-level aggregation.
    
    Score = 100 * FLAGGED_RECALL + 10 * BLOCK_RECALL - FLAGGED_FPR
    
    Constraints:
    - BLOCK_FPR must be <= block_fpr_target (hard constraint).
    - If BLOCK_FPR > block_fpr_target, returns -1e9.
    """
    # 1. Aggregate to Session Level (Mean Probability)
    session_df = df_val.groupby(group_col).agg({
        prob_col: "mean",
        label_col: "max"  # If any flow in session is VPN, session is VPN (usually all are same)
    }).reset_index()
    
    y_true = session_df[label_col].values
    y_prob = session_df[prob_col].values
    
    # Separate benign and VPN sessions
    benign_mask = (y_true == 0)
    vpn_mask = (y_true == 1)
    
    benign_probs = y_prob[benign_mask]
    vpn_probs = y_prob[vpn_mask]
    
    if len(benign_probs) == 0 or len(vpn_probs) == 0:
        return -1e9  # Invalid split for evaluation
        
    # 2. Find Thresholds
    # T_BLOCK: Threshold where FPR <= block_fpr_target (usually 0)
    # We want the highest threshold that satisfies the condition.
    # Sort benign probs descending
    benign_probs_sorted = np.sort(benign_probs)[::-1]
    
    # If target is 0, threshold must be > max(benign_probs)
    if block_fpr_target == 0:
        if len(benign_probs) > 0:
            t_block = benign_probs.max() + 1e-6 # Slightly above max benign
            if t_block > 1.0: t_block = 1.0 # Clamp if max is 1.0 (impossible to block 0 FP then)
        else:
            t_block = 0.5
    else:
        # Allow some FPs
        n_allowed = int(len(benign_probs) * block_fpr_target)
        if n_allowed == 0:
             t_block = benign_probs.max() + 1e-6
        else:
             t_block = benign_probs_sorted[n_allowed - 1]

    # T_MONITOR: Threshold where FPR <= flagged_fpr_target
    n_allowed_monitor = int(len(benign_probs) * flagged_fpr_target)
    if n_allowed_monitor == 0:
        t_monitor = benign_probs.max() + 1e-6
    else:
        # If we allow 5%, we pick the score at the 5th percentile from top
        t_monitor = benign_probs_sorted[n_allowed_monitor - 1]
        
    # Ensure monotonicity: Block threshold >= Monitor threshold
    if t_block < t_monitor:
        t_block = t_monitor

    # 3. Compute Metrics
    # Block Zone
    block_preds = (y_prob >= t_block).astype(int)
    block_tp = np.sum((block_preds == 1) & (y_true == 1))
    block_fp = np.sum((block_preds == 1) & (y_true == 0))
    
    block_recall = block_tp / len(vpn_probs)
    block_fpr = block_fp / len(benign_probs)
    
    # Flagged Zone (Monitor + Block)
    flagged_preds = (y_prob >= t_monitor).astype(int)
    flagged_tp = np.sum((flagged_preds == 1) & (y_true == 1))
    flagged_fp = np.sum((flagged_preds == 1) & (y_true == 0))
    
    flagged_recall = flagged_tp / len(vpn_probs)
    flagged_fpr = flagged_fp / len(benign_probs)
    
    # 4. Scoring Function
    # Hard Constraint Check
    if block_fpr > block_fpr_target:
        return -1e9
        
    # Objective
    # We want high flagged recall (catch everything)
    # We want high block recall (block as much as possible safely)
    # We penalize flagged FPR (keep noise low)
    score = (100 * flagged_recall) + (10 * block_recall) - (flagged_fpr)
    
    return score
