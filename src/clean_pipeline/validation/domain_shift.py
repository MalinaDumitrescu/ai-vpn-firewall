import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

def compute_jsd(p, q, bins=50):
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    # Add small constant to avoid zero
    p_hist = p_hist + 1e-8
    q_hist = q_hist + 1e-8
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return jensenshannon(p_hist, q_hist, base=2)

def compute_ks(p, q):
    return ks_2samp(p, q).statistic

def compute_smd(x1, x0):
    # Standardized mean difference
    m1, m0 = np.mean(x1), np.mean(x0)
    s1, s0 = np.std(x1), np.std(x0)
    pooled = np.sqrt((s1**2 + s0**2) / 2)
    if pooled == 0:
        return 0.0
    return (m1 - m0) / pooled
