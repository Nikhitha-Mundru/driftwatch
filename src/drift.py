import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def psi(expected, actual, buckets=10):
    """Population Stability Index: measure how much a feature's distribution shifted."""
    def bucket(s):
        return pd.qcut(s, q=buckets, duplicates='drop')
    
    exp_bucketed = bucket(expected)
    act_bucketed = bucket(actual)
    
    exp_counts = exp_bucketed.value_counts().sort_index() / len(expected)
    act_counts = act_bucketed.value_counts().sort_index() / len(actual)
    
    # Align indices
    idx = exp_counts.index.union(act_counts.index)
    exp_counts = exp_counts.reindex(idx, fill_value=0.001)
    act_counts = act_counts.reindex(idx, fill_value=0.001)
    
    return np.sum((act_counts - exp_counts) * np.log(act_counts / exp_counts))

def detect_drift(baseline_df, new_df, features, psi_threshold=0.2):
    """Detect drift in each feature using PSI and KS test."""
    results = {}
    for feat in features:
        psi_val = psi(baseline_df[feat], new_df[feat])
        ks_stat, ks_pval = ks_2samp(baseline_df[feat], new_df[feat])
        
        drifted = psi_val > psi_threshold or ks_pval < 0.05
        results[feat] = {
            "psi": round(psi_val, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_pval, 4),
            "drifted": drifted
        }
    return results