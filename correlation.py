"""Correlation and diversification analytics."""
import numpy as np

def rolling_correlation(r1, r2, window=60):
    corr = []
    for i in range(len(r1)):
        w1 = r1[max(0,i-window):i+1]
        w2 = r2[max(0,i-window):i+1]
        if len(w1) < 2:
            corr.append(0.0)
            continue
        c = np.corrcoef(w1, w2)[0,1]
        corr.append(c if not np.isnan(c) else 0.0)
    return np.array(corr)

def diversification_ratio(weights, vols, corr_matrix):
    weighted_vol = weights @ vols
    port_vol = np.sqrt(weights @ (corr_matrix * np.outer(vols, vols)) @ weights)
    return weighted_vol / (port_vol + 1e-8)

def avg_pairwise_correlation(returns_matrix):
    n = returns_matrix.shape[1]
    corr = np.corrcoef(returns_matrix.T)
    mask = ~np.eye(n, dtype=bool)
    return corr[mask].mean()
