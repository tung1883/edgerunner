"""Cleaning and alignment for raw OHLCV data."""
import numpy as np

def remove_outliers(prices, z_thresh=5.0):
    log_ret = np.diff(np.log(prices + 1e-8))
    z = (log_ret - log_ret.mean()) / (log_ret.std() + 1e-8)
    mask = np.abs(z) < z_thresh
    clean = prices[1:].copy()
    clean[~mask] = np.nan
    return np.interp(np.arange(len(clean)), np.where(mask)[0], clean[mask])

def align_dates(dates_a, prices_a, dates_b, prices_b):
    common = sorted(set(dates_a) & set(dates_b))
    idx_a = [list(dates_a).index(d) for d in common]
    idx_b = [list(dates_b).index(d) for d in common]
    return prices_a[idx_a], prices_b[idx_b], common

def forward_fill(prices):
    out = prices.copy().astype(float)
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i-1]
    return out
