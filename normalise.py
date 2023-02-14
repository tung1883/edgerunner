"""Feature normalisation helpers."""
import numpy as np

def min_max(x, eps=1e-8):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + eps)

def robust_scale(x):
    med = np.median(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    return (x - med) / (iqr + 1e-8)

def winsorise(x, low=1, high=99):
    lo, hi = np.percentile(x, low), np.percentile(x, high)
    return np.clip(x, lo, hi)
