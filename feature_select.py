"""Feature selection — mutual information and recursive elimination."""
import numpy as np

def mutual_information(x, y, bins=10):
    xy, _, _ = np.histogram2d(x, y, bins=bins)
    px  = xy.sum(axis=1, keepdims=True) / xy.sum()
    py  = xy.sum(axis=0, keepdims=True) / xy.sum()
    pxy = xy / xy.sum()
    mask = pxy > 0
    return (pxy[mask] * np.log(pxy[mask] / (px * py)[mask])).sum()

def select_k_best(X, y, k=20):
    scores = [mutual_information(X[:,i], y) for i in range(X.shape[1])]
    return np.argsort(scores)[-k:]

def variance_threshold(X, threshold=0.01):
    return np.where(X.var(axis=0) > threshold)[0]
