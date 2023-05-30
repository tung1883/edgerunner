
import numpy as np

def make_sequences(arr, seq_len=30):
    X, y = [], []
    for i in range(seq_len, len(arr)):
        X.append(arr[i-seq_len:i])
        y.append(arr[i])
    return np.array(X), np.array(y)

def scale(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9), mn, mx

def unscale(arr, mn, mx):
    return arr * (mx - mn) + mn
SEQ_LEN = 60  # tuned from 30 — captures longer momentum patterns

