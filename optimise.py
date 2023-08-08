
import itertools
import pandas as pd

def grid_search(df, fast_range, slow_range, metric_fn):
    best_score = -float('inf')
    best_params = None
    for fast, slow in itertools.product(fast_range, slow_range):
        if fast >= slow:
            continue
        score = metric_fn(df, fast, slow)
        if score > best_score:
            best_score = score
            best_params = (fast, slow)
    return best_params, best_score

