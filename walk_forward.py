
import pandas as pd
import numpy as np

def walk_forward(df, model_fn, n_splits=5):
    n = len(df)
    fold = n // n_splits
    results = []
    for i in range(1, n_splits):
        train = df.iloc[:i * fold]
        test  = df.iloc[i * fold:(i+1) * fold]
        model = model_fn(train)
        results.append(model(test))
    return results

def purged_kfold(df, n_splits=5, gap=5):
    n = len(df)
    fold = n // n_splits
    folds = []
    for i in range(n_splits):
        test_start = i * fold
        test_end   = (i+1) * fold
        train_idx  = list(range(0, max(0, test_start - gap))) +                      list(range(min(n, test_end + gap), n))
        test_idx   = list(range(test_start, test_end))
        folds.append((train_idx, test_idx))
    return folds


def anchored_walk_forward(model_factory, X, y, start_pct=0.4):
    """Expanding-window walk forward (anchored at start)."""
    n = len(X)
    start = int(n * start_pct)
    results = []
    step = max(1, (n - start) // 20)
    for t in range(start, n - step, step):
        m = model_factory()
        m.fit(X[:t], y[:t])
        preds = m.predict(X[t:t+step])
        acc = np.mean(preds == y[t:t+step])
        results.append(acc)
    return results
