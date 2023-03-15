"""Walk-forward cross-validation with expanding window."""
import numpy as np

def walk_forward_cv(model_factory, X, y, n_splits=5, min_train=252):
    n = len(X)
    step = (n - min_train) // n_splits
    scores = []
    for i in range(n_splits):
        t = min_train + i * step
        model = model_factory()
        model.fit(X[:t], y[:t])
        preds = model.predict(X[t:t+step])
        acc = np.mean(preds == y[t:t+step])
        scores.append(acc)
    return scores
