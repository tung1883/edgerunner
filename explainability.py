"""Model explainability — SHAP-style permutation importance."""
import numpy as np

def permutation_importance(model, X, y, n_repeats=10):
    baseline = np.mean(model.predict(X) == y)
    importances = np.zeros(X.shape[1])
    for col in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, col])
            scores.append(np.mean(model.predict(X_perm) == y))
        importances[col] = baseline - np.mean(scores)
    return importances

def partial_dependence(model, X, feature_idx, n_grid=20):
    grid = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), n_grid)
    pdp = []
    for val in grid:
        X_mod = X.copy()
        X_mod[:, feature_idx] = val
        pdp.append(model.predict(X_mod).mean())
    return grid, np.array(pdp)
