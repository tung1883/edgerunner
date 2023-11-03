
import numpy as np

def average(preds_list):
    return np.mean(preds_list, axis=0)

def majority_vote(preds_list):
    stacked = np.stack(preds_list, axis=1)
    return (np.sum(stacked, axis=1) > len(preds_list) / 2).astype(int)


class StackedEnsemble:
    """Layer 1: base learners. Layer 2: meta-learner on OOF predictions."""
    def __init__(self, base_models, meta_model):
        self.bases = base_models
        self.meta  = meta_model

    def fit(self, X, y, cv_splits=5):
        n = len(X)
        oof = np.zeros((n, len(self.bases)))
        step = n // cv_splits
        for fold in range(cv_splits):
            val_idx = range(fold*step, (fold+1)*step)
            tr_idx  = [i for i in range(n) if i not in val_idx]
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val = X[list(val_idx)]
            for j, m in enumerate(self.bases):
                m.fit(X_tr, y_tr)
                oof[list(val_idx), j] = m.predict(X_val)
        for m in self.bases:
            m.fit(X, y)
        self.meta.fit(oof, y)

    def predict(self, X):
        base_preds = np.column_stack([m.predict(X) for m in self.bases])
        return self.meta.predict(base_preds)
