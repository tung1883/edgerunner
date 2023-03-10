
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_lr(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def labels(df, horizon=1):
    return (df['Close'].shift(-horizon) > df['Close']).astype(int)
from sklearn.ensemble import RandomForestClassifier

def train_rf(X_train, y_train, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model
from sklearn.ensemble import GradientBoostingClassifier

def train_gb(X_train, y_train):
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                        max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_rf_regularised(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50, max_depth=4,
                                    min_samples_leaf=20, random_state=42)
    model.fit(X_train, y_train)
    return model

def sharpe_score(y_true, y_pred):
    rets = y_true * y_pred
    return rets.mean() / (rets.std() + 1e-9) * (252 ** 0.5)


class RidgeRegressor:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.w = None

    def fit(self, X, y):
        n, d = X.shape
        I = np.eye(d)
        self.w = np.linalg.solve(X.T @ X + self.alpha * I, X.T @ y)

    def predict(self, X):
        return X @ self.w

class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=500, C=1.0):
        self.lr, self.n_iter, self.C = lr, n_iter, C
        self.w = None

    def _sigmoid(self, z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        for _ in range(self.n_iter):
            p = self._sigmoid(X @ self.w)
            grad = X.T @ (p - y) / len(y) + self.w / self.C
            self.w -= self.lr * grad

    def predict_proba(self, X): return self._sigmoid(X @ self.w)
    def predict(self, X): return (self.predict_proba(X) >= 0.5).astype(int)

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train, self.y_train = X, y

    def predict(self, X):
        out = []
        for x in X:
            dists = np.sum((self.X_train - x)**2, axis=1)
            idx = np.argsort(dists)[:self.k]
            out.append(np.bincount(self.y_train[idx].astype(int)).argmax())
        return np.array(out)
