
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

