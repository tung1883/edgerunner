
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

