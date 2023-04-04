
import numpy as np

def from_model(model, X, threshold=0.55):
    proba = model.predict_proba(X)[:, 1]
    signal = np.zeros(len(proba))
    signal[proba > threshold]       =  1   # buy
    signal[proba < (1-threshold)]   = -1   # sell
    return signal

