
import pandas as pd
import numpy as np

def detect(returns, window=20, threshold=0.015):
    vol = returns.rolling(window).std()
    regime = pd.Series('normal', index=returns.index)
    regime[vol > threshold] = 'high_vol'
    regime[vol < threshold * 0.5] = 'low_vol'
    return regime


def hmm_regime(returns, n_states=2, n_iter=50):
    """Baum-Welch EM for a simple Gaussian HMM."""
    T = len(returns)
    mu = np.percentile(returns, [30, 70])
    sigma = np.array([returns.std()] * n_states)
    trans = np.full((n_states, n_states), 1/n_states)
    pi = np.full(n_states, 1/n_states)

    for _ in range(n_iter):
        # E-step
        emit = np.exp(-0.5 * ((returns[:,None] - mu) / sigma)**2) / (sigma * np.sqrt(2*np.pi))
        alpha = np.zeros((T, n_states))
        alpha[0] = pi * emit[0]
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ trans) * emit[t]
            alpha[t] /= alpha[t].sum() + 1e-300
        # M-step (simplified)
        states = alpha.argmax(axis=1)
        for s in range(n_states):
            mask = states == s
            if mask.sum() > 0:
                mu[s] = returns[mask].mean()
                sigma[s] = returns[mask].std() + 1e-6
    return alpha.argmax(axis=1)

def rolling_regime(returns, window=63):
    """Label each day as bull/bear/sideways based on rolling stats."""
    labels = np.zeros(len(returns), dtype=int)
    for i in range(window, len(returns)):
        r = returns[i-window:i]
        ann_ret = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        if ann_ret > 0.10:
            labels[i] = 2   # bull
        elif ann_ret < -0.10:
            labels[i] = 0   # bear
        else:
            labels[i] = 1   # sideways
    return labels
