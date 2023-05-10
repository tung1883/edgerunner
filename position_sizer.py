"""Position sizing algorithms."""
import numpy as np

def kelly_fraction(win_rate, avg_win, avg_loss):
    b = avg_win / (avg_loss + 1e-8)
    return win_rate - (1 - win_rate) / b

def volatility_targeting(signal, returns, target_vol=0.10, period=252):
    """Scale positions so portfolio hits target annualised vol."""
    vol = np.array([returns[max(0,i-20):i+1].std() * np.sqrt(period) for i in range(len(returns))])
    scale = target_vol / (vol + 1e-8)
    return signal * np.clip(scale, 0, 2.0)

def equal_risk_contribution(cov_matrix):
    n = cov_matrix.shape[0]
    w = np.ones(n) / n
    for _ in range(100):
        sigma = np.sqrt(w @ cov_matrix @ w)
        mrc = cov_matrix @ w / sigma
        w = w / (mrc + 1e-8)
        w /= w.sum()
    return w
