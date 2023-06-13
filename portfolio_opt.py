
import numpy as np

def min_variance(returns):
    cov   = returns.cov().values
    n     = len(cov)
    ones  = np.ones(n)
    inv   = np.linalg.pinv(cov)
    w     = inv @ ones
    return w / w.sum()

def sharpe_weights(returns, rfr=0.0):
    mu    = returns.mean().values
    cov   = returns.cov().values
    inv   = np.linalg.pinv(cov)
    w     = inv @ (mu - rfr)
    return w / w.sum()

