
import numpy as np
import pandas as pd

def simulate(returns, weights, n_sims=1000, horizon=252):
    mu  = returns.mean().values
    cov = returns.cov().values
    results = []
    for _ in range(n_sims):
        daily = np.random.multivariate_normal(mu, cov, horizon)
        port  = (daily * weights).sum(axis=1)
        results.append(np.cumprod(1 + port)[-1])
    return np.array(results)

