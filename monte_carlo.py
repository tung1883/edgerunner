
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


def bootstrap_sharpe(returns, n_bootstrap=1000):
    sharpes = []
    n = len(returns)
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=n, replace=True)
        s = sample.mean() / (sample.std() + 1e-8) * np.sqrt(252)
        sharpes.append(s)
    return np.percentile(sharpes, [5, 50, 95])

def path_dependent_var(returns, weights, n_sim=5000, horizon=10):
    paths = np.array([np.prod(1 + np.random.choice(returns, size=horizon, replace=True))
                      for _ in range(n_sim)])
    return np.percentile(paths - 1, 5)
