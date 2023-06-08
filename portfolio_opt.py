
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


def black_litterman(prior_returns, cov, P, Q, omega=None, tau=0.05):
    """Black-Litterman posterior estimate."""
    n = len(prior_returns)
    if omega is None:
        omega = np.diag(np.diag(P @ (tau * cov) @ P.T))
    M1 = np.linalg.inv(tau * cov)
    M2 = P.T @ np.linalg.inv(omega) @ P
    post_cov = np.linalg.inv(M1 + M2)
    post_ret = post_cov @ (M1 @ prior_returns + P.T @ np.linalg.inv(omega) @ Q)
    return post_ret, post_cov
