
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

def hierarchical_risk_parity(cov):
    """Simple HRP — cluster by correlation, allocate by inverse variance."""
    n = cov.shape[0]
    vols = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vols, vols)
    dist = np.sqrt((1 - corr) / 2)
    # single-linkage sort (simplified)
    order = list(range(n))
    order.sort(key=lambda i: dist[i].sum())
    weights = 1.0 / (vols[order] + 1e-8)
    weights /= weights.sum()
    final = np.zeros(n)
    for rank, idx in enumerate(order):
        final[idx] = weights[rank]
    return final

def risk_parity_newton(cov, tol=1e-8, max_iter=1000):
    """Faster risk parity via Newton's method."""
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        sigma = np.sqrt(w @ cov @ w)
        mrc = cov @ w / sigma
        rc  = w * mrc
        target = sigma / n
        grad = rc - target
        hess = np.diag(mrc) + np.outer(w, cov.sum(axis=1)) / sigma
        step = np.linalg.solve(hess + 1e-6*np.eye(n), grad)
        w -= step
        w = np.abs(w); w /= w.sum()
        if np.linalg.norm(grad) < tol:
            break
    return w
