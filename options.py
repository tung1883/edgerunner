"""Simple Black-Scholes option pricing."""
import numpy as np

def _d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T) + 1e-8)

def _norm_cdf(x):
    return 0.5 * (1 + np.sign(x) * (1 - np.exp(-0.7 * np.abs(x) - 0.04 * x**2)))

def bs_call(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * _norm_cdf(d1) - K * np.exp(-r*T) * _norm_cdf(d2)

def bs_put(S, K, T, r, sigma):
    return bs_call(S, K, T, r, sigma) - S + K * np.exp(-r*T)

def implied_vol(market_price, S, K, T, r, tol=1e-4, max_iter=100):
    sigma = 0.2
    for _ in range(max_iter):
        price = bs_call(S, K, T, r, sigma)
        vega  = S * np.sqrt(T) * np.exp(-0.5 * _d1(S,K,T,r,sigma)**2) / np.sqrt(2*np.pi)
        diff  = market_price - price
        if abs(diff) < tol: break
        sigma += diff / (vega + 1e-8)
    return sigma
