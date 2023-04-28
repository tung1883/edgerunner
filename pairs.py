"""Pairs trading — cointegration-based mean reversion."""
import numpy as np

def cointegration_residual(price_a, price_b):
    """OLS hedge ratio and residual spread."""
    X = np.column_stack([price_b, np.ones(len(price_b))])
    beta, alpha = np.linalg.lstsq(X, price_a, rcond=None)[0]
    spread = price_a - beta * price_b - alpha
    return spread, beta, alpha

def pairs_signal(spread, entry_z=2.0, exit_z=0.5):
    z = (spread - spread.mean()) / (spread.std() + 1e-8)
    sig = np.zeros(len(spread))
    pos = 0
    for i in range(len(spread)):
        if pos == 0:
            if z[i] > entry_z:  pos = -1
            elif z[i] < -entry_z: pos = 1
        elif pos == 1  and z[i] >= -exit_z: pos = 0
        elif pos == -1 and z[i] <=  exit_z: pos = 0
        sig[i] = pos
    return sig

def johansen_cointegration_simple(price_a, price_b):
    """Check stationarity of spread via ADF-proxy."""
    spread, beta, alpha = cointegration_residual(price_a, price_b)
    dspread = np.diff(spread)
    lag_spread = spread[:-1]
    ols_X = np.column_stack([lag_spread, np.ones(len(lag_spread))])
    coef = np.linalg.lstsq(ols_X, dspread, rcond=None)[0]
    adf_stat = coef[0] / (dspread.std() / np.sqrt(len(dspread)) + 1e-8)
    return adf_stat, spread
