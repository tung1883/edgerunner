"""Dynamic universe selection based on liquidity and volume."""
import numpy as np

def liquidity_filter(volumes, min_avg_volume=1_000_000, window=20):
    avg_vol = np.array([volumes[:,i] for i in range(volumes.shape[1])])
    avg_vol_ma = np.array([np.convolve(v, np.ones(window)/window, mode='same') for v in avg_vol])
    return np.where(avg_vol_ma.mean(axis=1) >= min_avg_volume)[0]

def momentum_universe(returns, top_n=20):
    """Select top-N by 12-1 month momentum."""
    mom = returns[-252:-21].sum(axis=0) if len(returns) > 252 else returns.sum(axis=0)
    return np.argsort(mom)[-top_n:]
