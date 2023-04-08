
def mean_reversion_signal(close, window=20, threshold=1.5):
    z = (close - np.convolve(close, np.ones(window)/window, mode='same')) / (close.std() + 1e-8)
    sig = np.zeros(len(close))
    sig[z < -threshold] = 1
    sig[z > threshold] = -1
    return sig

def breakout_signal(close, high, low, period=20):
    roll_high = np.array([high[max(0,i-period):i].max() if i > 0 else high[0] for i in range(len(high))])
    roll_low  = np.array([low[max(0,i-period):i].min()  if i > 0 else low[0]  for i in range(len(low))])
    sig = np.zeros(len(close))
    sig[close > roll_high] = 1
    sig[close < roll_low]  = -1
    return sig

def dual_momentum(returns_asset, returns_bench, lookback=252):
    """Gary Antonacci dual momentum — absolute + relative."""
    abs_mom = np.convolve(returns_asset, np.ones(lookback), mode='same')
    rel_mom = abs_mom - np.convolve(returns_bench, np.ones(lookback), mode='same')
    sig = np.zeros(len(returns_asset))
    sig[(abs_mom > 0) & (rel_mom > 0)] = 1
    return sig
