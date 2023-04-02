
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
