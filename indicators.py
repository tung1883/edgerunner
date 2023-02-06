
def williams_r(high, low, close, period=14):
    hh = np.array([high[max(0,i-period):i+1].max() for i in range(len(high))])
    ll = np.array([low[max(0,i-period):i+1].min()  for i in range(len(low))])
    return -100 * (hh - close) / (hh - ll + 1e-8)

def cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    ma = np.convolve(tp, np.ones(period)/period, mode='same')
    md = np.array([np.mean(np.abs(tp[max(0,i-period):i+1] - ma[i])) for i in range(len(tp))])
    return (tp - ma) / (0.015 * md + 1e-8)

def donchian_channel(high, low, period=20):
    upper = np.array([high[max(0,i-period):i+1].max() for i in range(len(high))])
    lower = np.array([low[max(0,i-period):i+1].min()  for i in range(len(low))])
    mid = (upper + lower) / 2
    return upper, lower, mid

def chaikin_money_flow(high, low, close, volume, period=20):
    clv = ((close - low) - (high - close)) / (high - low + 1e-8)
    mfv = clv * volume
    cmf = np.convolve(mfv, np.ones(period), mode='same') / (np.convolve(volume, np.ones(period), mode='same') + 1e-8)
    return cmf
