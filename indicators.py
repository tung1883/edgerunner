
def williams_r(high, low, close, period=14):
    hh = np.array([high[max(0,i-period):i+1].max() for i in range(len(high))])
    ll = np.array([low[max(0,i-period):i+1].min()  for i in range(len(low))])
    return -100 * (hh - close) / (hh - ll + 1e-8)

def cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    ma = np.convolve(tp, np.ones(period)/period, mode='same')
    md = np.array([np.mean(np.abs(tp[max(0,i-period):i+1] - ma[i])) for i in range(len(tp))])
    return (tp - ma) / (0.015 * md + 1e-8)
