"""Signal quality filters to reduce whipsaws."""
import numpy as np

def trend_filter(signal, close, ma_period=200):
    """Only take longs above 200-day MA."""
    ma = np.convolve(close, np.ones(ma_period)/ma_period, mode='same')
    filtered = signal.copy()
    filtered[(signal == 1) & (close < ma)] = 0
    return filtered

def volatility_filter(signal, returns, vol_threshold=0.03):
    """Suppress signals during high-volatility regimes."""
    vol = np.array([returns[max(0,i-20):i+1].std() for i in range(len(returns))])
    filtered = signal.copy()
    filtered[vol > vol_threshold] = 0
    return filtered

def min_holding(signal, min_bars=5):
    """Force minimum holding period to reduce churn."""
    out = signal.copy()
    last = 0
    for i in range(len(signal)):
        if signal[i] != 0 and i - last < min_bars:
            out[i] = 0
        elif signal[i] != 0:
            last = i
    return out
