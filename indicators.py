import pandas as pd
import numpy as np

def sma(series, window):
    return series.rolling(window).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    e1 = ema(series, fast)
    e2 = ema(series, slow)
    macd_line = e1 - e2
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line

def bollinger(series, window=20, std=2):
    mid = sma(series, window)
    band = series.rolling(window).std()
    return mid - std*band, mid, mid + std*band
def atr(df, period=14):
    import numpy as np
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low']  - df['Close'].shift()).abs()
    tr = hl.combine(hc, max).combine(lc, max)
    return tr.rolling(period).mean()

def stochastic(df, k=14, d=3):
    low_min  = df['Low'].rolling(k).min()
    high_max = df['High'].rolling(k).max()
    k_pct = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-9)
    return k_pct, k_pct.rolling(d).mean()
def obv(df):
    import numpy as np
    direction = np.sign(df['Close'].diff())
    return (direction * df['Volume']).cumsum()

def vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * df['Volume']).cumsum() / df['Volume'].cumsum()

