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
