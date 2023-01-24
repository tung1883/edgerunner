import pandas as pd
from indicators import sma, rsi

def sma_crossover(df, fast=20, slow=50):
    close = df["Close"]
    fast_ma = sma(close, fast)
    slow_ma = sma(close, slow)
    signal = (fast_ma > slow_ma).astype(int)
    return signal.diff().fillna(0)

def rsi_strategy(df, low=30, high=70):
    r = rsi(df["Close"])
    buy  = (r < low).astype(int)
    sell = (r > high).astype(int)
    return buy - sell
