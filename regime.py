
import pandas as pd
import numpy as np

def detect(returns, window=20, threshold=0.015):
    vol = returns.rolling(window).std()
    regime = pd.Series('normal', index=returns.index)
    regime[vol > threshold] = 'high_vol'
    regime[vol < threshold * 0.5] = 'low_vol'
    return regime

