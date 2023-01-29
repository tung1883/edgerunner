import numpy as np
import pandas as pd
from config import INITIAL_CAPITAL, COMMISSION

def run(df, signals):
    capital = INITIAL_CAPITAL
    position = 0
    portfolio = []

    for i, (date, sig) in enumerate(signals.items()):
        price = df.loc[date, "Close"]
        if sig > 0 and position == 0:
            position = capital / price
            capital -= position * price * (1 + COMMISSION)
        elif sig < 0 and position > 0:
            capital += position * price * (1 - COMMISSION)
            position = 0
        portfolio.append(capital + position * price)

    return pd.Series(portfolio, index=signals.index)

def sharpe(returns, rfr=0.0):
    excess = returns - rfr
    return excess.mean() / (excess.std() + 1e-9) * np.sqrt(252)

def max_drawdown(equity):
    peak = equity.cummax()
    return ((equity - peak) / peak).min()
