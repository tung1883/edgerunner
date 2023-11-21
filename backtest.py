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

def run_with_stops(df, signals, stop_pct=0.05, tp_pct=0.10):
    from config import INITIAL_CAPITAL, COMMISSION
    capital  = INITIAL_CAPITAL
    position = 0
    entry    = 0.0
    portfolio = []
    for date, sig in signals.items():
        price = df.loc[date, 'Close']
        if position > 0:
            if price <= entry * (1 - stop_pct):
                capital  += position * price * (1 - COMMISSION)
                position  = 0
            elif price >= entry * (1 + tp_pct):
                capital  += position * price * (1 - COMMISSION)
                position  = 0
        if sig > 0 and position == 0:
            position = capital / price
            capital -= position * price * (1 + COMMISSION)
            entry    = price
        elif sig < 0 and position > 0:
            capital += position * price * (1 - COMMISSION)
            position = 0
        portfolio.append(capital + position * price)
    import pandas as pd
    return pd.Series(portfolio, index=signals.index)

def trailing_stop(df, signals, trail_pct=0.07):
    from config import INITIAL_CAPITAL, COMMISSION
    capital  = INITIAL_CAPITAL
    position = 0
    peak     = 0.0
    portfolio = []
    for date, sig in signals.items():
        price = df.loc[date, 'Close']
        if position > 0:
            peak = max(peak, price)
            if price < peak * (1 - trail_pct):
                capital  += position * price * (1 - COMMISSION)
                position  = 0
                peak      = 0.0
        if sig > 0 and position == 0:
            position = capital / price
            capital -= position * price * (1 + COMMISSION)
            peak     = price
        elif sig < 0 and position > 0:
            capital  += position * price * (1 - COMMISSION)
            position  = 0
            peak      = 0.0
        portfolio.append(capital + position * price)
    import pandas as pd
    return pd.Series(portfolio, index=signals.index)

