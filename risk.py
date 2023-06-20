
import numpy as np

def var(returns, confidence=0.95):
    return np.percentile(returns, (1 - confidence) * 100)

def cvar(returns, confidence=0.95):
    v = var(returns, confidence)
    return returns[returns <= v].mean()

def calmar(equity, returns):
    ann_return = (equity.iloc[-1] / equity.iloc[0]) ** (252/len(equity)) - 1
    from backtest import max_drawdown
    mdd = abs(max_drawdown(equity))
    return ann_return / (mdd + 1e-9)

