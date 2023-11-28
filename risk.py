
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

def beta(asset_returns, market_returns):
    import numpy as np
    cov = np.cov(asset_returns, market_returns)[0, 1]
    return cov / market_returns.var()

def alpha(asset_returns, market_returns, rfr=0.0):
    b = beta(asset_returns, market_returns)
    return asset_returns.mean() - rfr - b * (market_returns.mean() - rfr)

def stress_test(weights, scenario_returns):
    losses = []
    for scenario in scenario_returns:
        port_loss = (weights * scenario).sum()
        losses.append(port_loss)
    import numpy as np
    return {'mean_loss': float(np.mean(losses)),
            'worst_loss': float(np.min(losses))}

