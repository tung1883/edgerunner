
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


def expected_shortfall(returns, alpha=0.05):
    var = np.percentile(returns, alpha * 100)
    return returns[returns <= var].mean()

def omega_ratio(returns, threshold=0.0):
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    return gains.sum() / (losses.sum() + 1e-8)

def calmar_ratio(returns, period=252):
    ann_return = returns.mean() * period
    roll_max = np.maximum.accumulate(np.cumprod(1 + returns))
    drawdown = (np.cumprod(1 + returns) - roll_max) / roll_max
    max_dd = drawdown.min()
    return ann_return / (-max_dd + 1e-8)

def rolling_var(returns, window=252, alpha=0.05):
    var = []
    for i in range(len(returns)):
        window_ret = returns[max(0, i-window):i+1]
        var.append(np.percentile(window_ret, alpha * 100))
    return np.array(var)

def stress_test_pnl(weights, returns_matrix, shock_pct=-0.20):
    """Simulate a sudden N% market shock across all assets."""
    shocked = returns_matrix.copy()
    shocked[0] = shock_pct
    port_ret = shocked @ weights
    return port_ret.cumsum()

def tail_ratio(returns, percentile=5):
    upper = np.abs(np.percentile(returns, 100 - percentile))
    lower = np.abs(np.percentile(returns, percentile))
    return upper / (lower + 1e-8)

def gain_to_pain(returns):
    return returns.sum() / (np.abs(returns[returns < 0]).sum() + 1e-8)

def recovery_factor(equity):
    total_return = equity[-1] / equity[0] - 1
    roll_max = np.maximum.accumulate(equity)
    max_dd = ((equity - roll_max) / roll_max).min()
    return total_return / (-max_dd + 1e-8)
