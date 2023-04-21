
def slippage_model(price, signal, slippage_bps=5):
    """Apply per-trade slippage in basis points."""
    adj = price.copy().astype(float)
    trades = np.diff(signal, prepend=0) != 0
    adj[trades & (signal == 1)]  *= (1 + slippage_bps / 10000)
    adj[trades & (signal == -1)] *= (1 - slippage_bps / 10000)
    return adj

def transaction_cost_pnl(returns, signal, cost_bps=10):
    """Subtract transaction costs on every signal change."""
    trades = np.abs(np.diff(signal, prepend=0))
    cost = trades * cost_bps / 10000
    return returns - cost
