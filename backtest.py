
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

def max_consecutive_losses(pnl):
    best, cur = 0, 0
    for p in pnl:
        cur = cur + 1 if p < 0 else 0
        best = max(best, cur)
    return best

def win_loss_stats(pnl):
    wins  = pnl[pnl > 0]
    losses= pnl[pnl < 0]
    return {
        "win_rate": len(wins) / (len(pnl) + 1e-8),
        "avg_win":  wins.mean() if len(wins)  else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "profit_factor": wins.sum() / (-losses.sum() + 1e-8),
    }
