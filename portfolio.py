
import pandas as pd

class Portfolio:
    def __init__(self, capital):
        self.capital  = capital
        self.positions = {}
        self.history   = []

    def buy(self, ticker, price, qty):
        cost = price * qty
        if cost > self.capital:
            return False
        self.capital -= cost
        self.positions[ticker] = self.positions.get(ticker, 0) + qty
        return True

    def sell(self, ticker, price):
        qty = self.positions.pop(ticker, 0)
        self.capital += qty * price

    def value(self, prices):
        pos_val = sum(self.positions.get(t, 0) * p for t, p in prices.items())
        return self.capital + pos_val

    def max_position_value(self, prices, max_pct=0.20):
        total = self.value(prices)
        return {t: min(self.positions.get(t,0)*p, total*max_pct)
                for t, p in prices.items()}


def rebalance_schedule(weights_history, threshold=0.05):
    """Trigger rebalance when any weight drifts beyond threshold."""
    triggers = []
    target = weights_history[0]
    for i, w in enumerate(weights_history):
        if np.any(np.abs(w - target) > threshold):
            triggers.append(i)
            target = w
    return triggers

def turnover(weights_history):
    diffs = np.diff(weights_history, axis=0)
    return np.abs(diffs).sum(axis=1)

def tax_loss_harvest(positions, prices, cost_basis, threshold=-0.05):
    """Flag positions with unrealised loss below threshold for harvesting."""
    candidates = []
    for ticker, qty in positions.items():
        if ticker not in prices or ticker not in cost_basis:
            continue
        pnl_pct = (prices[ticker] - cost_basis[ticker]) / cost_basis[ticker]
        if pnl_pct < threshold:
            candidates.append({"ticker": ticker, "qty": qty, "pnl_pct": pnl_pct})
    return sorted(candidates, key=lambda x: x["pnl_pct"])
