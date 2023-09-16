"""Simulated broker with commissions and partial fills."""
import numpy as np

COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.0

class SimBroker:
    def __init__(self, initial_cash=100_000):
        self.cash = initial_cash
        self.positions = {}
        self.trades = []

    def commission(self, qty): return max(abs(qty) * COMMISSION_PER_SHARE, MIN_COMMISSION)

    def buy(self, ticker, qty, price):
        cost = qty * price + self.commission(qty)
        if cost > self.cash:
            qty = int((self.cash - MIN_COMMISSION) / (price + COMMISSION_PER_SHARE))
        if qty <= 0: return 0
        self.cash -= qty * price + self.commission(qty)
        self.positions[ticker] = self.positions.get(ticker, 0) + qty
        self.trades.append({"ticker": ticker, "qty": qty, "price": price, "side": "buy"})
        return qty

    def sell(self, ticker, qty, price):
        qty = min(qty, self.positions.get(ticker, 0))
        if qty <= 0: return 0
        self.cash += qty * price - self.commission(qty)
        self.positions[ticker] -= qty
        self.trades.append({"ticker": ticker, "qty": qty, "price": price, "side": "sell"})
        return qty

    def equity(self, prices):
        pos_val = sum(self.positions.get(t, 0) * p for t, p in prices.items())
        return self.cash + pos_val
