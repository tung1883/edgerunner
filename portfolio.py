
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

