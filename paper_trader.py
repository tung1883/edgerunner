
import pandas as pd
from portfolio import Portfolio

class PaperTrader:
    def __init__(self, capital=10000):
        self.portfolio = Portfolio(capital)
        self.log = []

    def on_signal(self, ticker, signal, price):
        if signal > 0:
            qty = int(self.portfolio.capital * 0.1 / price)
            if self.portfolio.buy(ticker, price, qty):
                self.log.append({'action': 'BUY', 'ticker': ticker,
                                  'price': price, 'qty': qty})
        elif signal < 0:
            self.portfolio.sell(ticker, price)
            self.log.append({'action': 'SELL', 'ticker': ticker, 'price': price})
    def export(self, path='trades.csv'):
        import pandas as pd
        pd.DataFrame(self.log).to_csv(path, index=False)


class RiskGate:
    """Block orders that would breach risk limits."""
    def __init__(self, max_drawdown=0.15, daily_loss_limit=0.02):
        self.max_dd = max_drawdown
        self.daily_limit = daily_loss_limit
        self.peak_equity = None
        self.day_start = None

    def check(self, equity):
        if self.peak_equity is None:
            self.peak_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        dd = (self.peak_equity - equity) / self.peak_equity
        if dd > self.max_dd:
            return False, f"max drawdown breached ({dd:.1%})"
        return True, "ok"
