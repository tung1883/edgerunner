
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

