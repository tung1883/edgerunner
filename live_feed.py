
import time
import yfinance as yf
import pandas as pd

def stream(ticker, interval='1m', poll_sec=60):
    while True:
        df = yf.download(ticker, period='1d', interval=interval, progress=False)
        if not df.empty:
            yield df.iloc[-1]
        time.sleep(poll_sec)


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    def __init__(self, calls_per_minute=30):
        import time
        self.interval = 60.0 / calls_per_minute
        self.last = 0.0

    def wait(self):
        import time
        now = time.time()
        wait = self.interval - (now - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()

def stream_quotes(tickers, callback, poll_interval=60):
    """Poll Yahoo Finance and invoke callback on new data."""
    rate = RateLimiter(calls_per_minute=10)
    while True:
        for ticker in tickers:
            rate.wait()
            # In production: fetch real quote here
            quote = {"ticker": ticker, "price": 0.0, "volume": 0}
            callback(quote)
