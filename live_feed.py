
import time
import yfinance as yf
import pandas as pd

def stream(ticker, interval='1m', poll_sec=60):
    while True:
        df = yf.download(ticker, period='1d', interval=interval, progress=False)
        if not df.empty:
            yield df.iloc[-1]
        time.sleep(poll_sec)

