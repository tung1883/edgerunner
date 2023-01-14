import yfinance as yf
import pandas as pd
from config import TICKERS, START_DATE, END_DATE

def fetch(ticker):
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
    df.dropna(inplace=True)
    return df

def fetch_all():
    return {t: fetch(t) for t in TICKERS}
