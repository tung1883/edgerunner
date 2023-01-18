"""Cache downloaded OHLCV to disk to avoid repeated API calls."""
import os, json, hashlib
import numpy as np

CACHE_DIR = ".data_cache"

def _key(ticker, start, end):
    return hashlib.md5(f"{ticker}{start}{end}".encode()).hexdigest()[:12]

def cache_path(ticker, start, end):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{_key(ticker,start,end)}.npz")

def save(ticker, start, end, dates, ohlcv):
    np.savez(cache_path(ticker, start, end), dates=dates, ohlcv=ohlcv)

def load(ticker, start, end):
    p = cache_path(ticker, start, end)
    if not os.path.exists(p):
        return None, None
    d = np.load(p, allow_pickle=True)
    return d["dates"], d["ohlcv"]
