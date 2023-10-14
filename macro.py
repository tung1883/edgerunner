"""Macro indicator proxies derived from market data."""
import numpy as np

def yield_curve_proxy(long_rate, short_rate):
    """Positive = normal, negative = inverted (recession signal)."""
    return long_rate - short_rate

def credit_spread(corp_yield, treasury_yield):
    return corp_yield - treasury_yield

def fear_greed(vix, put_call_ratio, momentum):
    """Composite fear/greed index [0=fear, 100=greed]."""
    vix_norm   = 1 - np.clip((vix - 10) / 40, 0, 1)
    pc_norm    = 1 - np.clip((put_call_ratio - 0.5) / 1.0, 0, 1)
    mom_norm   = np.clip(momentum / 0.10 * 0.5 + 0.5, 0, 1)
    return 100 * (vix_norm + pc_norm + mom_norm) / 3

def economic_surprise(actual, expected):
    return (actual - expected) / (np.abs(expected) + 1e-8)
