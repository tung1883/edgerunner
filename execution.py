"""Execution algorithms to minimise market impact."""
import numpy as np

def twap(total_qty, n_slices, prices):
    """Time-Weighted Average Price — equal slices."""
    slice_qty = total_qty // n_slices
    fills = []
    for i in range(n_slices):
        fills.append({"qty": slice_qty, "price": prices[i % len(prices)]})
    return fills

def vwap_target(total_qty, volume_profile, prices):
    """Slice order proportional to expected volume profile."""
    weights = volume_profile / (volume_profile.sum() + 1e-8)
    fills = []
    for i, w in enumerate(weights):
        qty = int(total_qty * w)
        if qty > 0:
            fills.append({"qty": qty, "price": prices[i % len(prices)]})
    return fills

def pov(total_qty, market_volumes, pov_rate=0.10):
    """Percentage of Volume — trade at fixed fraction of volume."""
    fills = []
    remaining = total_qty
    for vol in market_volumes:
        if remaining <= 0: break
        qty = min(int(vol * pov_rate), remaining)
        fills.append({"qty": qty})
        remaining -= qty
    return fills
