
import pandas as pd
from indicators import sma, ema, rsi, macd, bollinger, atr

def build(df):
    close = df['Close']
    feats = pd.DataFrame(index=df.index)
    feats['sma_20']   = sma(close, 20)
    feats['sma_50']   = sma(close, 50)
    feats['ema_12']   = ema(close, 12)
    feats['rsi_14']   = rsi(close)
    feats['macd'], feats['macd_sig'] = macd(close)
    feats['bb_lo'], feats['bb_mid'], feats['bb_hi'] = bollinger(close)
    feats['atr']      = atr(df)
    feats['returns']  = close.pct_change()
    feats['vol_20']   = feats['returns'].rolling(20).std()
    feats.dropna(inplace=True)
    return feats
def normalise(df):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index), scaler
def split(df, test_ratio=0.2):
    n = int(len(df) * (1 - test_ratio))
    return df.iloc[:n], df.iloc[n:]


def lag_features(series, lags=(1, 2, 3, 5, 10)):
    out = {}
    for l in lags:
        out[f"lag_{l}"] = np.roll(series, l)
    return out

def rolling_zscore(series, window=20):
    mu = np.convolve(series, np.ones(window)/window, mode='same')
    diff = series - mu
    sigma = np.sqrt(np.convolve(diff**2, np.ones(window)/window, mode='same'))
    return np.divide(diff, sigma, where=sigma > 1e-8)

def price_momentum(close, periods=(5, 10, 20)):
    return {f"mom_{p}": close / np.roll(close, p) - 1 for p in periods}

def volume_ratio(volume, window=20):
    avg_vol = np.convolve(volume, np.ones(window)/window, mode='same')
    return np.divide(volume, avg_vol, where=avg_vol > 0)

def gap_feature(open_, prev_close):
    return (open_ - prev_close) / (prev_close + 1e-8)

def earnings_proximity(dates, earnings_dates, window=5):
    """Binary feature: within `window` days of earnings."""
    feat = np.zeros(len(dates))
    for i, d in enumerate(dates):
        for ed in earnings_dates:
            if abs((d - ed).days) <= window:
                feat[i] = 1
                break
    return feat

def day_of_week(dates):
    return np.array([d.weekday() for d in dates])

def month_of_year(dates):
    return np.array([d.month for d in dates])
