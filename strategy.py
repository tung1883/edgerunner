import pandas as pd
from indicators import sma, rsi

def sma_crossover(df, fast=20, slow=50):
    close = df["Close"]
    fast_ma = sma(close, fast)
    slow_ma = sma(close, slow)
    signal = (fast_ma > slow_ma).astype(int)
    return signal.diff().fillna(0)

def rsi_strategy(df, low=30, high=70):
    r = rsi(df["Close"])
    buy  = (r < low).astype(int)
    sell = (r > high).astype(int)
    return buy - sell

def regime_adaptive(df, regime, aggressive_fast=10, aggressive_slow=30,
                    defensive_fast=30, defensive_slow=100):
    from indicators import sma
    close = df['Close']
    signal = sma(close, aggressive_fast) - sma(close, aggressive_slow)
    def_signal = sma(close, defensive_fast) - sma(close, defensive_slow)
    combined = signal.copy()
    combined[regime == 'high_vol'] = def_signal[regime == 'high_vol']
    return (combined > 0).astype(int).diff().fillna(0)

def momentum(df, formation=252, skip=21):
    close = df['Close']
    past_return = close.shift(skip) / close.shift(formation) - 1
    return (past_return > past_return.median()).astype(int).diff().fillna(0)

def mean_reversion(df, window=30, z_entry=1.5, z_exit=0.5):
    close = df['Close']
    z = (close - close.rolling(window).mean()) / close.rolling(window).std()
    signal = pd.Series(0, index=close.index)
    signal[z < -z_entry] =  1
    signal[z >  z_entry] = -1
    signal[(z.abs() < z_exit)] = 0
    import pandas as pd
    return signal.diff().fillna(0)

