
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

