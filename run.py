
from data_loader import fetch_all
from features import build, normalise, split
from models import train_rf_regularised, labels
from signals import from_model
from backtest import run, sharpe, max_drawdown
from config import TICKERS

def main():
    print('Fetching data...')
    data = fetch_all()
    results = {}
    for ticker, df in data.items():
        print(f'  {ticker}...')
        feats = build(df)
        y     = labels(feats, horizon=5)
        feats, _  = normalise(feats)
        X_train, X_test = split(feats)
        y_train, y_test = split(y)
        model = train_rf_regularised(X_train, y_train)
        signals_arr = from_model(model, X_test)
        import pandas as pd
        sig_series = pd.Series(signals_arr, index=X_test.index)
        equity = run(df.loc[X_test.index], sig_series)
        results[ticker] = {
            'sharpe': sharpe(equity.pct_change().dropna()),
            'mdd':    max_drawdown(equity),
            'final':  equity.iloc[-1],
        }
    print()
    for t, r in results.items():
        print(f'{t:<6}  Sharpe={r["sharpe"]:+.2f}  MDD={r["mdd"]:.1%}  Final=${r["final"]:,.0f}')

if __name__ == '__main__':
    main()

