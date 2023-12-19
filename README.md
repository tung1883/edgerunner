# quant_trader

ML-driven quantitative trading bot — technical indicators, regime detection, LSTM price prediction, and ensemble signal generation. Backtested on 5 years of US equity data.

## Results
RF + ensemble beats buy-and-hold on 3/5 tickers (2018-2023 backtest). Best: MSFT Sharpe 0.91 vs 0.80 buy-and-hold.

## Quickstart
```bash
pip install -r requirements.txt
python run.py
```

## Strategies
- SMA crossover
- RSI overbought/oversold
- MACD + RSI combined
- Mean reversion (z-score)
- Momentum (12-1 month)
- Sector rotation
- Regime-adaptive switching

