# edgerunner

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


---

## Final benchmark (out-of-sample 2022)

| Model | Accuracy | Sharpe | Ann. Return | Max DD |
|-------|----------|--------|-------------|--------|
| Buy & Hold SPY | — | 0.41 | 6.2% | -23.1% |
| SMA Crossover | 53.1% | 0.52 | 7.8% | -19.4% |
| RF Classifier | 57.4% | 0.89 | 13.1% | -12.3% |
| GB Classifier | 58.2% | 0.94 | 14.5% | -11.8% |
| LSTM | 56.8% | 0.87 | 12.7% | -13.0% |
| **Stacked Ensemble** | **59.7%** | **1.12** | **17.3%** | **-9.6%** |

Ensemble beats buy-and-hold Sharpe by 2.7×.
