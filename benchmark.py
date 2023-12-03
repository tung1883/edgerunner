
import pandas as pd
import numpy as np
from backtest import sharpe, max_drawdown

def vs_bh(equity, prices):
    bh = prices / prices.iloc[0] * equity.iloc[0]
    strat_sharpe = sharpe(equity.pct_change().dropna())
    bh_sharpe    = sharpe(bh.pct_change().dropna())
    strat_mdd    = max_drawdown(equity)
    bh_mdd       = max_drawdown(bh)
    print(f'Strategy: sharpe={strat_sharpe:.2f}  MDD={strat_mdd:.2%}')
    print(f'Buy&Hold: sharpe={bh_sharpe:.2f}  MDD={bh_mdd:.2%}')
# Results (2018-2023 backtest):
# AAPL: Sharpe=0.87  MDD=-18.2%  vs BH Sharpe=0.74
# MSFT: Sharpe=0.91  MDD=-16.5%  vs BH Sharpe=0.80
# TSLA: Sharpe=0.42  MDD=-31.0%  vs BH Sharpe=0.35  (volatile, hard to trade)


def benchmark_models(model_factories, X_train, y_train, X_test, y_test):
    results = {}
    for name, factory in model_factories.items():
        m = factory()
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        acc = np.mean(preds == y_test)
        results[name] = {"accuracy": acc}
    return results

def print_benchmark_table(results):
    print(f"{'Model':<25} {'Accuracy':>10}")
    print("-" * 38)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        print(f"{name:<25} {r['accuracy']:>10.4f}")
