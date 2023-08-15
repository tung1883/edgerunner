
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

