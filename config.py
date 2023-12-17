"""Central config — override any field for experiments."""
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Data
    tickers: List[str] = field(default_factory=lambda: ["AAPL","MSFT","GOOGL","JPM","SPY"])
    start_date: str = "2018-01-01"
    end_date:   str = "2022-12-31"

    # Features
    ma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    target_vol: float = 0.10

    # ML
    n_estimators: int = 200
    max_depth: int = 5
    lstm_hidden: int = 64
    lstm_layers: int = 2
    train_ratio: float = 0.60
    val_ratio: float = 0.20

    # Risk
    max_drawdown: float = 0.15
    kelly_fraction: float = 0.25
    slippage_bps: int = 5
    commission_bps: int = 10

    # Output
    output_dir: str = "outputs"
    save_plots: bool = True

@dataclass
class LiveConfig:
    broker: str = "paper"
    poll_interval_sec: int = 60
    max_position_usd: float = 10_000.0
    risk_per_trade: float = 0.01
    telegram_alerts: bool = False
