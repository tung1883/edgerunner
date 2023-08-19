
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(model, feature_names, out='feature_importance.png'):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(imp)), imp[idx])
    plt.xticks(range(len(imp)), [feature_names[i] for i in idx], rotation=45)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_predictions(dates, actual, predicted, out='predictions.png'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.plot(dates, actual,    label='Actual',    linewidth=1.5)
    plt.plot(dates, predicted, label='Predicted', linestyle='--', linewidth=1.5)
    plt.title('LSTM Price Prediction')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_correlation(df, out='correlation.png'):
    import matplotlib.pyplot as plt
    corr = df.corr()
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=45)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title('Asset Correlation Matrix')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_equity_curve(dates, equity, benchmark=None, title="Equity Curve"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, equity, label="Strategy", linewidth=1.5)
    if benchmark is not None:
        ax.plot(dates, benchmark, label="Buy & Hold", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_title(title)
    ax.legend()
    ax.set_ylabel("Portfolio value")
    fig.tight_layout()
    return fig

def plot_drawdown(dates, equity):
    import matplotlib.pyplot as plt
    roll_max = np.maximum.accumulate(equity)
    dd = (equity - roll_max) / roll_max
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(dates, dd, 0, color="red", alpha=0.4, label="Drawdown")
    ax.set_title("Drawdown")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_rolling_sharpe(dates, returns, window=252):
    import matplotlib.pyplot as plt
    sharpe = []
    for i in range(len(returns)):
        r = returns[max(0,i-window):i+1]
        s = r.mean() / (r.std() + 1e-8) * np.sqrt(252)
        sharpe.append(s)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(dates, sharpe, label=f"Rolling {window}d Sharpe")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_monthly_returns_heatmap(returns, dates):
    import matplotlib.pyplot as plt
    from collections import defaultdict
    monthly = defaultdict(dict)
    for r, d in zip(returns, dates):
        monthly[d.year][d.month] = monthly[d.year].get(d.month, 1) * (1 + r)
    years = sorted(monthly.keys())
    data = [[monthly[y].get(m, 0) - 1 for m in range(1,13)] for y in years]
    fig, ax = plt.subplots(figsize=(12, len(years)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
    ax.set_yticks(range(len(years))); ax.set_yticklabels(years)
    ax.set_xticks(range(12)); ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig
