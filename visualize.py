
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
