
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

