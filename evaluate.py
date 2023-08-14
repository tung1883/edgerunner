
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def report(y_true, y_pred):
    print(f'Accuracy : {accuracy_score(y_true, y_pred):.3f}')
    print(f'Precision: {precision_score(y_true, y_pred):.3f}')
    print(f'Recall   : {recall_score(y_true, y_pred):.3f}')
    print(f'F1       : {f1_score(y_true, y_pred):.3f}')

def in_vs_out(in_sample, out_sample):
    ratio = out_sample / (in_sample + 1e-9)
    print(f'In-sample  Sharpe: {in_sample:.3f}')
    print(f'Out-sample Sharpe: {out_sample:.3f}')
    print(f'Ratio: {ratio:.2f}  ({'good' if ratio > 0.6 else 'overfitting'})')


def confusion_matrix(y_true, y_pred, classes=(0, 1)):
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm

def classification_report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    prec = cm[1,1] / (cm[1,1] + cm[0,1] + 1e-8)
    rec  = cm[1,1] / (cm[1,1] + cm[1,0] + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return {"precision": prec, "recall": rec, "f1": f1, "cm": cm}

def portfolio_attribution(returns, factor_returns, weights):
    """Brinson-Hood-Beebower attribution."""
    selection = weights * (returns - factor_returns)
    allocation = (weights - 1/len(weights)) * factor_returns
    return {"selection": selection.sum(), "allocation": allocation.sum()}

def information_coefficient(predicted_returns, actual_returns, n_quantiles=5):
    """Rank IC — Spearman correlation between predicted and actual."""
    pred_rank   = predicted_returns.argsort().argsort()
    actual_rank = actual_returns.argsort().argsort()
    n = len(pred_rank)
    d2 = ((pred_rank - actual_rank)**2).sum()
    return 1 - 6 * d2 / (n * (n**2 - 1) + 1e-8)

def quantile_returns(predicted_returns, actual_returns, n_quantiles=5):
    """Actual returns per predicted-return quantile bucket."""
    edges = np.percentile(predicted_returns, np.linspace(0, 100, n_quantiles + 1))
    qret  = []
    for i in range(n_quantiles):
        mask = (predicted_returns >= edges[i]) & (predicted_returns < edges[i+1])
        qret.append(actual_returns[mask].mean() if mask.sum() else 0.0)
    return np.array(qret)

def sortino_ratio(returns, target_return=0.0, period=252):
    excess = returns - target_return / period
    downside = excess[excess < 0]
    downside_std = np.sqrt((downside**2).mean()) * np.sqrt(period)
    return excess.mean() * period / (downside_std + 1e-8)
