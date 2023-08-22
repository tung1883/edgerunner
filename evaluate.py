
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

