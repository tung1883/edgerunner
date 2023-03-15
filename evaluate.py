
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def report(y_true, y_pred):
    print(f'Accuracy : {accuracy_score(y_true, y_pred):.3f}')
    print(f'Precision: {precision_score(y_true, y_pred):.3f}')
    print(f'Recall   : {recall_score(y_true, y_pred):.3f}')
    print(f'F1       : {f1_score(y_true, y_pred):.3f}')

