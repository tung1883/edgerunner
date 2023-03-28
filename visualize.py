
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

