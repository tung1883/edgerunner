
import numpy as np
from lstm import LSTMCell

def mse(y, yhat): return ((y - yhat) ** 2).mean()

def train(X, y, hidden=64, epochs=50, lr=0.001):
    T, input_dim = X.shape[1], X.shape[2]
    cell = LSTMCell(input_dim, hidden)
    losses = []
    for ep in range(epochs):
        ep_loss = 0.0
        for seq, target in zip(X, y):
            h = np.zeros(hidden)
            c = np.zeros(hidden)
            for t in range(T):
                h, c = cell.forward(seq[t], h, c)
            pred    = h.mean()
            ep_loss += (pred - target) ** 2
        losses.append(ep_loss / len(X))
        if ep % 10 == 0:
            print(f'epoch {ep:3d}  loss={losses[-1]:.5f}')
    return cell, losses


def lr_schedule(epoch, warmup=10, decay=0.98):
    if epoch < warmup:
        return epoch / warmup
    return decay ** (epoch - warmup)

def train_with_early_stopping(model, X_tr, y_tr, X_val, y_val,
                               max_epochs=200, patience=15, base_lr=0.001):
    best_val, wait, best_w = np.inf, 0, None
    for epoch in range(max_epochs):
        lr = base_lr * lr_schedule(epoch)
        loss = _train_epoch(model, X_tr, y_tr, lr)
        val_loss = _eval(model, X_val, y_val)
        if val_loss < best_val:
            best_val, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best_val
