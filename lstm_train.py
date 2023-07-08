
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

def batch_generator(X, y, batch_size=32, shuffle=True):
    n = len(X)
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        b = idx[start:start+batch_size]
        yield X[b], y[b]

def cosine_annealing_lr(epoch, T_max, eta_min=1e-5, eta_max=1e-3):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * epoch / T_max))
