
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

