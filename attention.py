"""Scaled dot-product attention over LSTM hidden states."""
import numpy as np

def attention(queries, keys, values, mask=None):
    d_k = keys.shape[-1]
    scores = queries @ keys.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = np.exp(scores - scores.max())
    weights /= weights.sum() + 1e-8
    return weights @ values, weights

class AttentionLSTM:
    def __init__(self, input_size, hidden_size):
        from lstm import LSTMCell
        self.cell = LSTMCell(input_size, hidden_size)
        self.W_q = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_k = np.random.randn(hidden_size, hidden_size) * 0.01

    def forward(self, X):
        hidden_states = []
        h, c = np.zeros(self.cell.hidden_size), np.zeros(self.cell.hidden_size)
        for t in range(X.shape[0]):
            h, c = self.cell.forward(X[t], h, c)
            hidden_states.append(h.copy())
        H = np.stack(hidden_states)
        q = h @ self.W_q
        k = H @ self.W_k
        ctx, _ = attention(q[None], k, H)
        return ctx[0]
