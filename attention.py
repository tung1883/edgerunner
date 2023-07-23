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

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        scale = 0.01
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def forward(self, X):
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        scores = Q @ K.T / np.sqrt(self.d_k)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True) + 1e-8
        out = weights @ V @ self.W_o
        return out
