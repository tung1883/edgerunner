
import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))
def tanh(x):    return np.tanh(np.clip(x, -30, 30))

class LSTMCell:
    def __init__(self, input_dim, hidden_dim, rng=None):
        rng = rng or np.random.default_rng(42)
        s   = 0.1
        d   = input_dim + hidden_dim
        self.Wf = rng.normal(0, s, (d, hidden_dim))
        self.Wi = rng.normal(0, s, (d, hidden_dim))
        self.Wc = rng.normal(0, s, (d, hidden_dim))
        self.Wo = rng.normal(0, s, (d, hidden_dim))
        self.bf = np.zeros(hidden_dim)
        self.bi = np.zeros(hidden_dim)
        self.bc = np.zeros(hidden_dim)
        self.bo = np.zeros(hidden_dim)

    def forward(self, x, h_prev, c_prev):
        xh  = np.concatenate([x, h_prev])
        f   = sigmoid(xh @ self.Wf + self.bf)
        i   = sigmoid(xh @ self.Wi + self.bi)
        c_  = tanh(xh @ self.Wc + self.bc)
        o   = sigmoid(xh @ self.Wo + self.bo)
        c   = f * c_prev + i * c_
        h   = o * tanh(c)
        return h, c


class StackedLSTM:
    """Two-layer stacked LSTM."""
    def __init__(self, input_size, hidden_sizes=(64, 32), output_size=1):
        self.layer1 = LSTMCell(input_size, hidden_sizes[0])
        self.layer2 = LSTMCell(hidden_sizes[0], hidden_sizes[1])
        self.W_out = np.random.randn(hidden_sizes[1], output_size) * 0.01
        self.b_out = np.zeros(output_size)

    def forward(self, X):
        h1, c1 = np.zeros(self.layer1.hidden_size), np.zeros(self.layer1.hidden_size)
        h2, c2 = np.zeros(self.layer2.hidden_size), np.zeros(self.layer2.hidden_size)
        for t in range(X.shape[0]):
            h1, c1 = self.layer1.forward(X[t], h1, c1)
            h2, c2 = self.layer2.forward(h1, h2, c2)
        return h2 @ self.W_out + self.b_out
