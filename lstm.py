
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

