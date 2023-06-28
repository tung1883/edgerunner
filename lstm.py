
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

def dropout_mask(size, keep_prob):
    mask = (np.random.rand(*size) < keep_prob).astype(float)
    return mask / keep_prob

class LSTMWithDropout:
    def __init__(self, input_size, hidden_size, keep_prob=0.8):
        self.cell = LSTMCell(input_size, hidden_size)
        self.keep_prob = keep_prob
        self.training = True

    def forward(self, X):
        h = np.zeros(self.cell.hidden_size)
        c = np.zeros(self.cell.hidden_size)
        for t in range(X.shape[0]):
            x = X[t]
            if self.training:
                x = x * dropout_mask(x.shape, self.keep_prob)
            h, c = self.cell.forward(x, h, c)
        return h

class GRUCell:
    """Gated Recurrent Unit — fewer parameters than LSTM."""
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        s = np.sqrt(2 / (input_size + hidden_size))
        self.W_z = np.random.randn(input_size + hidden_size, hidden_size) * s
        self.W_r = np.random.randn(input_size + hidden_size, hidden_size) * s
        self.W_h = np.random.randn(input_size + hidden_size, hidden_size) * s
        self.b_z = np.zeros(hidden_size)
        self.b_r = np.zeros(hidden_size)
        self.b_h = np.zeros(hidden_size)

    def forward(self, x, h):
        xh = np.concatenate([x, h])
        z = 1 / (1 + np.exp(-np.clip(xh @ self.W_z + self.b_z, -30, 30)))
        r = 1 / (1 + np.exp(-np.clip(xh @ self.W_r + self.b_r, -30, 30)))
        xrh = np.concatenate([x, r * h])
        h_hat = np.tanh(xrh @ self.W_h + self.b_h)
        return (1 - z) * h + z * h_hat
