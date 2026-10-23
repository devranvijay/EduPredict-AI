"""A one-hidden-layer neural network (ReLU hidden, linear output) in pure NumPy.

Shapes: X (n, d) -> z1 = X W1 + b1 -> a1 = ReLU(z1) -> y_hat = a1 W2 + b2 (n, 1)
Loss:   MSE = mean((y_hat - y)^2)
"""
import copy

import numpy as np


class MLP:
    def __init__(self, n_inputs, n_hidden=16, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {  # He initialisation suits ReLU layers
            "W1": rng.normal(0, np.sqrt(2 / n_inputs), (n_inputs, n_hidden)),
            "b1": np.zeros((1, n_hidden)),
            "W2": rng.normal(0, np.sqrt(1 / n_hidden), (n_hidden, 1)),
            "b2": np.zeros((1, 1)),
        }

    # ---- forward propagation ------------------------------------------------
    def forward(self, X, params=None):
        p = params or self.params
        z1 = X @ p["W1"] + p["b1"]          # weighted sum of hidden neurons
        a1 = np.maximum(0.0, z1)            # ReLU activation
        y_hat = a1 @ p["W2"] + p["b2"]      # linear output neuron
        return y_hat, (X, z1, a1)

    def predict(self, X):
        return self.forward(X)[0].ravel()

    @staticmethod
    def mse(y_hat, y):
        return float(np.mean((y_hat - y) ** 2))

    # ---- backpropagation (chain rule, layer by layer) ------------------------
    def backward(self, cache, y_hat, y):
        X, z1, a1 = cache
        n = len(y)
        dz2 = 2.0 * (y_hat - y) / n                 # dL/dz2
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        dz1 = (dz2 @ self.params["W2"].T) * (z1 > 0)  # ReLU derivative
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # ---- gradient descent ---------------------------------------------------
    def fit(self, X, y, X_val, y_val, lr=0.01, epochs=800, batch_size=32,
            patience=80, seed=0):
        """Mini-batch gradient descent with early stopping on validation MSE.

        Restores the weights from the best validation epoch. Returns the history.
        """
        y, y_val = y.reshape(-1, 1), y_val.reshape(-1, 1)
        rng = np.random.default_rng(seed)
        history, best, best_params, since_best = [], np.inf, None, 0
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                idx = order[start:start + batch_size]
                y_hat, cache = self.forward(X[idx])
                grads = self.backward(cache, y_hat, y[idx])
                for k in self.params:                      # weight update
                    self.params[k] -= lr * grads[k]
            train_loss = self.mse(self.forward(X)[0], y)
            val_loss = self.mse(self.forward(X_val)[0], y_val)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best - 1e-6:
                best, best_params, since_best = val_loss, copy.deepcopy(self.params), 0
            else:
                since_best += 1
                if since_best >= patience:
                    break
        self.params = best_params
        return history

    # ---- persistence --------------------------------------------------------
    def save(self, path):
        np.savez(path, **self.params)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        net = cls(data["W1"].shape[0], data["W1"].shape[1])
        net.params = {k: data[k] for k in ("W1", "b1", "W2", "b2")}
        return net


def gradient_check(net, X, y, eps=1e-5):
    """Max relative error between backprop gradients and central differences."""
    y = y.reshape(-1, 1)
    # ReLU is not differentiable at 0: drop samples whose pre-activations sit
    # within the finite-difference step of the kink.
    keep = np.abs(net.forward(X)[1][1]).min(axis=1) > 1e-3
    X, y = X[keep], y[keep]
    y_hat, cache = net.forward(X)
    analytic = net.backward(cache, y_hat, y)
    worst = 0.0
    for name, w in net.params.items():
        for i in np.ndindex(w.shape):
            orig = w[i]
            w[i] = orig + eps
            plus = net.mse(net.forward(X)[0], y)
            w[i] = orig - eps
            minus = net.mse(net.forward(X)[0], y)
            w[i] = orig
            numeric = (plus - minus) / (2 * eps)
            denom = max(abs(numeric) + abs(analytic[name][i]), 1e-12)
            worst = max(worst, abs(numeric - analytic[name][i]) / denom)
    return worst
