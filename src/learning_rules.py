"""Classic neuron models with every calculation exposed as a table row.

Run `python -m src.learning_rules` to print all three demonstrations.
"""
import numpy as np


# ---- McCulloch-Pitts: binary inputs, fixed weights, hard threshold -------------
def mp_neuron(x, w, threshold):
    net = int(np.dot(x, w))
    return net, int(net >= threshold)


def demo_mp(gate="AND"):
    """AND needs threshold 2, OR needs threshold 1 (weights are both +1)."""
    threshold = {"AND": 2, "OR": 1}[gate]
    w = [1, 1]
    rows = []
    for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        net, out = mp_neuron(x, w, threshold)
        rows.append({"x1": x[0], "x2": x[1], "w1": w[0], "w2": w[1],
                     "weighted_sum": net, "threshold": threshold, "output": out})
    return rows


# ---- Hebb: delta_w = x * y, bias += y (bipolar data) ---------------------------
def demo_hebb():
    """Learn bipolar AND with one pass of Hebb's rule. Returns (rows, weights, bias)."""
    X = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    T = [1, -1, -1, -1]
    w, b, rows = np.zeros(2), 0, []
    for x, t in zip(X, T):
        dw = np.array(x) * t
        w, b = w + dw, b + t
        rows.append({"x1": x[0], "x2": x[1], "target": t, "dw1": int(dw[0]),
                     "dw2": int(dw[1]), "db": t, "w1": int(w[0]), "w2": int(w[1]), "b": int(b)})
    return rows, w, b


# ---- Delta rule: w <- w + lr * (t - y) * x on a linear unit --------------------
def demo_delta(lr=0.1, epochs=30):
    """Train a single linear neuron on AND (0/1). Returns (step rows, epoch MSEs, w, b)."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    T = np.array([0, 0, 0, 1], float)
    w, b, rows, losses = np.zeros(2), 0.0, [], []
    for epoch in range(1, epochs + 1):
        sq = 0.0
        for x, t in zip(X, T):
            y = float(w @ x + b)
            err = t - y
            w, b = w + lr * err * x, b + lr * err
            sq += err ** 2
            rows.append({"epoch": epoch, "x1": x[0], "x2": x[1], "target": t,
                         "output": round(y, 4), "error": round(err, 4),
                         "w1": round(w[0], 4), "w2": round(w[1], 4), "b": round(b, 4)})
        losses.append(sq / len(X))
    return rows, losses, w, b


# ---- Delta rule on real data: one linear neuron, online updates ---------------------
def delta_rule_fit(X, y, X_val, y_val, lr=0.01, epochs=10, seed=0, n_steps=5):
    """Train y_hat = w.x + b with the Delta rule, one sample at a time (shuffled each epoch).

    Returns a dict with: `checkpoints` (validation MSE and weights after 0, 1, 2, 5, 10 ... updates and after every
    epoch - the neuron learns most within the first epoch, so a per-epoch view alone looks flat), the per-epoch
    `weights`/`biases`/`val_mse`, the first `n_steps` updates (x, output, error, weights after) and a `diverged` flag.
    """
    rng = np.random.default_rng(seed)
    w, b = np.zeros(X.shape[1]), 0.0
    marks = {0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000}
    weights, biases, val_mse, steps, checkpoints = [], [], [], [], []

    def checkpoint(n):
        checkpoints.append({"updates": n, "val_mse": float(np.mean((X_val @ w + b - y_val) ** 2)), "w": w.copy()})

    def result(diverged):
        return {"weights": weights, "biases": biases, "val_mse": val_mse, "steps": steps,
                "checkpoints": checkpoints, "diverged": diverged}

    n = 0
    with np.errstate(over="ignore", invalid="ignore"):  # diverging runs are reported via `diverged`
        checkpoint(0)
        for epoch in range(epochs):
            for i in rng.permutation(len(X)):
                out = float(w @ X[i] + b)
                err = float(y[i] - out)
                w, b = w + lr * err * X[i], b + lr * err
                n += 1
                if epoch == 0 and len(steps) < n_steps:
                    steps.append({"x": X[i].copy(), "output": out, "target": float(y[i]), "error": err,
                                  "w": w.copy(), "b": b})
                if n in marks and n < epochs * len(X):
                    checkpoint(n)
            weights.append(w.copy())
            biases.append(b)
            checkpoint(n)
            val_mse.append(checkpoints[-1]["val_mse"])
            if not np.isfinite(w).all():
                return result(True)
    return result(False)


def least_squares(X, y):
    """Closed-form best linear neuron (weights, bias) - the target the Delta rule converges toward."""
    coef = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), y, rcond=None)[0]
    return coef[:-1], float(coef[-1])


if __name__ == "__main__":
    for gate in ("AND", "OR"):
        print(f"\nMcCulloch-Pitts {gate}")
        for r in demo_mp(gate):
            print(r)
    rows, w, b = demo_hebb()
    print("\nHebb (bipolar AND)")
    for r in rows:
        print(r)
    rows, losses, w, b = demo_delta()
    print(f"\nDelta rule: final w={w.round(3)}, b={b:.3f}, MSE {losses[0]:.3f} -> {losses[-1]:.3f}")
