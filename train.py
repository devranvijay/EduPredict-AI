"""Train and evaluate EduPredict ANN. Usage: python train.py

Trains three variants that differ in WHEN their inputs are known (early / during_term / late),
compares each to a predict-the-training-mean baseline, and writes models/ and results/.
Requires data/raw/edupredict_synthetic.csv (python -m src.synthetic_data).
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import preprocessing as pp
from src.ann import MLP, gradient_check
from src.evaluation import summarize

SEED = 42
HIDDEN = 16
FIT = dict(lr=0.01, epochs=300, batch_size=128, patience=25, seed=SEED)
VARIANTS = ["early", "during_term", "late"]
HIDDEN_SWEEP = [2, 4, 8, 16, 32]
CONTEXT_COLS = ["student_id", "subject", "prev_year_score", "attendance_pct", "study_hours_week", "midterm_score"]


def run_variant(name, df, idx):
    parts = {k: df.iloc[i] for k, i in zip(("train", "val", "test"), idx)}
    cfg = pp.fit_config(parts["train"], variant=name)
    X = {k: pp.transform(d, cfg) for k, d in parts.items()}
    y = {k: pp.scale_target(d[pp.TARGET], cfg) for k, d in parts.items()}

    sweep = {}
    for h in HIDDEN_SWEEP:  # validation-only comparison of hidden sizes
        m = MLP(X["train"].shape[1], h, seed=SEED)
        m.fit(X["train"], y["train"], X["val"], y["val"], **FIT)
        sweep[h] = summarize(parts["val"][pp.TARGET], pp.unscale_target(m.predict(X["val"]), cfg))["rmse"]

    net = MLP(X["train"].shape[1], HIDDEN, seed=SEED)
    grad_err = gradient_check(net, X["train"][:20], y["train"][:20])
    history = net.fit(X["train"], y["train"], X["val"], y["val"], **FIT)

    out = {"gradient_check_max_rel_error": grad_err, "epochs_run": len(history),
           "hidden_sweep_val_rmse": sweep, "n_inputs": int(X["train"].shape[1])}
    preds = {}
    for split, d in parts.items():
        pred = pp.unscale_target(net.predict(X[split]), cfg)
        base = np.full(len(d), parts["train"][pp.TARGET].mean())
        out[split] = {"ann": summarize(d[pp.TARGET], pred), "baseline_mean": summarize(d[pp.TARGET], base)}
        preds[split] = pred

    folder = Path("models") / name
    folder.mkdir(parents=True, exist_ok=True)
    net.save(folder / "weights.npz")
    pp.save_config(cfg, folder / "preprocessing.json")
    return out, pd.DataFrame(history).assign(variant=name), preds["test"]


def plot_results(hists, test_df):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in hists.items():
        ax[0].plot(h.epoch, h.train_loss, label=f"{name} train")
        ax[0].plot(h.epoch, h.val_loss, "--", label=f"{name} val")
    ax[0].set(xlabel="epoch", ylabel="MSE (standardised marks)", title="Training vs validation loss")
    ax[0].legend(fontsize=7)
    for name in hists:
        ax[1].scatter(test_df["actual"], test_df[f"pred_{name}"], s=5, alpha=.35, label=name)
    ax[1].plot([0, 100], [0, 100], "k:")
    ax[1].set(xlabel="actual marks", ylabel="predicted marks", title="Test set: actual vs predicted")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("results/training_overview.png", dpi=130)


def main():
    df = pp.load_raw()
    idx = pp.group_split(df, seed=SEED)
    Path("results").mkdir(exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    split = np.select([df.index.isin(idx[0]), df.index.isin(idx[1])], ["train", "val"], "test")
    df.assign(split=split).to_csv("data/processed/dataset_with_split.csv", index=False)

    metrics = {"data": "synthetic", "n_rows": len(df), "n_students": int(df.student_id.nunique()),
               "split_sizes": {k: len(i) for k, i in zip(("train", "val", "test"), idx)}}
    hists = {}
    test_df = df.iloc[idx[2]][CONTEXT_COLS].copy()
    test_df.insert(0, "actual", df.iloc[idx[2]][pp.TARGET])
    for name in VARIANTS:
        metrics[name], hists[name], test_df[f"pred_{name}"] = run_variant(name, df, idx)
        t = metrics[name]["test"]
        print(f"{name}: test MAE {t['ann']['mae']:.2f} RMSE {t['ann']['rmse']:.2f} R2 {t['ann']['r2']:.3f} "
              f"| baseline RMSE {t['baseline_mean']['rmse']:.2f} "
              f"| grad-check {metrics[name]['gradient_check_max_rel_error']:.1e} "
              f"| epochs {metrics[name]['epochs_run']}", flush=True)
    test_df.to_csv("results/predictions.csv", index=False)
    pd.concat(hists.values()).to_csv("results/training_history.csv", index=False)
    Path("results/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_results(hists, test_df)


if __name__ == "__main__":
    main()
