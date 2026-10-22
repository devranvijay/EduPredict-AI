"""Regression metrics (all on the 0-100 marks scale)."""
import numpy as np


def mae(y, y_hat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(y_hat))))


def rmse(y, y_hat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(y_hat)) ** 2)))


def r2(y, y_hat):
    y, y_hat = np.asarray(y), np.asarray(y_hat)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - y_hat) ** 2) / ss_tot)


def summarize(y, y_hat):
    return {"mae": mae(y, y_hat), "rmse": rmse(y, y_hat), "r2": r2(y, y_hat)}
