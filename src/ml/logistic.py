"""
logistic.py
===========
A tiny NumPy logistic-regression implementation used for the first ML
benchmark. Keeping this in-repo makes the baseline easy to audit and
avoids adding a heavy dependency before the modeling layer needs one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogisticModel:
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    prior: float | None = None


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1 / (1 + np.exp(-z))


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.08,
    iterations: int = 500,
    l2: float = 0.001,
) -> LogisticModel:
    """
    Fit a binary logistic model with standardized features.

    If the training window contains only one class, return a constant
    prior model instead of failing. That can happen in short crypto
    windows and is better than silently leaking future data.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mean = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0)
    scale = np.where(scale == 0, 1, scale)
    x_std = np.nan_to_num((x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    x_design = np.column_stack([np.ones(len(x_std)), x_std])

    class_count = len(np.unique(y))
    if class_count < 2:
        prior = float(y.mean()) if len(y) else 0.5
        return LogisticModel(weights=np.zeros(x_design.shape[1]), mean=mean, scale=scale, prior=prior)

    weights = np.zeros(x_design.shape[1])
    for _ in range(iterations):
        pred = _sigmoid(x_design @ weights)
        gradient = (x_design.T @ (pred - y)) / len(y)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient

    return LogisticModel(weights=weights, mean=mean, scale=scale)


def predict_proba(model: LogisticModel, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if model.prior is not None:
        return np.full(len(x), model.prior)

    x_std = np.nan_to_num((x - model.mean) / model.scale, nan=0.0, posinf=0.0, neginf=0.0)
    x_design = np.column_stack([np.ones(len(x_std)), x_std])
    return _sigmoid(x_design @ model.weights)

