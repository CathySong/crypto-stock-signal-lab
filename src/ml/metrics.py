"""
metrics.py
==========
Classification metrics for direction-probability benchmarks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def roc_auc_score(y_true: pd.Series, y_score: pd.Series) -> float:
    """
    Compute ROC AUC from ranks without external dependencies.

    Returns NaN if the sample contains only one class.
    """
    y = pd.Series(y_true).astype(int)
    score = pd.Series(y_score).astype(float)
    valid = y.notna() & score.notna()
    y = y[valid]
    score = score[valid]

    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    ranks = score.rank(method="average")
    positive_rank_sum = float(ranks[y == 1].sum())
    auc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc)


def summarize_classifier(y_true: pd.Series, y_prob: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    y = pd.Series(y_true).astype(int)
    prob = pd.Series(y_prob).astype(float)
    valid = y.notna() & prob.notna()
    y = y[valid]
    prob = prob[valid]
    pred = (prob >= threshold).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    accuracy = float((pred == y).mean()) if len(y) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    brier = float(np.mean((prob - y) ** 2)) if len(y) else float("nan")

    return {
        "samples": float(len(y)),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "brier_score": brier,
        "roc_auc": roc_auc_score(y, prob),
    }

