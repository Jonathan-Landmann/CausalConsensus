"""Graph recovery and probabilistic calibration metrics."""

from __future__ import annotations

import numpy as np


def directed_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction must have the same shape")
    mask = ~np.eye(truth.shape[0], dtype=bool)
    true_edges = truth[mask]
    predicted_edges = prediction[mask]
    tp = int(np.logical_and(true_edges, predicted_edges).sum())
    fp = int(np.logical_and(~true_edges, predicted_edges).sum())
    fn = int(np.logical_and(true_edges, ~predicted_edges).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    reversals = sum(
        truth[i, j] and prediction[j, i]
        for i in range(truth.shape[0])
        for j in range(i + 1, truth.shape[0])
    )
    shd = fp + fn - reversals
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": float(shd),
        "reversals": float(reversals),
    }


def brier_score(truth: np.ndarray, probabilities: np.ndarray) -> float:
    mask = ~np.eye(truth.shape[0], dtype=bool)
    return float(np.mean((probabilities[mask] - truth[mask]) ** 2))


def expected_calibration_error(
    truth: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    mask = ~np.eye(truth.shape[0], dtype=bool)
    labels = truth[mask].astype(float)
    scores = probabilities[mask]
    bins = np.linspace(0, 1, n_bins + 1)
    error = 0.0
    for index in range(n_bins):
        in_bin = (scores >= bins[index]) & (
            scores <= bins[index + 1] if index == n_bins - 1 else scores < bins[index + 1]
        )
        if in_bin.any():
            error += in_bin.mean() * abs(scores[in_bin].mean() - labels[in_bin].mean())
    return float(error)


def pair_coverage(effective_evidence: np.ndarray) -> float:
    mask = ~np.eye(effective_evidence.shape[0], dtype=bool)
    return float((effective_evidence[mask] > 0).mean())

