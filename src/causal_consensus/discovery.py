"""Transparent ordered-regression causal discovery baseline."""

from __future__ import annotations

import numpy as np


def _fill_missing(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float).copy()
    means = np.nanmean(data, axis=0)
    if np.isnan(means).any():
        raise ValueError("a variable is entirely missing")
    rows, columns = np.where(np.isnan(data))
    data[rows, columns] = means[columns]
    return data


def ordered_regression_dag(
    data: np.ndarray,
    order: list[int] | tuple[int, ...],
    threshold: float = 0.12,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Estimate a DAG by standardized ridge regression under a supplied order.

    A known or hypothesized causal order is common in longitudinal and experimental
    settings. The explicit assumption isolates aggregation quality from orientation.
    """
    data = _fill_missing(data)
    if data.ndim != 2 or len(order) != data.shape[1] or set(order) != set(range(data.shape[1])):
        raise ValueError("order must contain every local variable exactly once")
    centered = data - data.mean(axis=0)
    scale = data.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = centered / scale
    adjacency = np.zeros((data.shape[1], data.shape[1]), dtype=np.int8)
    for position, target in enumerate(order):
        parents = list(order[:position])
        if not parents:
            continue
        design = standardized[:, parents]
        response = standardized[:, target]
        gram = design.T @ design + ridge * np.eye(len(parents))
        coefficients = np.linalg.solve(gram, design.T @ response)
        selected = np.abs(coefficients) >= threshold
        adjacency[np.asarray(parents)[selected], target] = 1
    return adjacency


def local_order(global_order: tuple[int, ...] | list[int], nodes: np.ndarray) -> list[int]:
    lookup = {global_node: local for local, global_node in enumerate(nodes.tolist())}
    return [lookup[node] for node in global_order if node in lookup]

