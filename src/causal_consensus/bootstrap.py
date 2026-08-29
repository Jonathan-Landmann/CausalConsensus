"""Bootstrap stability estimation for local causal graphs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .discovery import local_order, ordered_regression_dag


@dataclass(frozen=True)
class LocalEstimate:
    nodes: tuple[int, ...]
    adjacency: np.ndarray
    edge_frequencies: np.ndarray
    instability: float
    weight: float


def sample_node_subsets(
    n_nodes: int,
    subset_size: int,
    n_subsets: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Sample subsets while greedily favoring under-covered node pairs."""
    if not 2 <= subset_size <= n_nodes:
        raise ValueError("subset_size must be between 2 and n_nodes")
    coverage = np.zeros((n_nodes, n_nodes), dtype=int)
    subsets: list[np.ndarray] = []
    candidates = max(20, n_nodes)
    for _ in range(n_subsets):
        proposals = [np.sort(rng.choice(n_nodes, subset_size, replace=False)) for _ in range(candidates)]
        scores = [coverage[np.ix_(nodes, nodes)].sum() for nodes in proposals]
        nodes = proposals[int(np.argmin(scores))]
        coverage[np.ix_(nodes, nodes)] += 1
        subsets.append(nodes)
    return subsets


def estimate_stable_subgraph(
    data: np.ndarray,
    nodes: np.ndarray,
    global_order: tuple[int, ...] | list[int],
    n_bootstrap: int = 30,
    edge_threshold: float = 0.12,
    stability_lambda: float = 4.0,
    seed: int = 7,
) -> LocalEstimate:
    """Estimate edge frequencies and a graph-level exponential stability weight."""
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    rng = np.random.default_rng(seed)
    local_data = np.asarray(data)[:, nodes]
    order = local_order(global_order, nodes)
    estimates = []
    for _ in range(n_bootstrap):
        rows = rng.integers(0, local_data.shape[0], size=local_data.shape[0])
        estimates.append(ordered_regression_dag(local_data[rows], order, edge_threshold))
    stack = np.stack(estimates)
    frequencies = stack.mean(axis=0)
    consensus = (frequencies >= 0.5).astype(np.int8)
    possible_edges = len(nodes) * (len(nodes) - 1) / 2
    disagreements = np.abs(stack - consensus).sum(axis=(1, 2)) / max(1.0, possible_edges)
    instability = float(disagreements.mean())
    weight = float(np.exp(-stability_lambda * instability))
    return LocalEstimate(tuple(int(node) for node in nodes), consensus, frequencies, instability, weight)

