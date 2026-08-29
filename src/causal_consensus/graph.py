"""Directed-acyclic-graph utilities."""

from __future__ import annotations

import numpy as np


def is_dag(adjacency: np.ndarray) -> bool:
    """Return whether a square binary adjacency matrix is acyclic."""
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    indegree = adjacency.astype(bool).sum(axis=0).astype(int)
    queue = list(np.flatnonzero(indegree == 0))
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in np.flatnonzero(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))
    return visited == adjacency.shape[0]


def topological_order(adjacency: np.ndarray) -> list[int]:
    if not is_dag(adjacency):
        raise ValueError("graph contains a directed cycle")
    indegree = adjacency.astype(bool).sum(axis=0).astype(int)
    queue = sorted(np.flatnonzero(indegree == 0).tolist(), reverse=True)
    order: list[int] = []
    while queue:
        node = queue.pop()
        order.append(node)
        for child in np.flatnonzero(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))
                queue.sort(reverse=True)
    return order


def project_to_dag(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Greedily maximize edge log-odds subject to acyclicity.

    The exact maximum-weight acyclic subgraph problem is NP-hard. This deterministic
    approximation inserts candidate edges in descending confidence order and rejects
    any edge that would create a cycle.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != probabilities.shape[1]:
        raise ValueError("probabilities must be square")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    n_nodes = probabilities.shape[0]
    result = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    candidates = [
        (float(probabilities[i, j]), i, j)
        for i in range(n_nodes)
        for j in range(n_nodes)
        if i != j and probabilities[i, j] >= threshold
    ]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, source, target in candidates:
        result[source, target] = 1
        if not is_dag(result):
            result[source, target] = 0
    return result


def random_dag(
    n_nodes: int,
    expected_degree: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[int]]:
    """Sample an Erdos-Renyi DAG and return its hidden topological order."""
    if n_nodes < 2 or expected_degree <= 0:
        raise ValueError("n_nodes must be >= 2 and expected_degree must be positive")
    order = rng.permutation(n_nodes).tolist()
    probability = min(1.0, expected_degree / max(1, n_nodes - 1))
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for earlier in range(n_nodes):
        for later in range(earlier + 1, n_nodes):
            if rng.random() < probability:
                adjacency[order[earlier], order[later]] = 1
    return adjacency, order

