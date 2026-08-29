"""Weighted consensus and uncertainty aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bootstrap import LocalEstimate
from .graph import project_to_dag


@dataclass(frozen=True)
class AggregationResult:
    adjacency: np.ndarray
    probabilities: np.ndarray
    effective_evidence: np.ndarray
    entropy: np.ndarray


def aggregate_subgraphs(
    estimates: list[LocalEstimate],
    n_nodes: int,
    threshold: float = 0.5,
    weighted: bool = True,
    use_frequencies: bool = True,
) -> AggregationResult:
    """Combine local estimates and project their edge beliefs to a global DAG."""
    numerator = np.zeros((n_nodes, n_nodes), dtype=float)
    denominator = np.zeros((n_nodes, n_nodes), dtype=float)
    evidence_squared = np.zeros((n_nodes, n_nodes), dtype=float)
    for estimate in estimates:
        nodes = np.asarray(estimate.nodes)
        weight = estimate.weight if weighted else 1.0
        local_values = estimate.edge_frequencies if use_frequencies else estimate.adjacency
        for local_i, global_i in enumerate(nodes):
            for local_j, global_j in enumerate(nodes):
                if global_i == global_j:
                    continue
                numerator[global_i, global_j] += weight * local_values[local_i, local_j]
                denominator[global_i, global_j] += weight
                evidence_squared[global_i, global_j] += weight**2
    probabilities = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    effective = np.divide(
        denominator**2,
        evidence_squared,
        out=np.zeros_like(denominator),
        where=evidence_squared > 0,
    )
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    entropy = -(clipped * np.log2(clipped) + (1 - clipped) * np.log2(1 - clipped))
    entropy[denominator == 0] = 1.0
    adjacency = project_to_dag(probabilities, threshold)
    return AggregationResult(adjacency, probabilities, effective, entropy)

