"""End-to-end CausalConsensus experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aggregation import AggregationResult, aggregate_subgraphs
from .bootstrap import LocalEstimate, estimate_stable_subgraph, sample_node_subsets
from .discovery import ordered_regression_dag
from .metrics import brier_score, directed_metrics, expected_calibration_error, pair_coverage
from .scm import SCMData


@dataclass(frozen=True)
class PipelineResult:
    weighted: AggregationResult
    unweighted: AggregationResult
    full_graph: np.ndarray
    estimates: tuple[LocalEstimate, ...]
    metrics: dict[str, dict[str, float]]


def run_pipeline(
    scm: SCMData,
    subset_size: int = 6,
    n_subsets: int = 80,
    n_bootstrap: int = 25,
    edge_threshold: float = 0.12,
    aggregation_threshold: float = 0.5,
    stability_lambda: float = 4.0,
    seed: int = 7,
) -> PipelineResult:
    rng = np.random.default_rng(seed)
    subsets = sample_node_subsets(scm.samples.shape[1], subset_size, n_subsets, rng)
    estimates = tuple(
        estimate_stable_subgraph(
            scm.samples,
            nodes,
            scm.order,
            n_bootstrap=n_bootstrap,
            edge_threshold=edge_threshold,
            stability_lambda=stability_lambda,
            seed=seed + index + 1,
        )
        for index, nodes in enumerate(subsets)
    )
    weighted = aggregate_subgraphs(
        list(estimates), scm.samples.shape[1], aggregation_threshold, weighted=True
    )
    unweighted = aggregate_subgraphs(
        list(estimates), scm.samples.shape[1], aggregation_threshold, weighted=False
    )
    full_graph = ordered_regression_dag(scm.samples, scm.order, edge_threshold)

    metrics: dict[str, dict[str, float]] = {}
    for name, adjacency, probabilities in (
        ("weighted", weighted.adjacency, weighted.probabilities),
        ("unweighted", unweighted.adjacency, unweighted.probabilities),
        ("full", full_graph, full_graph.astype(float)),
    ):
        values = directed_metrics(scm.adjacency, adjacency)
        values["brier"] = brier_score(scm.adjacency, probabilities)
        values["ece"] = expected_calibration_error(scm.adjacency, probabilities)
        metrics[name] = values
    metrics["weighted"]["pair_coverage"] = pair_coverage(weighted.effective_evidence)
    return PipelineResult(weighted, unweighted, full_graph, estimates, metrics)

