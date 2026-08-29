import numpy as np

from causal_consensus.aggregation import aggregate_subgraphs
from causal_consensus.bootstrap import LocalEstimate, sample_node_subsets
from causal_consensus.graph import is_dag


def estimate(nodes, edge, weight):
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.int8)
    adjacency[edge] = 1
    return LocalEstimate(tuple(nodes), adjacency, adjacency.astype(float), 0.0, weight)


def test_reliable_estimate_dominates_unstable_disagreement():
    reliable = estimate([0, 1], (0, 1), 0.9)
    unstable = estimate([0, 1], (1, 0), 0.1)
    result = aggregate_subgraphs([reliable, unstable], 2)
    assert result.probabilities[0, 1] == 0.9
    assert result.adjacency[0, 1] == 1
    assert is_dag(result.adjacency)


def test_subset_sampler_covers_pairs():
    subsets = sample_node_subsets(8, 4, 25, np.random.default_rng(1))
    coverage = np.zeros((8, 8), dtype=int)
    for nodes in subsets:
        coverage[np.ix_(nodes, nodes)] += 1
    assert coverage[~np.eye(8, dtype=bool)].min() > 0

