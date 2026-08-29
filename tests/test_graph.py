import numpy as np

from causal_consensus.graph import is_dag, project_to_dag, random_dag


def test_random_graph_is_acyclic():
    graph, order = random_dag(20, 2.5, np.random.default_rng(2))
    assert is_dag(graph)
    positions = {node: index for index, node in enumerate(order)}
    assert all(positions[i] < positions[j] for i, j in zip(*np.where(graph)))


def test_projection_rejects_cycle():
    probabilities = np.zeros((3, 3))
    probabilities[0, 1] = 0.9
    probabilities[1, 2] = 0.8
    probabilities[2, 0] = 0.7
    graph = project_to_dag(probabilities)
    assert graph.sum() == 2
    assert is_dag(graph)

