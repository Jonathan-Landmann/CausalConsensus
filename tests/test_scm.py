import numpy as np

from causal_consensus.graph import is_dag
from causal_consensus.scm import SCMConfig, generate_scm


def test_scm_is_reproducible():
    config = SCMConfig(n_nodes=8, n_samples=100, seed=11)
    first = generate_scm(config)
    second = generate_scm(config)
    np.testing.assert_allclose(first.samples, second.samples)
    assert is_dag(first.adjacency)


def test_missing_data_is_generated():
    data = generate_scm(SCMConfig(n_nodes=6, n_samples=100, missing_rate=0.2))
    assert np.isnan(data.samples).any()

