from causal_consensus.pipeline import run_pipeline
from causal_consensus.scm import SCMConfig, generate_scm


def test_end_to_end_pipeline():
    scm = generate_scm(SCMConfig(n_nodes=8, n_samples=180, seed=4))
    result = run_pipeline(scm, subset_size=4, n_subsets=20, n_bootstrap=6, seed=4)
    assert set(result.metrics) == {"weighted", "unweighted", "full"}
    assert result.metrics["weighted"]["pair_coverage"] == 1.0
    assert 0 <= result.metrics["weighted"]["f1"] <= 1

