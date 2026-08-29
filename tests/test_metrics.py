import numpy as np

from causal_consensus.metrics import brier_score, directed_metrics, expected_calibration_error


def test_perfect_graph_metrics():
    truth = np.array([[0, 1], [0, 0]])
    values = directed_metrics(truth, truth)
    assert values["precision"] == 1
    assert values["recall"] == 1
    assert values["shd"] == 0
    assert brier_score(truth, truth.astype(float)) == 0
    assert expected_calibration_error(truth, truth.astype(float)) == 0

