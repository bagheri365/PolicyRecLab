import numpy as np

from policyreclab.policies import GreedyPolicy


def test_greedy_policy_is_one_hot_and_chooses_maximum() -> None:
    scores = np.array([[0.1, 0.9, 0.2], [0.8, 0.3, 0.4]], dtype=float)
    probabilities = GreedyPolicy().probabilities(scores)

    np.testing.assert_array_equal(
        probabilities,
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
