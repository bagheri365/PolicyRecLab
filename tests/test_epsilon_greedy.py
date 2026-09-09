import numpy as np
import pytest

from policyreclab.policies import EpsilonGreedyPolicy


def test_epsilon_greedy_probabilities_match_definition() -> None:
    scores = np.array([[0.1, 0.8, 0.2, 0.3]])
    probabilities = EpsilonGreedyPolicy(0.2).probabilities(scores)

    expected = np.array([[0.05, 0.85, 0.05, 0.05]])
    np.testing.assert_allclose(probabilities, expected)


@pytest.mark.parametrize("epsilon", [-0.1, 1.1])
def test_invalid_epsilon_fails(epsilon: float) -> None:
    with pytest.raises(ValueError):
        EpsilonGreedyPolicy(epsilon)
