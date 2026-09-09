import numpy as np
import pytest

from policyreclab.estimators import estimate_dm


def test_dm_matches_hand_calculation() -> None:
    target = np.array([[0.25, 0.75], [1.0, 0.0]])
    rewards = np.array([[0.2, 0.6], [0.8, 0.1]])

    estimate = estimate_dm(
        target_probabilities=target,
        predicted_reward_means=rewards,
    )

    # context values: 0.25*0.2 + 0.75*0.6 = 0.5; second = 0.8
    np.testing.assert_allclose(estimate.context_values, [0.5, 0.8])
    assert estimate.value == pytest.approx(0.65)


def test_dm_validates_shapes_and_probabilities() -> None:
    with pytest.raises(ValueError):
        estimate_dm(
            target_probabilities=np.array([[0.5, 0.5]]),
            predicted_reward_means=np.array([[0.2, 0.3, 0.4]]),
        )

    with pytest.raises(ValueError):
        estimate_dm(
            target_probabilities=np.array([[0.4, 0.4]]),
            predicted_reward_means=np.array([[0.2, 0.3]]),
        )
