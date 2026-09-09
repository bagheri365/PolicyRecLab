import numpy as np
import pytest

from policyreclab.estimators import estimate_dr
from policyreclab.logging import LoggedBanditData


def test_dr_matches_hand_calculation() -> None:
    logs = LoggedBanditData(
        context_indices=np.array([0, 0], dtype=np.int64),
        actions=np.array([0, 1], dtype=np.int64),
        rewards=np.array([1, 0], dtype=np.int64),
        behavior_propensities=np.array([0.5, 0.5]),
    )
    behavior = np.array([[0.5, 0.5]])
    target = np.array([[0.8, 0.2]])
    predictions = np.array([[0.6, 0.4]])

    estimate = estimate_dr(
        logs,
        behavior_probabilities=behavior,
        target_probabilities=target,
        predicted_reward_means=predictions,
    )

    # direct = 0.8*0.6 + 0.2*0.4 = 0.56
    # weights = [1.6, 0.4]
    # corrections = [1.6*(1-.6), .4*(0-.4)] = [.64, -.16]
    # average DR = mean([1.20, .40]) = .80
    np.testing.assert_allclose(estimate.direct_terms, [0.56, 0.56])
    np.testing.assert_allclose(estimate.correction_terms, [0.64, -0.16])
    assert estimate.value == pytest.approx(0.8)


def test_dr_refuses_zero_support() -> None:
    logs = LoggedBanditData(
        context_indices=np.array([0]),
        actions=np.array([0]),
        rewards=np.array([1]),
        behavior_propensities=np.array([1.0]),
    )
    behavior = np.array([[1.0, 0.0]])
    target = np.array([[0.5, 0.5]])
    predictions = np.array([[0.5, 0.5]])

    with pytest.raises(ValueError):
        estimate_dr(
            logs,
            behavior_probabilities=behavior,
            target_probabilities=target,
            predicted_reward_means=predictions,
        )


def test_dr_rejects_inconsistent_logged_propensity() -> None:
    logs = LoggedBanditData(
        context_indices=np.array([0]),
        actions=np.array([0]),
        rewards=np.array([1]),
        behavior_propensities=np.array([0.7]),
    )
    behavior = np.array([[0.6, 0.4]])
    target = np.array([[0.5, 0.5]])
    predictions = np.array([[0.5, 0.5]])

    with pytest.raises(ValueError):
        estimate_dr(
            logs,
            behavior_probabilities=behavior,
            target_probabilities=target,
            predicted_reward_means=predictions,
        )
