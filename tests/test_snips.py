import numpy as np
import pytest

from policyreclab.estimators import estimate_snips
from policyreclab.logging import LoggedBanditData


def test_snips_matches_hand_calculation() -> None:
    logs = LoggedBanditData(
        context_indices=np.array([0, 0], dtype=np.int64),
        actions=np.array([0, 1], dtype=np.int64),
        rewards=np.array([1, 0], dtype=np.int64),
        behavior_propensities=np.array([0.5, 0.5], dtype=np.float64),
    )
    behavior = np.array([[0.5, 0.5]])
    target = np.array([[0.8, 0.2]])

    estimate = estimate_snips(
        logs,
        behavior_probabilities=behavior,
        target_probabilities=target,
    )

    # weights = [1.6, 0.4], so SNIPS = 1.6 / 2.0 = 0.8
    assert estimate.value == pytest.approx(0.8)
    assert estimate.weight_sum == pytest.approx(2.0)


def test_snips_equals_logged_mean_when_target_equals_behavior() -> None:
    logs = LoggedBanditData(
        context_indices=np.array([0, 0, 0], dtype=np.int64),
        actions=np.array([0, 1, 0], dtype=np.int64),
        rewards=np.array([1, 0, 1], dtype=np.int64),
        behavior_propensities=np.array([0.5, 0.5, 0.5]),
    )
    behavior = np.array([[0.5, 0.5]])

    estimate = estimate_snips(
        logs,
        behavior_probabilities=behavior,
        target_probabilities=behavior,
    )

    assert estimate.value == pytest.approx(2.0 / 3.0)
