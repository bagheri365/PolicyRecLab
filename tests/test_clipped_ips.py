import numpy as np
import pytest

from policyreclab.estimators import estimate_clipped_ips
from policyreclab.logging import LoggedBanditData


def test_clipped_ips_matches_hand_calculation() -> None:
    logs = LoggedBanditData(
        context_indices=np.array([0, 0], dtype=np.int64),
        actions=np.array([0, 1], dtype=np.int64),
        rewards=np.array([1, 1], dtype=np.int64),
        behavior_propensities=np.array([0.9, 0.1], dtype=np.float64),
    )
    behavior = np.array([[0.9, 0.1]])
    target = np.array([[0.5, 0.5]])

    estimate = estimate_clipped_ips(
        logs,
        behavior_probabilities=behavior,
        target_probabilities=target,
        clip_threshold=2.0,
    )

    # Unclipped weights are 5/9 and 5. The latter is clipped to 2.
    np.testing.assert_allclose(
        estimate.unclipped_weights,
        [5.0 / 9.0, 5.0],
    )
    np.testing.assert_allclose(
        estimate.clipped_weights,
        [5.0 / 9.0, 2.0],
    )
    assert estimate.value == pytest.approx((5.0 / 9.0 + 2.0) / 2.0)
    assert estimate.clipping_fraction == pytest.approx(0.5)


def test_clipped_ips_rejects_invalid_threshold() -> None:
    logs = LoggedBanditData(
        context_indices=np.array([0]),
        actions=np.array([0]),
        rewards=np.array([1]),
        behavior_propensities=np.array([1.0]),
    )
    behavior = np.array([[1.0]])
    target = np.array([[1.0]])

    with pytest.raises(ValueError):
        estimate_clipped_ips(
            logs,
            behavior_probabilities=behavior,
            target_probabilities=target,
            clip_threshold=0.0,
        )
