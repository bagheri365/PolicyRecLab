from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LoggedActionIPSEstimate:
    """IPS estimate when target probabilities are known on logged actions."""

    value: float
    importance_weights: FloatArray

    @property
    def mean_weight(self) -> float:
        return float(np.mean(self.importance_weights))


def estimate_logged_action_ips(
    *,
    rewards: FloatArray,
    behavior_propensities: FloatArray,
    target_propensities: FloatArray,
) -> LoggedActionIPSEstimate:
    """Estimate policy value from per-row behavior and target propensities.

    This form is useful for real logs where each impression is a distinct
    context and materializing an n_rounds x n_actions policy matrix would be
    unnecessary.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    behavior = np.asarray(behavior_propensities, dtype=np.float64)
    target = np.asarray(target_propensities, dtype=np.float64)

    if rewards.ndim != 1 or rewards.size == 0:
        raise ValueError("rewards must be a non-empty 1D array")
    if behavior.shape != rewards.shape or target.shape != rewards.shape:
        raise ValueError(
            "rewards, behavior_propensities, and target_propensities "
            "must have the same shape"
        )
    if not (
        np.all(np.isfinite(rewards))
        and np.all(np.isfinite(behavior))
        and np.all(np.isfinite(target))
    ):
        raise ValueError("all estimator inputs must be finite")
    if np.any((rewards < 0.0) | (rewards > 1.0)):
        raise ValueError("rewards must lie in [0, 1]")
    if np.any((behavior <= 0.0) | (behavior > 1.0)):
        raise ValueError("behavior propensities must lie in (0, 1]")
    if np.any((target < 0.0) | (target > 1.0)):
        raise ValueError("target propensities must lie in [0, 1]")

    weights = target / behavior
    return LoggedActionIPSEstimate(
        value=float(np.mean(weights * rewards)),
        importance_weights=weights,
    )
