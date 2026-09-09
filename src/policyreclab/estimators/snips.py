from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from policyreclab.diagnostics import analyze_support
from policyreclab.logging import LoggedBanditData


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SNIPSEstimate:
    value: float
    importance_weights: FloatArray
    weight_sum: float


def estimate_snips(
    logged_data: LoggedBanditData,
    *,
    behavior_probabilities: FloatArray,
    target_probabilities: FloatArray,
) -> SNIPSEstimate:
    """Self-normalized inverse propensity scoring.

    V_hat_SNIPS = sum_i w_i r_i / sum_i w_i.

    SNIPS is generally finite-sample biased, although it can reduce variance and
    is consistent under standard regularity conditions.
    """
    support = analyze_support(behavior_probabilities, target_probabilities)
    if not support.has_full_support:
        raise ValueError(
            "target policy is not fully supported by the behavior policy"
        )

    contexts = np.asarray(logged_data.context_indices, dtype=np.int64)
    actions = np.asarray(logged_data.actions, dtype=np.int64)
    rewards = np.asarray(logged_data.rewards, dtype=np.float64)

    behavior_on_logged = behavior_probabilities[contexts, actions]
    target_on_logged = target_probabilities[contexts, actions]

    if not np.allclose(
        behavior_on_logged,
        logged_data.behavior_propensities,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "behavior_probabilities are inconsistent with logged propensities"
        )
    if np.any(behavior_on_logged <= 0.0):
        raise ValueError("logged behavior propensity must be strictly positive")

    weights = target_on_logged / behavior_on_logged
    denominator = float(np.sum(weights))
    if denominator <= 0.0:
        raise ValueError("sum of importance weights must be positive")

    value = float(np.sum(weights * rewards) / denominator)

    return SNIPSEstimate(
        value=value,
        importance_weights=weights.astype(np.float64, copy=False),
        weight_sum=denominator,
    )
