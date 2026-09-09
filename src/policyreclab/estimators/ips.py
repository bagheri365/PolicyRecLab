from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from policyreclab.diagnostics import analyze_support
from policyreclab.logging import LoggedBanditData


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class IPSEstimate:
    """Inverse propensity scoring result for a fixed target policy."""

    value: float
    importance_weights: FloatArray

    @property
    def mean_weight(self) -> float:
        return float(np.mean(self.importance_weights))

    @property
    def max_weight(self) -> float:
        return float(np.max(self.importance_weights))


def estimate_ips(
    logged_data: LoggedBanditData,
    *,
    behavior_probabilities: FloatArray,
    target_probabilities: FloatArray,
) -> IPSEstimate:
    """Estimate one-step target-policy value with inverse propensity scoring.

    The estimator is

        V_hat_IPS = (1 / n) sum_i
                    [pi_e(a_i | x_i) / pi_b(a_i | x_i)] r_i.

    This implementation deliberately refuses target/behavior pairs with exact
    contextual support violations. v0.3 studies the standard supported setting;
    weak-overlap variance is deferred to v0.4.
    """
    support = analyze_support(behavior_probabilities, target_probabilities)
    if not support.has_full_support:
        raise ValueError(
            "target policy is not fully supported by the behavior policy; "
            "IPS target value is not nonparametrically identified"
        )

    if logged_data.n_rounds <= 0:
        raise ValueError("logged_data must contain at least one interaction")

    contexts = np.asarray(logged_data.context_indices, dtype=np.int64)
    actions = np.asarray(logged_data.actions, dtype=np.int64)
    rewards = np.asarray(logged_data.rewards, dtype=np.float64)

    n_contexts, n_actions = behavior_probabilities.shape
    if np.any(contexts < 0) or np.any(contexts >= n_contexts):
        raise ValueError("logged context index is outside the policy matrix")
    if np.any(actions < 0) or np.any(actions >= n_actions):
        raise ValueError("logged action index is outside the policy matrix")

    behavior_on_logged_actions = behavior_probabilities[contexts, actions]
    target_on_logged_actions = target_probabilities[contexts, actions]

    # Cross-check the propensity recorded at logging time. OPE should not
    # silently evaluate against a behavior matrix inconsistent with the log.
    if not np.allclose(
        behavior_on_logged_actions,
        logged_data.behavior_propensities,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "behavior_probabilities are inconsistent with logged propensities"
        )

    if np.any(behavior_on_logged_actions <= 0.0):
        raise ValueError("logged behavior propensity must be strictly positive")

    weights = target_on_logged_actions / behavior_on_logged_actions
    value = float(np.mean(weights * rewards))

    return IPSEstimate(
        value=value,
        importance_weights=weights.astype(np.float64, copy=False),
    )
