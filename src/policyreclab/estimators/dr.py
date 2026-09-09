from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from policyreclab.diagnostics import analyze_support
from policyreclab.logging import LoggedBanditData


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DREstimate:
    value: float
    direct_terms: FloatArray
    correction_terms: FloatArray
    importance_weights: FloatArray


def estimate_dr(
    logged_data: LoggedBanditData,
    *,
    behavior_probabilities: FloatArray,
    target_probabilities: FloatArray,
    predicted_reward_means: FloatArray,
) -> DREstimate:
    """Doubly robust policy-value estimator for one-step contextual bandits.

    This implementation uses the supplied behavior probabilities directly.
    v0.8's primary experiment treats them as known and correct.
    """
    behavior = np.asarray(behavior_probabilities, dtype=np.float64)
    target = np.asarray(target_probabilities, dtype=np.float64)
    predictions = np.asarray(predicted_reward_means, dtype=np.float64)

    if behavior.ndim != 2 or target.ndim != 2 or predictions.ndim != 2:
        raise ValueError("behavior, target, and predictions must be 2D")
    if behavior.shape != target.shape or behavior.shape != predictions.shape:
        raise ValueError(
            "behavior, target, and reward predictions must have equal shape"
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError("reward predictions must be finite")

    support = analyze_support(behavior, target)
    if not support.has_full_support:
        raise ValueError(
            "target policy is not fully supported by the behavior policy"
        )

    contexts = np.asarray(logged_data.context_indices, dtype=np.int64)
    actions = np.asarray(logged_data.actions, dtype=np.int64)
    rewards = np.asarray(logged_data.rewards, dtype=np.float64)

    if contexts.size == 0:
        raise ValueError("logged data must be non-empty")
    if np.any(contexts < 0) or np.any(contexts >= behavior.shape[0]):
        raise ValueError("logged context index is out of range")
    if np.any(actions < 0) or np.any(actions >= behavior.shape[1]):
        raise ValueError("logged action index is out of range")

    behavior_on_logged = behavior[contexts, actions]
    target_on_logged = target[contexts, actions]

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

    # m_hat(x, pi_e) for every context in the fixed finite population.
    target_model_values = np.sum(target * predictions, axis=1)

    # Logged contexts are sampled uniformly from that finite population, so the
    # DR direct term is evaluated at each logged context.
    direct_terms = target_model_values[contexts]
    logged_predictions = predictions[contexts, actions]
    weights = target_on_logged / behavior_on_logged
    correction_terms = weights * (rewards - logged_predictions)

    return DREstimate(
        value=float(np.mean(direct_terms + correction_terms)),
        direct_terms=direct_terms.astype(np.float64, copy=False),
        correction_terms=correction_terms.astype(np.float64, copy=False),
        importance_weights=weights.astype(np.float64, copy=False),
    )
