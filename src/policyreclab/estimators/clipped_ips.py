from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from policyreclab.diagnostics import analyze_support
from policyreclab.logging import LoggedBanditData


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ClippedIPSEstimate:
    value: float
    unclipped_weights: FloatArray
    clipped_weights: FloatArray
    clip_threshold: float

    @property
    def clipping_fraction(self) -> float:
        return float(np.mean(self.unclipped_weights > self.clip_threshold))


def estimate_clipped_ips(
    logged_data: LoggedBanditData,
    *,
    behavior_probabilities: FloatArray,
    target_probabilities: FloatArray,
    clip_threshold: float,
) -> ClippedIPSEstimate:
    """Estimate target-policy value with clipped importance weights.

    Clipping replaces each weight w_i by min(w_i, c). This can reduce variance,
    but generally introduces bias. The threshold is therefore an explicit
    analysis choice rather than a universally optimal correction.
    """
    if not np.isfinite(clip_threshold) or clip_threshold <= 0.0:
        raise ValueError("clip_threshold must be finite and positive")

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
    clipped = np.minimum(weights, clip_threshold)
    value = float(np.mean(clipped * rewards))

    return ClippedIPSEstimate(
        value=value,
        unclipped_weights=weights.astype(np.float64, copy=False),
        clipped_weights=clipped.astype(np.float64, copy=False),
        clip_threshold=float(clip_threshold),
    )
