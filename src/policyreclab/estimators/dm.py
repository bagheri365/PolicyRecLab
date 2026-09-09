from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DMEstimate:
    value: float
    context_values: FloatArray


def estimate_dm(
    *,
    target_probabilities: FloatArray,
    predicted_reward_means: FloatArray,
) -> DMEstimate:
    """Direct-method policy value from a full context-action reward model."""
    target = np.asarray(target_probabilities, dtype=np.float64)
    predictions = np.asarray(predicted_reward_means, dtype=np.float64)

    if target.ndim != 2 or predictions.ndim != 2:
        raise ValueError("target probabilities and predictions must be 2D")
    if target.shape != predictions.shape:
        raise ValueError(
            "target probabilities and reward predictions must have equal shape"
        )
    if target.shape[0] == 0 or target.shape[1] == 0:
        raise ValueError("probability matrix must be non-empty")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(predictions)):
        raise ValueError("inputs must be finite")
    if np.any(target < 0.0):
        raise ValueError("target probabilities must be non-negative")
    if not np.allclose(np.sum(target, axis=1), 1.0):
        raise ValueError("target probability rows must sum to one")

    context_values = np.sum(target * predictions, axis=1)
    return DMEstimate(
        value=float(np.mean(context_values)),
        context_values=context_values.astype(np.float64, copy=False),
    )
