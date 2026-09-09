from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WeightDiagnostics:
    n: int
    mean: float
    std: float
    maximum: float
    p90: float
    p99: float
    effective_sample_size: float
    effective_sample_fraction: float


def effective_sample_size(weights: FloatArray) -> float:
    """Return the common self-normalized weight concentration heuristic.

    ESS = (sum_i w_i)^2 / sum_i w_i^2.

    This is a descriptive weight-concentration diagnostic. It is not a literal
    inferential sample size and does not by itself establish estimator validity.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a non-empty 1D array")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")

    denominator = float(np.sum(weights**2))
    if denominator == 0.0:
        return 0.0

    numerator = float(np.sum(weights)) ** 2
    return numerator / denominator


def summarize_weights(weights: FloatArray) -> WeightDiagnostics:
    weights = np.asarray(weights, dtype=np.float64)
    ess = effective_sample_size(weights)

    return WeightDiagnostics(
        n=int(weights.size),
        mean=float(np.mean(weights)),
        std=float(np.std(weights)),
        maximum=float(np.max(weights)),
        p90=float(np.quantile(weights, 0.90)),
        p99=float(np.quantile(weights, 0.99)),
        effective_sample_size=float(ess),
        effective_sample_fraction=float(ess / weights.size),
    )
