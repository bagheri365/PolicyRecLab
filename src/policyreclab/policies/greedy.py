from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class GreedyPolicy:
    """Deterministic policy choosing the highest supplied score per context.

    Ties are resolved by ``numpy.argmax``: the first maximal action is chosen.
    In v0.1 the scores are supplied independently of the sampled logged rewards.
    """

    def probabilities(self, scores: FloatArray) -> FloatArray:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
            raise ValueError("scores must be a non-empty 2D array")
        if not np.all(np.isfinite(scores)):
            raise ValueError("scores must be finite")

        probabilities = np.zeros_like(scores, dtype=np.float64)
        greedy_actions = np.argmax(scores, axis=1)
        probabilities[np.arange(scores.shape[0]), greedy_actions] = 1.0
        return probabilities
