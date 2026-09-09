from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class EpsilonGreedyPolicy:
    """Epsilon-greedy policy over a score matrix.

    For each context,

        pi(a | x)
        =
        (1 - epsilon) * 1{a = a*(x)}
        + epsilon / |A|.

    Thus every action has positive probability whenever ``epsilon > 0``.
    """

    def __init__(self, epsilon: float) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = float(epsilon)

    def probabilities(self, scores: FloatArray) -> FloatArray:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
            raise ValueError("scores must be a non-empty 2D array")
        if not np.all(np.isfinite(scores)):
            raise ValueError("scores must be finite")

        n_contexts, n_actions = scores.shape
        probabilities = np.full(
            (n_contexts, n_actions),
            self.epsilon / n_actions,
            dtype=np.float64,
        )
        greedy_actions = np.argmax(scores, axis=1)
        probabilities[np.arange(n_contexts), greedy_actions] += 1.0 - self.epsilon
        return probabilities
