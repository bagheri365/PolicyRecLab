from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class UniformPolicy:
    """Uniform stochastic policy over all actions."""

    def probabilities(self, *, n_contexts: int, n_actions: int) -> FloatArray:
        if n_contexts <= 0 or n_actions <= 0:
            raise ValueError("n_contexts and n_actions must be positive")
        return np.full(
            (n_contexts, n_actions),
            1.0 / n_actions,
            dtype=np.float64,
        )

    def sample_action(
        self,
        *,
        n_actions: int,
        rng: np.random.Generator,
    ) -> int:
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        return int(rng.integers(0, n_actions))
