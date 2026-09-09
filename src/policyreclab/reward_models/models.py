from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from policyreclab.logging import LoggedBanditData
from policyreclab.simulation import SyntheticBanditEnvironment


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OracleRewardModel:
    """Synthetic benchmark model that returns the simulator's true means.

    This is a diagnostic upper bound, not a learnable production estimator.
    """

    reward_means: FloatArray

    @classmethod
    def from_environment(
        cls,
        environment: SyntheticBanditEnvironment,
    ) -> "OracleRewardModel":
        return cls(
            reward_means=np.asarray(
                environment.reward_probabilities,
                dtype=np.float64,
            ).copy()
        )

    def predict_all(self) -> FloatArray:
        return self.reward_means.copy()


@dataclass(frozen=True)
class GlobalMeanRewardModel:
    """Intentionally misspecified reward model with no context/action features."""

    mean_reward: float
    n_contexts: int
    n_actions: int

    @classmethod
    def fit(
        cls,
        logged_data: LoggedBanditData,
        *,
        n_contexts: int,
        n_actions: int,
    ) -> "GlobalMeanRewardModel":
        if logged_data.n_rounds == 0:
            raise ValueError("cannot fit reward model on empty logged data")
        if n_contexts <= 0 or n_actions <= 0:
            raise ValueError("n_contexts and n_actions must be positive")

        return cls(
            mean_reward=float(np.mean(logged_data.rewards)),
            n_contexts=int(n_contexts),
            n_actions=int(n_actions),
        )

    def predict_all(self) -> FloatArray:
        return np.full(
            (self.n_contexts, self.n_actions),
            self.mean_reward,
            dtype=np.float64,
        )
