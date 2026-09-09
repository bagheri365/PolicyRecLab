from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _sigmoid(x: FloatArray) -> FloatArray:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


@dataclass(frozen=True)
class SyntheticBanditEnvironment:
    """Finite contextual-bandit environment with known reward probabilities.

    The environment is intentionally simple for v0.0:
    - each user is one fixed context,
    - each item is one action,
    - rewards are Bernoulli,
    - there are no slate, sequential, or interference effects.

    Reward probabilities are generated as

        sigmoid(user @ item.T / sqrt(latent_dim) + item_bias)

    The sqrt(latent_dim) scaling keeps logit variance from growing
    mechanically with the embedding dimension.
    """

    user_features: FloatArray
    item_features: FloatArray
    item_bias: FloatArray
    reward_probabilities: FloatArray

    @classmethod
    def generate(
        cls,
        *,
        n_users: int = 1_000,
        n_items: int = 100,
        latent_dim: int = 8,
        seed: int = 0,
    ) -> "SyntheticBanditEnvironment":
        if n_users <= 0 or n_items <= 0 or latent_dim <= 0:
            raise ValueError("n_users, n_items, and latent_dim must all be positive")

        rng = np.random.default_rng(seed)
        user_features = rng.normal(size=(n_users, latent_dim))
        item_features = rng.normal(size=(n_items, latent_dim))
        item_bias = rng.normal(loc=0.0, scale=0.5, size=n_items)

        logits = (
            user_features @ item_features.T / np.sqrt(float(latent_dim))
            + item_bias[None, :]
        )
        reward_probabilities = _sigmoid(logits)

        return cls(
            user_features=user_features.astype(np.float64, copy=False),
            item_features=item_features.astype(np.float64, copy=False),
            item_bias=item_bias.astype(np.float64, copy=False),
            reward_probabilities=reward_probabilities.astype(np.float64, copy=False),
        )

    @property
    def n_users(self) -> int:
        return int(self.reward_probabilities.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.reward_probabilities.shape[1])

    def sample_reward(
        self,
        *,
        user_index: int,
        item_index: int,
        rng: np.random.Generator,
    ) -> int:
        p = float(self.reward_probabilities[user_index, item_index])
        return int(rng.random() < p)
