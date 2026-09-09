from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from policyreclab.simulation.environment import SyntheticBanditEnvironment


FloatArray = NDArray[np.float64]


def _validate_policy_probabilities(
    policy_probabilities: FloatArray,
    *,
    n_contexts: int,
    n_actions: int,
) -> None:
    if policy_probabilities.shape != (n_contexts, n_actions):
        raise ValueError(
            "policy_probabilities must have shape "
            f"({n_contexts}, {n_actions}), got {policy_probabilities.shape}"
        )
    if not np.all(np.isfinite(policy_probabilities)):
        raise ValueError("policy probabilities must be finite")
    if np.any(policy_probabilities < 0.0):
        raise ValueError("policy probabilities must be non-negative")
    if not np.allclose(policy_probabilities.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("each context's policy probabilities must sum to 1")


def finite_population_policy_value(
    environment: SyntheticBanditEnvironment,
    policy_probabilities: FloatArray,
) -> float:
    """Exact finite-population expected reward for a one-step policy.

    For the fixed contexts contained in ``environment``,

        V(pi) = (1 / N) sum_i sum_a pi(a | x_i) mu(x_i, a)

    where mu(x_i, a) is the environment's known Bernoulli mean reward.
    """
    _validate_policy_probabilities(
        policy_probabilities,
        n_contexts=environment.n_users,
        n_actions=environment.n_items,
    )
    expected_reward_per_context = np.sum(
        policy_probabilities * environment.reward_probabilities,
        axis=1,
    )
    return float(np.mean(expected_reward_per_context))


def monte_carlo_policy_value(
    environment: SyntheticBanditEnvironment,
    policy_probabilities: FloatArray,
    *,
    n_rounds: int,
    seed: int = 0,
) -> float:
    """Monte Carlo estimate of policy value using sampled contexts/actions/rewards.

    Contexts are sampled uniformly with replacement from the finite population.
    Actions are sampled from the supplied target policy for that context.
    """
    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive")

    _validate_policy_probabilities(
        policy_probabilities,
        n_contexts=environment.n_users,
        n_actions=environment.n_items,
    )

    rng = np.random.default_rng(seed)
    users = rng.integers(0, environment.n_users, size=n_rounds)

    rewards = np.empty(n_rounds, dtype=np.float64)
    for i, user_index in enumerate(users):
        action = int(
            rng.choice(
                environment.n_items,
                p=policy_probabilities[int(user_index)],
            )
        )
        rewards[i] = environment.sample_reward(
            user_index=int(user_index),
            item_index=action,
            rng=rng,
        )

    return float(np.mean(rewards))
