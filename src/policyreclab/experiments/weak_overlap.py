from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from policyreclab.diagnostics import summarize_weights
from policyreclab.estimators import estimate_ips
from policyreclab.evaluation import finite_population_policy_value
from policyreclab.logging import simulate_logged_bandit_data
from policyreclab.policies import EpsilonGreedyPolicy, UniformPolicy
from policyreclab.simulation import SyntheticBanditEnvironment


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WeakOverlapPoint:
    epsilon: float
    min_behavior_probability: float
    true_value: float
    mean_ips: float
    bias: float
    variance: float
    rmse: float
    mean_max_weight: float
    mean_p99_weight: float
    mean_ess: float
    mean_ess_fraction: float


@dataclass(frozen=True)
class WeakOverlapStudy:
    points: tuple[WeakOverlapPoint, ...]


def run_weak_overlap_study(
    *,
    epsilons: tuple[float, ...] = (1.0, 0.5, 0.2, 0.1, 0.05),
    n_repetitions: int = 40,
    n_users: int = 300,
    n_items: int = 20,
    latent_dim: int = 6,
    n_rounds: int = 12_000,
    environment_seed: int = 51,
    base_log_seed: int = 10_000,
) -> WeakOverlapStudy:
    """Study IPS reliability as overlap weakens but remains strictly positive.

    The behavior policy is epsilon-greedy with respect to the environment's
    oracle reward means. The target is uniform. As epsilon decreases, the
    behavior probability assigned to non-greedy actions is epsilon / |A|.

    Thus contextual support remains intact for every epsilon > 0, while
    importance weights for non-greedy target actions grow like 1 / epsilon.
    """
    if n_repetitions < 2:
        raise ValueError("n_repetitions must be at least 2")
    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive")
    if not epsilons:
        raise ValueError("epsilons must be non-empty")
    if any(e <= 0.0 or e > 1.0 for e in epsilons):
        raise ValueError("all epsilons must be in (0, 1]")

    env = SyntheticBanditEnvironment.generate(
        n_users=n_users,
        n_items=n_items,
        latent_dim=latent_dim,
        seed=environment_seed,
    )
    target = UniformPolicy().probabilities(
        n_contexts=env.n_users,
        n_actions=env.n_items,
    )
    truth = finite_population_policy_value(env, target)

    points: list[WeakOverlapPoint] = []

    for epsilon_index, epsilon in enumerate(epsilons):
        behavior = EpsilonGreedyPolicy(epsilon).probabilities(
            env.reward_probabilities
        )
        estimates = []
        max_weights = []
        p99_weights = []
        esses = []
        ess_fractions = []

        for repetition in range(n_repetitions):
            seed = (
                base_log_seed
                + 100_000 * epsilon_index
                + repetition
            )
            logs = simulate_logged_bandit_data(
                env,
                behavior,
                n_rounds=n_rounds,
                seed=seed,
            )
            estimate = estimate_ips(
                logs,
                behavior_probabilities=behavior,
                target_probabilities=target,
            )
            diagnostics = summarize_weights(estimate.importance_weights)

            estimates.append(estimate.value)
            max_weights.append(diagnostics.maximum)
            p99_weights.append(diagnostics.p99)
            esses.append(diagnostics.effective_sample_size)
            ess_fractions.append(diagnostics.effective_sample_fraction)

        estimates_array = np.asarray(estimates, dtype=np.float64)
        mean_ips = float(np.mean(estimates_array))
        bias = mean_ips - truth
        variance = float(np.var(estimates_array, ddof=1))
        rmse = float(np.sqrt(np.mean((estimates_array - truth) ** 2)))

        points.append(
            WeakOverlapPoint(
                epsilon=float(epsilon),
                min_behavior_probability=float(epsilon / n_items),
                true_value=float(truth),
                mean_ips=mean_ips,
                bias=float(bias),
                variance=variance,
                rmse=rmse,
                mean_max_weight=float(np.mean(max_weights)),
                mean_p99_weight=float(np.mean(p99_weights)),
                mean_ess=float(np.mean(esses)),
                mean_ess_fraction=float(np.mean(ess_fractions)),
            )
        )

    return WeakOverlapStudy(points=tuple(points))
