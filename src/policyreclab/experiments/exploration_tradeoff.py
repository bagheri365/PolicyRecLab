from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyreclab.diagnostics import analyze_support, summarize_weights
from policyreclab.estimators import estimate_ips
from policyreclab.evaluation import finite_population_policy_value
from policyreclab.logging import simulate_logged_bandit_data
from policyreclab.policies import EpsilonGreedyPolicy, UniformPolicy
from policyreclab.simulation import SyntheticBanditEnvironment


@dataclass(frozen=True)
class ExplorationTradeoffPoint:
    epsilon: float
    behavior_true_value: float
    mean_behavior_logged_reward: float
    target_true_value: float
    mean_ips_target_value: float
    ips_bias: float
    ips_variance: float
    ips_rmse: float
    min_behavior_probability_on_target_support: float
    mean_max_weight: float
    mean_ess_fraction: float


@dataclass(frozen=True)
class ExplorationTradeoffStudy:
    points: tuple[ExplorationTradeoffPoint, ...]


def run_exploration_tradeoff_study(
    *,
    epsilons: tuple[float, ...] = (1.0, 0.5, 0.2, 0.1, 0.05),
    n_repetitions: int = 40,
    n_users: int = 300,
    n_items: int = 20,
    latent_dim: int = 6,
    n_rounds: int = 12_000,
    environment_seed: int = 71,
    base_log_seed: int = 30_000,
) -> ExplorationTradeoffStudy:
    """Measure immediate reward versus future OPE quality.

    The behavior policy is epsilon-greedy with oracle simulator scores. The
    future target is uniform. This target is deliberately different from the
    behavior policy: reducing exploration improves the oracle behavior policy's
    immediate reward while making the uniform target harder to evaluate.

    The experiment demonstrates this controlled setting; it does not assert a
    universal monotonic exploration law for arbitrary environments or targets.
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
    target_truth = finite_population_policy_value(env, target)

    points: list[ExplorationTradeoffPoint] = []

    for epsilon_index, epsilon in enumerate(epsilons):
        behavior = EpsilonGreedyPolicy(epsilon).probabilities(
            env.reward_probabilities
        )
        behavior_truth = finite_population_policy_value(env, behavior)
        support = analyze_support(behavior, target)

        logged_rewards: list[float] = []
        ips_values: list[float] = []
        max_weights: list[float] = []
        ess_fractions: list[float] = []

        for repetition in range(n_repetitions):
            seed = base_log_seed + 100_000 * epsilon_index + repetition
            logs = simulate_logged_bandit_data(
                env,
                behavior,
                n_rounds=n_rounds,
                seed=seed,
            )
            ips = estimate_ips(
                logs,
                behavior_probabilities=behavior,
                target_probabilities=target,
            )
            diagnostics = summarize_weights(ips.importance_weights)

            logged_rewards.append(logs.mean_reward)
            ips_values.append(ips.value)
            max_weights.append(diagnostics.maximum)
            ess_fractions.append(diagnostics.effective_sample_fraction)

        ips_array = np.asarray(ips_values, dtype=np.float64)
        mean_ips = float(np.mean(ips_array))

        points.append(
            ExplorationTradeoffPoint(
                epsilon=float(epsilon),
                behavior_true_value=float(behavior_truth),
                mean_behavior_logged_reward=float(np.mean(logged_rewards)),
                target_true_value=float(target_truth),
                mean_ips_target_value=mean_ips,
                ips_bias=float(mean_ips - target_truth),
                ips_variance=float(np.var(ips_array, ddof=1)),
                ips_rmse=float(
                    np.sqrt(np.mean((ips_array - target_truth) ** 2))
                ),
                min_behavior_probability_on_target_support=float(
                    support.min_positive_behavior_probability_on_target_support
                ),
                mean_max_weight=float(np.mean(max_weights)),
                mean_ess_fraction=float(np.mean(ess_fractions)),
            )
        )

    return ExplorationTradeoffStudy(points=tuple(points))
