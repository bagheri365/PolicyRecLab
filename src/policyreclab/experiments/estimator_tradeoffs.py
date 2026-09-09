from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyreclab.estimators import (
    estimate_clipped_ips,
    estimate_ips,
    estimate_snips,
)
from policyreclab.evaluation import finite_population_policy_value
from policyreclab.logging import simulate_logged_bandit_data
from policyreclab.policies import EpsilonGreedyPolicy, UniformPolicy
from policyreclab.simulation import SyntheticBanditEnvironment


@dataclass(frozen=True)
class EstimatorMetrics:
    mean: float
    bias: float
    variance: float
    rmse: float


@dataclass(frozen=True)
class TradeoffPoint:
    epsilon: float
    clip_threshold: float
    true_value: float
    ips: EstimatorMetrics
    clipped_ips: EstimatorMetrics
    snips: EstimatorMetrics
    mean_clipping_fraction: float


@dataclass(frozen=True)
class EstimatorTradeoffStudy:
    points: tuple[TradeoffPoint, ...]


def _metrics(estimates: np.ndarray, truth: float) -> EstimatorMetrics:
    mean = float(np.mean(estimates))
    bias = mean - truth
    variance = float(np.var(estimates, ddof=1))
    rmse = float(np.sqrt(np.mean((estimates - truth) ** 2)))
    return EstimatorMetrics(
        mean=mean,
        bias=float(bias),
        variance=variance,
        rmse=rmse,
    )


def run_estimator_tradeoff_study(
    *,
    epsilons: tuple[float, ...] = (0.2, 0.1, 0.05),
    clip_threshold: float = 10.0,
    n_repetitions: int = 40,
    n_users: int = 300,
    n_items: int = 20,
    latent_dim: int = 6,
    n_rounds: int = 12_000,
    environment_seed: int = 61,
    base_log_seed: int = 20_000,
) -> EstimatorTradeoffStudy:
    """Compare IPS, clipped IPS, and SNIPS under weak overlap."""
    if n_repetitions < 2:
        raise ValueError("n_repetitions must be at least 2")
    if not epsilons:
        raise ValueError("epsilons must be non-empty")
    if any(e <= 0.0 or e > 1.0 for e in epsilons):
        raise ValueError("all epsilons must be in (0, 1]")
    if not np.isfinite(clip_threshold) or clip_threshold <= 0.0:
        raise ValueError("clip_threshold must be finite and positive")

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

    points: list[TradeoffPoint] = []

    for epsilon_index, epsilon in enumerate(epsilons):
        behavior = EpsilonGreedyPolicy(epsilon).probabilities(
            env.reward_probabilities
        )

        ips_values: list[float] = []
        clipped_values: list[float] = []
        snips_values: list[float] = []
        clipping_fractions: list[float] = []

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
            clipped = estimate_clipped_ips(
                logs,
                behavior_probabilities=behavior,
                target_probabilities=target,
                clip_threshold=clip_threshold,
            )
            snips = estimate_snips(
                logs,
                behavior_probabilities=behavior,
                target_probabilities=target,
            )

            ips_values.append(ips.value)
            clipped_values.append(clipped.value)
            snips_values.append(snips.value)
            clipping_fractions.append(clipped.clipping_fraction)

        points.append(
            TradeoffPoint(
                epsilon=float(epsilon),
                clip_threshold=float(clip_threshold),
                true_value=float(truth),
                ips=_metrics(np.asarray(ips_values), truth),
                clipped_ips=_metrics(np.asarray(clipped_values), truth),
                snips=_metrics(np.asarray(snips_values), truth),
                mean_clipping_fraction=float(np.mean(clipping_fractions)),
            )
        )

    return EstimatorTradeoffStudy(points=tuple(points))
