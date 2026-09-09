from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyreclab.estimators import estimate_dm, estimate_ips
from policyreclab.evaluation import finite_population_policy_value
from policyreclab.logging import simulate_logged_bandit_data
from policyreclab.policies import EpsilonGreedyPolicy, UniformPolicy
from policyreclab.reward_models import GlobalMeanRewardModel, OracleRewardModel
from policyreclab.simulation import SyntheticBanditEnvironment


@dataclass(frozen=True)
class DirectMethodMetrics:
    mean: float
    bias: float
    variance: float
    rmse: float


@dataclass(frozen=True)
class DirectMethodStudy:
    true_value: float
    ips: DirectMethodMetrics
    oracle_dm: DirectMethodMetrics
    misspecified_dm: DirectMethodMetrics


def _metrics(values: np.ndarray, truth: float) -> DirectMethodMetrics:
    mean = float(np.mean(values))
    return DirectMethodMetrics(
        mean=mean,
        bias=float(mean - truth),
        variance=float(np.var(values, ddof=1)),
        rmse=float(np.sqrt(np.mean((values - truth) ** 2))),
    )


def run_direct_method_study(
    *,
    behavior_epsilon: float = 0.1,
    n_repetitions: int = 40,
    n_users: int = 300,
    n_items: int = 20,
    latent_dim: int = 6,
    n_rounds: int = 12_000,
    environment_seed: int = 81,
    base_log_seed: int = 40_000,
) -> DirectMethodStudy:
    """Compare IPS with oracle and intentionally misspecified DM branches.

    The oracle branch exposes the best-case DM behavior when the conditional
    reward means are known exactly. The global-mean branch is deliberately
    misspecified: it ignores context and action entirely and is fitted only on
    the logged behavior-policy rewards.

    The misspecified model is fitted separately within each repetition. The
    oracle branch is not learned from logs and is explicitly a simulator
    diagnostic rather than a realistic production procedure.
    """
    if behavior_epsilon <= 0.0 or behavior_epsilon > 1.0:
        raise ValueError("behavior_epsilon must be in (0, 1]")
    if n_repetitions < 2:
        raise ValueError("n_repetitions must be at least 2")
    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive")

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
    behavior = EpsilonGreedyPolicy(behavior_epsilon).probabilities(
        env.reward_probabilities
    )
    truth = finite_population_policy_value(env, target)

    oracle_predictions = OracleRewardModel.from_environment(env).predict_all()
    oracle_value = estimate_dm(
        target_probabilities=target,
        predicted_reward_means=oracle_predictions,
    ).value

    ips_values: list[float] = []
    oracle_values: list[float] = []
    misspecified_values: list[float] = []

    for repetition in range(n_repetitions):
        logs = simulate_logged_bandit_data(
            env,
            behavior,
            n_rounds=n_rounds,
            seed=base_log_seed + repetition,
        )

        ips_values.append(
            estimate_ips(
                logs,
                behavior_probabilities=behavior,
                target_probabilities=target,
            ).value
        )

        # Constant across repetitions by construction: exact simulator means.
        oracle_values.append(oracle_value)

        misspecified_model = GlobalMeanRewardModel.fit(
            logs,
            n_contexts=env.n_users,
            n_actions=env.n_items,
        )
        misspecified_values.append(
            estimate_dm(
                target_probabilities=target,
                predicted_reward_means=misspecified_model.predict_all(),
            ).value
        )

    return DirectMethodStudy(
        true_value=float(truth),
        ips=_metrics(np.asarray(ips_values), truth),
        oracle_dm=_metrics(np.asarray(oracle_values), truth),
        misspecified_dm=_metrics(np.asarray(misspecified_values), truth),
    )
