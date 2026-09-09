from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyreclab.estimators import estimate_dm, estimate_dr, estimate_ips
from policyreclab.evaluation import finite_population_policy_value
from policyreclab.logging import simulate_logged_bandit_data
from policyreclab.policies import EpsilonGreedyPolicy, UniformPolicy
from policyreclab.reward_models import GlobalMeanRewardModel, OracleRewardModel
from policyreclab.simulation import SyntheticBanditEnvironment


@dataclass(frozen=True)
class DRMetrics:
    mean: float
    bias: float
    variance: float
    rmse: float


@dataclass(frozen=True)
class DoublyRobustStudy:
    true_value: float
    ips: DRMetrics
    misspecified_dm: DRMetrics
    oracle_dr: DRMetrics
    misspecified_dr: DRMetrics


def _metrics(values: np.ndarray, truth: float) -> DRMetrics:
    mean = float(np.mean(values))
    return DRMetrics(
        mean=mean,
        bias=float(mean - truth),
        variance=float(np.var(values, ddof=1)),
        rmse=float(np.sqrt(np.mean((values - truth) ** 2))),
    )


def run_doubly_robust_study(
    *,
    behavior_epsilon: float = 0.1,
    n_repetitions: int = 50,
    n_users: int = 300,
    n_items: int = 20,
    latent_dim: int = 6,
    n_rounds: int = 12_000,
    environment_seed: int = 91,
    base_log_seed: int = 50_000,
) -> DoublyRobustStudy:
    """Compare DR under exact versus deliberately misspecified reward models.

    Behavior propensities are supplied by the known synthetic logging policy and
    are correct in this milestone. Therefore the experiment isolates robustness
    to reward-model misspecification; it is not yet the full two-nuisance
    estimated-propensity demonstration.
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

    ips_values: list[float] = []
    misspecified_dm_values: list[float] = []
    oracle_dr_values: list[float] = []
    misspecified_dr_values: list[float] = []

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

        misspecified_model = GlobalMeanRewardModel.fit(
            logs,
            n_contexts=env.n_users,
            n_actions=env.n_items,
        )
        misspecified_predictions = misspecified_model.predict_all()

        misspecified_dm_values.append(
            estimate_dm(
                target_probabilities=target,
                predicted_reward_means=misspecified_predictions,
            ).value
        )
        oracle_dr_values.append(
            estimate_dr(
                logs,
                behavior_probabilities=behavior,
                target_probabilities=target,
                predicted_reward_means=oracle_predictions,
            ).value
        )
        misspecified_dr_values.append(
            estimate_dr(
                logs,
                behavior_probabilities=behavior,
                target_probabilities=target,
                predicted_reward_means=misspecified_predictions,
            ).value
        )

    return DoublyRobustStudy(
        true_value=float(truth),
        ips=_metrics(np.asarray(ips_values), truth),
        misspecified_dm=_metrics(np.asarray(misspecified_dm_values), truth),
        oracle_dr=_metrics(np.asarray(oracle_dr_values), truth),
        misspecified_dr=_metrics(np.asarray(misspecified_dr_values), truth),
    )
