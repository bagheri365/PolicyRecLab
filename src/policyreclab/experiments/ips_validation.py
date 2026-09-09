from __future__ import annotations

from dataclasses import dataclass

from policyreclab.estimators import estimate_ips
from policyreclab.evaluation import finite_population_policy_value
from policyreclab.logging import simulate_logged_bandit_data
from policyreclab.policies import EpsilonGreedyPolicy, GreedyPolicy, UniformPolicy
from policyreclab.simulation import SyntheticBanditEnvironment


@dataclass(frozen=True)
class IPSValidationResult:
    target_true_value: float
    behavior_logged_mean: float
    ips_value: float
    ips_absolute_error: float
    mean_importance_weight: float
    max_importance_weight: float


def run_ips_validation_experiment(
    *,
    n_users: int = 400,
    n_items: int = 20,
    latent_dim: int = 6,
    epsilon: float = 0.3,
    n_rounds: int = 100_000,
    environment_seed: int = 41,
    log_seed: int = 43,
) -> IPSValidationResult:
    """Evaluate a greedy target from epsilon-greedy behavior logs.

    Epsilon-greedy behavior has positive probability on every action, so the
    deterministic greedy target is fully supported. The simulator supplies the
    target policy's exact finite-population value for validation.
    """
    env = SyntheticBanditEnvironment.generate(
        n_users=n_users,
        n_items=n_items,
        latent_dim=latent_dim,
        seed=environment_seed,
    )
    behavior = EpsilonGreedyPolicy(epsilon).probabilities(
        env.reward_probabilities
    )
    target = GreedyPolicy().probabilities(env.reward_probabilities)

    logs = simulate_logged_bandit_data(
        env,
        behavior,
        n_rounds=n_rounds,
        seed=log_seed,
    )
    ips = estimate_ips(
        logs,
        behavior_probabilities=behavior,
        target_probabilities=target,
    )
    truth = finite_population_policy_value(env, target)

    return IPSValidationResult(
        target_true_value=truth,
        behavior_logged_mean=logs.mean_reward,
        ips_value=ips.value,
        ips_absolute_error=abs(ips.value - truth),
        mean_importance_weight=ips.mean_weight,
        max_importance_weight=ips.max_weight,
    )
