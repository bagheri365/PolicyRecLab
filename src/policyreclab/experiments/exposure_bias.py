from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyreclab.evaluation import finite_population_policy_value
from policyreclab.logging import simulate_logged_bandit_data
from policyreclab.policies import GreedyPolicy, UniformPolicy
from policyreclab.simulation import SyntheticBanditEnvironment


@dataclass(frozen=True)
class ExposureBiasResult:
    uniform_true_value: float
    greedy_true_value: float
    uniform_logged_mean: float
    greedy_logged_mean: float
    greedy_action_share_uniform_logs: float
    greedy_action_share_greedy_logs: float


def run_exposure_bias_experiment(
    *,
    n_users: int = 500,
    n_items: int = 50,
    latent_dim: int = 8,
    n_rounds: int = 50_000,
    environment_seed: int = 17,
    log_seed: int = 23,
) -> ExposureBiasResult:
    """Compare logs from two policies in the same fixed reward environment.

    The greedy behavior policy uses the simulator's known conditional means as
    scores. This is an oracle *data-generating intervention* used only to make
    exposure differences unambiguous; it is not a learned production model.
    """
    env = SyntheticBanditEnvironment.generate(
        n_users=n_users,
        n_items=n_items,
        latent_dim=latent_dim,
        seed=environment_seed,
    )
    uniform_probs = UniformPolicy().probabilities(
        n_contexts=env.n_users, n_actions=env.n_items
    )
    greedy_probs = GreedyPolicy().probabilities(env.reward_probabilities)

    uniform_logs = simulate_logged_bandit_data(
        env, uniform_probs, n_rounds=n_rounds, seed=log_seed
    )
    greedy_logs = simulate_logged_bandit_data(
        env, greedy_probs, n_rounds=n_rounds, seed=log_seed
    )

    greedy_actions = np.argmax(env.reward_probabilities, axis=1)
    uniform_greedy_share = float(
        np.mean(uniform_logs.actions == greedy_actions[uniform_logs.context_indices])
    )
    greedy_greedy_share = float(
        np.mean(greedy_logs.actions == greedy_actions[greedy_logs.context_indices])
    )

    return ExposureBiasResult(
        uniform_true_value=finite_population_policy_value(env, uniform_probs),
        greedy_true_value=finite_population_policy_value(env, greedy_probs),
        uniform_logged_mean=uniform_logs.mean_reward,
        greedy_logged_mean=greedy_logs.mean_reward,
        greedy_action_share_uniform_logs=uniform_greedy_share,
        greedy_action_share_greedy_logs=greedy_greedy_share,
    )
