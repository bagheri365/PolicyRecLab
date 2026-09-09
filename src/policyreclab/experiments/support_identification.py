from __future__ import annotations

from dataclasses import dataclass

from policyreclab.diagnostics import SupportReport, analyze_support
from policyreclab.policies import (
    EpsilonGreedyPolicy,
    GreedyPolicy,
    UniformPolicy,
)
from policyreclab.simulation import SyntheticBanditEnvironment


@dataclass(frozen=True)
class SupportIdentificationResult:
    uniform_to_greedy: SupportReport
    epsilon_greedy_to_uniform: SupportReport
    greedy_to_uniform: SupportReport


def run_support_identification_experiment(
    *,
    n_users: int = 200,
    n_items: int = 20,
    latent_dim: int = 6,
    epsilon: float = 0.1,
    seed: int = 31,
) -> SupportIdentificationResult:
    """Construct three controlled support cases in one fixed environment."""
    env = SyntheticBanditEnvironment.generate(
        n_users=n_users,
        n_items=n_items,
        latent_dim=latent_dim,
        seed=seed,
    )

    uniform = UniformPolicy().probabilities(
        n_contexts=env.n_users,
        n_actions=env.n_items,
    )
    greedy = GreedyPolicy().probabilities(env.reward_probabilities)
    epsilon_greedy = EpsilonGreedyPolicy(epsilon).probabilities(
        env.reward_probabilities
    )

    return SupportIdentificationResult(
        uniform_to_greedy=analyze_support(uniform, greedy),
        epsilon_greedy_to_uniform=analyze_support(epsilon_greedy, uniform),
        greedy_to_uniform=analyze_support(greedy, uniform),
    )
