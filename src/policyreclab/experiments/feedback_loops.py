from __future__ import annotations

from dataclasses import dataclass

from policyreclab.feedback import FeedbackLoopResult, run_feedback_loop
from policyreclab.simulation import SyntheticBanditEnvironment


@dataclass(frozen=True)
class FeedbackLoopComparison:
    exploratory: FeedbackLoopResult
    exploitative: FeedbackLoopResult


def run_feedback_loop_comparison(
    *,
    exploratory_epsilon: float = 0.25,
    exploitative_epsilon: float = 0.01,
    n_deployments: int = 8,
    n_rounds_per_deployment: int = 4_000,
    n_users: int = 120,
    n_items: int = 10,
    latent_dim: int = 4,
    environment_seed: int = 111,
    base_seed: int = 80_000,
) -> FeedbackLoopComparison:
    """Compare repeated learning under higher and lower exploration."""
    if exploratory_epsilon <= exploitative_epsilon:
        raise ValueError(
            "exploratory_epsilon must be greater than exploitative_epsilon"
        )

    env = SyntheticBanditEnvironment.generate(
        n_users=n_users,
        n_items=n_items,
        latent_dim=latent_dim,
        seed=environment_seed,
    )

    exploratory = run_feedback_loop(
        env,
        n_deployments=n_deployments,
        n_rounds_per_deployment=n_rounds_per_deployment,
        epsilon=exploratory_epsilon,
        base_seed=base_seed,
    )
    exploitative = run_feedback_loop(
        env,
        n_deployments=n_deployments,
        n_rounds_per_deployment=n_rounds_per_deployment,
        epsilon=exploitative_epsilon,
        base_seed=base_seed + 10_000,
    )

    return FeedbackLoopComparison(
        exploratory=exploratory,
        exploitative=exploitative,
    )
