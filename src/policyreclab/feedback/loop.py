from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from policyreclab.logging import simulate_logged_bandit_data
from policyreclab.simulation import SyntheticBanditEnvironment


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FeedbackRound:
    round_index: int
    policy_value: float
    exposure_entropy: float
    exposure_gini: float
    minimum_action_share: float
    maximum_action_share: float
    uniform_target_min_support: float


@dataclass(frozen=True)
class FeedbackLoopResult:
    rounds: tuple[FeedbackRound, ...]


def _policy_value(env: SyntheticBanditEnvironment, policy: FloatArray) -> float:
    return float(np.mean(np.sum(policy * env.reward_probabilities, axis=1)))


def _exposure_stats(actions: np.ndarray, n_actions: int) -> tuple[float, float, float, float]:
    counts = np.bincount(actions, minlength=n_actions).astype(np.float64)
    shares = counts / np.sum(counts)

    positive = shares[shares > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    if n_actions > 1:
        entropy /= float(np.log(n_actions))

    sorted_shares = np.sort(shares)
    cumulative = np.cumsum(sorted_shares)
    gini = float(
        (n_actions + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n_actions
    )

    return (
        entropy,
        gini,
        float(np.min(shares)),
        float(np.max(shares)),
    )


def _learn_action_scores(
    *,
    context_indices: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    n_contexts: int,
    n_actions: int,
    prior_mean: float,
    prior_strength: float,
) -> FloatArray:
    successes = np.full(
        (n_contexts, n_actions),
        prior_mean * prior_strength,
        dtype=np.float64,
    )
    totals = np.full(
        (n_contexts, n_actions),
        prior_strength,
        dtype=np.float64,
    )

    np.add.at(successes, (context_indices, actions), rewards)
    np.add.at(totals, (context_indices, actions), 1.0)
    return successes / totals


def _epsilon_greedy_from_scores(scores: FloatArray, epsilon: float) -> FloatArray:
    n_contexts, n_actions = scores.shape
    probs = np.full(
        (n_contexts, n_actions),
        epsilon / n_actions,
        dtype=np.float64,
    )
    best = np.argmax(scores, axis=1)
    probs[np.arange(n_contexts), best] += 1.0 - epsilon
    return probs


def run_feedback_loop(
    env: SyntheticBanditEnvironment,
    *,
    n_deployments: int = 8,
    n_rounds_per_deployment: int = 4_000,
    epsilon: float = 0.05,
    prior_strength: float = 2.0,
    base_seed: int = 70_000,
) -> FeedbackLoopResult:
    """Simulate repeated policy-generated data and policy updating.

    Each deployment logs data from the current policy. A simple smoothed
    context-action reward table is then fitted to that deployment's data only,
    and the next policy is epsilon-greedy with respect to those learned scores.

    This is a controlled feedback-loop mechanism, not a production learner.
    """
    if n_deployments < 1:
        raise ValueError("n_deployments must be positive")
    if n_rounds_per_deployment <= 0:
        raise ValueError("n_rounds_per_deployment must be positive")
    if epsilon < 0.0 or epsilon > 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if prior_strength <= 0.0:
        raise ValueError("prior_strength must be positive")

    n_contexts = env.n_users
    n_actions = env.n_items
    policy = np.full((n_contexts, n_actions), 1.0 / n_actions, dtype=np.float64)
    prior_mean = float(np.mean(env.reward_probabilities))

    rounds: list[FeedbackRound] = []

    for deployment in range(n_deployments):
        logs = simulate_logged_bandit_data(
            env,
            policy,
            n_rounds=n_rounds_per_deployment,
            seed=base_seed + deployment,
        )

        entropy, gini, minimum_share, maximum_share = _exposure_stats(
            logs.actions,
            n_actions,
        )

        # For a uniform evaluation target, every action must have positive
        # logging probability. This diagnostic records the smallest behavior
        # probability anywhere in the context-action table.
        min_support = float(np.min(policy))

        rounds.append(
            FeedbackRound(
                round_index=deployment,
                policy_value=_policy_value(env, policy),
                exposure_entropy=entropy,
                exposure_gini=gini,
                minimum_action_share=minimum_share,
                maximum_action_share=maximum_share,
                uniform_target_min_support=min_support,
            )
        )

        learned_scores = _learn_action_scores(
            context_indices=logs.context_indices,
            actions=logs.actions,
            rewards=logs.rewards,
            n_contexts=n_contexts,
            n_actions=n_actions,
            prior_mean=prior_mean,
            prior_strength=prior_strength,
        )
        policy = _epsilon_greedy_from_scores(learned_scores, epsilon)

    return FeedbackLoopResult(rounds=tuple(rounds))
