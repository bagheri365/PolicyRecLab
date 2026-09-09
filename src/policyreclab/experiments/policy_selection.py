from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from policyreclab.estimators import estimate_ips
from policyreclab.evaluation import finite_population_policy_value
from policyreclab.logging import simulate_logged_bandit_data
from policyreclab.policies import UniformPolicy
from policyreclab.simulation import SyntheticBanditEnvironment


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PolicySelectionPoint:
    n_candidates: int
    mean_selection_estimate: float
    mean_selected_true_value: float
    mean_holdout_estimate: float
    mean_selection_optimism: float
    mean_holdout_error: float
    mean_selected_regret: float


@dataclass(frozen=True)
class PolicySelectionStudy:
    points: tuple[PolicySelectionPoint, ...]


def _candidate_policies(
    env: SyntheticBanditEnvironment,
    *,
    n_candidates: int,
    candidate_seed: int,
) -> tuple[FloatArray, ...]:
    """Create fixed stochastic candidate policies with varying quality.

    Each candidate uses a noisy version of the simulator reward surface as its
    score matrix. Softmax temperature and score noise vary across candidates.
    Candidates are fixed before any selection/evaluation logs are generated.
    """
    rng = np.random.default_rng(candidate_seed)
    policies: list[FloatArray] = []

    for index in range(n_candidates):
        noise_scale = 0.08 + 0.45 * index / max(n_candidates - 1, 1)
        temperature = 0.10 + 0.30 * index / max(n_candidates - 1, 1)
        noisy_scores = env.reward_probabilities + rng.normal(
            0.0,
            noise_scale,
            size=env.reward_probabilities.shape,
        )
        centered = noisy_scores - np.max(noisy_scores, axis=1, keepdims=True)
        exp_scores = np.exp(centered / temperature)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        policies.append(probabilities.astype(np.float64, copy=False))

    return tuple(policies)


def run_policy_selection_study(
    *,
    candidate_counts: tuple[int, ...] = (2, 5, 10, 20),
    n_repetitions: int = 60,
    n_users: int = 250,
    n_items: int = 12,
    latent_dim: int = 5,
    n_rounds_selection: int = 3_000,
    n_rounds_holdout: int = 8_000,
    environment_seed: int = 101,
    candidate_seed: int = 102,
    base_log_seed: int = 60_000,
) -> PolicySelectionStudy:
    """Show selection optimism when OPE selects and reports on the same log.

    Candidate policies are fixed independently of the logged samples. A
    uniform behavior policy gives exact positive support to every candidate.

    Within each repetition:
      1. evaluate all candidates with IPS on one selection log;
      2. choose the candidate with the largest selection estimate;
      3. compare that estimate with the selected policy's exact simulator value;
      4. evaluate only the already-selected policy on a fresh independent log.

    The holdout is not used to re-select the policy.
    """
    if not candidate_counts:
        raise ValueError("candidate_counts must be non-empty")
    if any(count < 1 for count in candidate_counts):
        raise ValueError("candidate counts must be positive")
    if n_repetitions < 2:
        raise ValueError("n_repetitions must be at least 2")
    if n_rounds_selection <= 0 or n_rounds_holdout <= 0:
        raise ValueError("log sizes must be positive")

    max_candidates = max(candidate_counts)
    env = SyntheticBanditEnvironment.generate(
        n_users=n_users,
        n_items=n_items,
        latent_dim=latent_dim,
        seed=environment_seed,
    )
    behavior = UniformPolicy().probabilities(
        n_contexts=env.n_users,
        n_actions=env.n_items,
    )
    candidates = _candidate_policies(
        env,
        n_candidates=max_candidates,
        candidate_seed=candidate_seed,
    )
    true_values = np.asarray(
        [finite_population_policy_value(env, policy) for policy in candidates],
        dtype=np.float64,
    )

    points: list[PolicySelectionPoint] = []

    for count_index, n_candidates in enumerate(candidate_counts):
        selection_estimates: list[float] = []
        selected_truths: list[float] = []
        holdout_estimates: list[float] = []
        selection_optimism: list[float] = []
        holdout_errors: list[float] = []
        selected_regrets: list[float] = []

        candidate_subset = candidates[:n_candidates]
        subset_truths = true_values[:n_candidates]
        best_true_value = float(np.max(subset_truths))

        for repetition in range(n_repetitions):
            selection_seed = (
                base_log_seed + 1_000_000 * count_index + 2 * repetition
            )
            holdout_seed = selection_seed + 1

            selection_logs = simulate_logged_bandit_data(
                env,
                behavior,
                n_rounds=n_rounds_selection,
                seed=selection_seed,
            )

            estimates = np.asarray(
                [
                    estimate_ips(
                        selection_logs,
                        behavior_probabilities=behavior,
                        target_probabilities=policy,
                    ).value
                    for policy in candidate_subset
                ],
                dtype=np.float64,
            )
            selected_index = int(np.argmax(estimates))
            selected_policy = candidate_subset[selected_index]
            selected_estimate = float(estimates[selected_index])
            selected_truth = float(subset_truths[selected_index])

            # Fresh data are generated only after selection. The selected policy
            # is fixed before this holdout is inspected.
            holdout_logs = simulate_logged_bandit_data(
                env,
                behavior,
                n_rounds=n_rounds_holdout,
                seed=holdout_seed,
            )
            holdout_estimate = estimate_ips(
                holdout_logs,
                behavior_probabilities=behavior,
                target_probabilities=selected_policy,
            ).value

            selection_estimates.append(selected_estimate)
            selected_truths.append(selected_truth)
            holdout_estimates.append(holdout_estimate)
            selection_optimism.append(selected_estimate - selected_truth)
            holdout_errors.append(holdout_estimate - selected_truth)
            selected_regrets.append(best_true_value - selected_truth)

        points.append(
            PolicySelectionPoint(
                n_candidates=int(n_candidates),
                mean_selection_estimate=float(np.mean(selection_estimates)),
                mean_selected_true_value=float(np.mean(selected_truths)),
                mean_holdout_estimate=float(np.mean(holdout_estimates)),
                mean_selection_optimism=float(np.mean(selection_optimism)),
                mean_holdout_error=float(np.mean(holdout_errors)),
                mean_selected_regret=float(np.mean(selected_regrets)),
            )
        )

    return PolicySelectionStudy(points=tuple(points))
