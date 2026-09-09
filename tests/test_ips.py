import numpy as np
import pytest

from policyreclab.estimators import estimate_ips
from policyreclab.logging import LoggedBanditData, simulate_logged_bandit_data
from policyreclab.policies import GreedyPolicy, UniformPolicy
from policyreclab.simulation import SyntheticBanditEnvironment


def test_target_equals_behavior_gives_unit_weights_and_logged_mean() -> None:
    env = SyntheticBanditEnvironment.generate(
        n_users=30, n_items=5, latent_dim=3, seed=2
    )
    behavior = UniformPolicy().probabilities(
        n_contexts=env.n_users,
        n_actions=env.n_items,
    )
    logs = simulate_logged_bandit_data(
        env, behavior, n_rounds=5_000, seed=3
    )

    estimate = estimate_ips(
        logs,
        behavior_probabilities=behavior,
        target_probabilities=behavior,
    )

    np.testing.assert_allclose(estimate.importance_weights, 1.0)
    assert estimate.value == pytest.approx(logs.mean_reward, abs=1e-15)


def test_ips_refuses_exact_support_violation() -> None:
    env = SyntheticBanditEnvironment.generate(
        n_users=20, n_items=4, latent_dim=3, seed=4
    )
    behavior = GreedyPolicy().probabilities(env.reward_probabilities)
    target = UniformPolicy().probabilities(
        n_contexts=env.n_users,
        n_actions=env.n_items,
    )
    logs = simulate_logged_bandit_data(
        env, behavior, n_rounds=1_000, seed=5
    )

    with pytest.raises(ValueError, match="not fully supported"):
        estimate_ips(
            logs,
            behavior_probabilities=behavior,
            target_probabilities=target,
        )


def test_ips_rejects_behavior_matrix_inconsistent_with_logged_propensity() -> None:
    logs = LoggedBanditData(
        context_indices=np.array([0], dtype=np.int64),
        actions=np.array([0], dtype=np.int64),
        rewards=np.array([1], dtype=np.int64),
        behavior_propensities=np.array([0.25], dtype=np.float64),
    )
    behavior = np.array([[0.5, 0.5]], dtype=np.float64)
    target = np.array([[0.5, 0.5]], dtype=np.float64)

    with pytest.raises(ValueError, match="inconsistent"):
        estimate_ips(
            logs,
            behavior_probabilities=behavior,
            target_probabilities=target,
        )


def test_ips_matches_hand_calculation() -> None:
    logs = LoggedBanditData(
        context_indices=np.array([0, 0], dtype=np.int64),
        actions=np.array([0, 1], dtype=np.int64),
        rewards=np.array([1, 1], dtype=np.int64),
        behavior_propensities=np.array([0.5, 0.5], dtype=np.float64),
    )
    behavior = np.array([[0.5, 0.5]], dtype=np.float64)
    target = np.array([[0.8, 0.2]], dtype=np.float64)

    estimate = estimate_ips(
        logs,
        behavior_probabilities=behavior,
        target_probabilities=target,
    )

    np.testing.assert_allclose(estimate.importance_weights, [1.6, 0.4])
    assert estimate.value == pytest.approx(1.0)
