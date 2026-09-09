import numpy as np
import pytest

from policyreclab.logging import simulate_logged_bandit_data
from policyreclab.policies import UniformPolicy
from policyreclab.simulation import SyntheticBanditEnvironment


def test_logged_data_records_chosen_action_propensity() -> None:
    env = SyntheticBanditEnvironment.generate(
        n_users=20, n_items=5, latent_dim=3, seed=4
    )
    probabilities = UniformPolicy().probabilities(
        n_contexts=env.n_users, n_actions=env.n_items
    )
    logs = simulate_logged_bandit_data(
        env, probabilities, n_rounds=2_000, seed=9
    )

    assert logs.n_rounds == 2_000
    np.testing.assert_allclose(logs.behavior_propensities, 0.2)
    assert set(np.unique(logs.rewards)).issubset({0, 1})


def test_logging_is_reproducible() -> None:
    env = SyntheticBanditEnvironment.generate(
        n_users=10, n_items=4, latent_dim=2, seed=5
    )
    probabilities = UniformPolicy().probabilities(
        n_contexts=env.n_users, n_actions=env.n_items
    )
    a = simulate_logged_bandit_data(env, probabilities, n_rounds=500, seed=8)
    b = simulate_logged_bandit_data(env, probabilities, n_rounds=500, seed=8)

    np.testing.assert_array_equal(a.context_indices, b.context_indices)
    np.testing.assert_array_equal(a.actions, b.actions)
    np.testing.assert_array_equal(a.rewards, b.rewards)
    np.testing.assert_array_equal(a.behavior_propensities, b.behavior_propensities)


def test_invalid_behavior_probabilities_fail() -> None:
    env = SyntheticBanditEnvironment.generate(
        n_users=2, n_items=2, latent_dim=2, seed=3
    )
    bad = np.array([[0.8, 0.4], [0.5, 0.5]], dtype=float)

    with pytest.raises(ValueError, match="sum to 1"):
        simulate_logged_bandit_data(env, bad, n_rounds=10)
