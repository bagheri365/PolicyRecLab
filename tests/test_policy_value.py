import numpy as np
import pytest

from policyreclab.evaluation import (
    finite_population_policy_value,
    monte_carlo_policy_value,
)
from policyreclab.policies import UniformPolicy
from policyreclab.simulation import SyntheticBanditEnvironment


def test_uniform_policy_probabilities_sum_to_one() -> None:
    probs = UniformPolicy().probabilities(n_contexts=25, n_actions=7)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_uniform_policy_exact_value_matches_mean_reward_probability() -> None:
    env = SyntheticBanditEnvironment.generate(
        n_users=100,
        n_items=25,
        latent_dim=6,
        seed=2,
    )
    probs = UniformPolicy().probabilities(
        n_contexts=env.n_users,
        n_actions=env.n_items,
    )

    exact = finite_population_policy_value(env, probs)

    assert exact == pytest.approx(float(env.reward_probabilities.mean()))


def test_monte_carlo_converges_to_finite_population_value() -> None:
    env = SyntheticBanditEnvironment.generate(
        n_users=200,
        n_items=30,
        latent_dim=6,
        seed=11,
    )
    probs = UniformPolicy().probabilities(
        n_contexts=env.n_users,
        n_actions=env.n_items,
    )

    exact = finite_population_policy_value(env, probs)
    estimate = monte_carlo_policy_value(
        env,
        probs,
        n_rounds=120_000,
        seed=12,
    )

    # Bernoulli standard error is at most sqrt(.25 / 120000) ~= .00144.
    # A .006 tolerance is intentionally generous enough to avoid flaky tests
    # while still catching implementation errors.
    assert estimate == pytest.approx(exact, abs=0.006)


def test_invalid_policy_probabilities_fail() -> None:
    env = SyntheticBanditEnvironment.generate(
        n_users=2,
        n_items=2,
        latent_dim=2,
        seed=3,
    )

    bad = np.array([[0.7, 0.4], [0.5, 0.5]], dtype=float)

    with pytest.raises(ValueError, match="sum to 1"):
        finite_population_policy_value(env, bad)
