import pytest

from policyreclab.experiments import run_support_identification_experiment


def test_three_support_cases_have_expected_identification_status() -> None:
    result = run_support_identification_experiment(
        n_users=100,
        n_items=10,
        latent_dim=4,
        epsilon=0.1,
        seed=13,
    )

    # Uniform behavior has support everywhere, so it supports a greedy target.
    assert result.uniform_to_greedy.has_full_support

    # Epsilon-greedy with epsilon > 0 also has support everywhere.
    assert result.epsilon_greedy_to_uniform.has_full_support
    assert (
        result.epsilon_greedy_to_uniform
        .min_positive_behavior_probability_on_target_support
        == pytest.approx(0.01)
    )

    # Deterministic greedy behavior assigns zero probability to every
    # non-greedy action; a uniform target therefore puts 90% of its probability
    # mass off support in every context when there are 10 actions.
    assert not result.greedy_to_uniform.has_full_support
    assert result.greedy_to_uniform.violating_context_fraction == pytest.approx(1.0)
    assert result.greedy_to_uniform.target_mass_off_support == pytest.approx(0.9)
