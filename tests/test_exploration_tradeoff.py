import pytest

from policyreclab.experiments import run_exploration_tradeoff_study


def test_exploration_trades_immediate_reward_for_ope_quality() -> None:
    study = run_exploration_tradeoff_study(
        epsilons=(1.0, 0.2, 0.05),
        n_repetitions=30,
        n_users=140,
        n_items=10,
        latent_dim=4,
        n_rounds=5_000,
        environment_seed=31,
        base_log_seed=11_000,
    )
    by_epsilon = {point.epsilon: point for point in study.points}

    exploratory = by_epsilon[1.0]
    middle = by_epsilon[0.2]
    exploitative = by_epsilon[0.05]

    # In this oracle epsilon-greedy simulator, less exploration increases the
    # behavior policy's exact immediate reward.
    assert (
        exploitative.behavior_true_value
        > middle.behavior_true_value
        > exploratory.behavior_true_value
    )

    # But the uniform future target gets weaker overlap.
    assert (
        exploitative.min_behavior_probability_on_target_support
        < middle.min_behavior_probability_on_target_support
        < exploratory.min_behavior_probability_on_target_support
    )
    assert (
        exploitative.mean_ess_fraction
        < middle.mean_ess_fraction
        < exploratory.mean_ess_fraction
    )

    # The same change worsens repeated-sample IPS quality in this controlled
    # experiment. This is an empirical result, not a universal theorem.
    assert exploitative.ips_rmse > exploratory.ips_rmse * 1.5


def test_behavior_logged_reward_tracks_its_exact_policy_value() -> None:
    study = run_exploration_tradeoff_study(
        epsilons=(1.0, 0.1),
        n_repetitions=25,
        n_users=120,
        n_items=10,
        latent_dim=4,
        n_rounds=5_000,
        environment_seed=37,
        base_log_seed=13_000,
    )

    for point in study.points:
        assert point.mean_behavior_logged_reward == pytest.approx(
            point.behavior_true_value,
            abs=0.015,
        )
        assert abs(point.ips_bias) < 0.03
