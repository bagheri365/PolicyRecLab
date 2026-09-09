import pytest

from policyreclab.experiments import run_ips_validation_experiment


def test_ips_recovers_supported_target_value_in_large_synthetic_log() -> None:
    result = run_ips_validation_experiment(
        n_users=250,
        n_items=10,
        latent_dim=5,
        epsilon=0.4,
        n_rounds=80_000,
        environment_seed=41,
        log_seed=43,
    )

    assert result.ips_value == pytest.approx(
        result.target_true_value,
        abs=0.012,
    )
    assert result.mean_importance_weight == pytest.approx(1.0, abs=0.02)

    # The target is greedier than the behavior policy in this oracle synthetic
    # setup, so its value should exceed the behavior's on-policy logged mean.
    assert result.target_true_value > result.behavior_logged_mean
