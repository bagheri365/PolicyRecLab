import pytest

from policyreclab.experiments import run_direct_method_study


def test_oracle_dm_recovers_exact_finite_population_value() -> None:
    study = run_direct_method_study(
        behavior_epsilon=0.1,
        n_repetitions=12,
        n_users=100,
        n_items=10,
        latent_dim=4,
        n_rounds=3_000,
        environment_seed=41,
        base_log_seed=15_000,
    )

    assert study.oracle_dm.mean == pytest.approx(study.true_value, abs=1e-12)
    assert study.oracle_dm.bias == pytest.approx(0.0, abs=1e-12)
    assert study.oracle_dm.variance == pytest.approx(0.0, abs=1e-20)
    assert study.oracle_dm.rmse == pytest.approx(0.0, abs=1e-12)


def test_misspecified_dm_is_low_variance_but_biased() -> None:
    study = run_direct_method_study(
        behavior_epsilon=0.05,
        n_repetitions=40,
        n_users=140,
        n_items=10,
        latent_dim=4,
        n_rounds=5_000,
        environment_seed=43,
        base_log_seed=17_000,
    )

    # The global-mean model learns behavior-policy reward and then extrapolates
    # it to every target action. In this setup that is deliberately wrong.
    assert abs(study.misspecified_dm.bias) > 0.05

    # Its repeated-sample variance can nevertheless look attractively small.
    assert study.misspecified_dm.variance < study.ips.variance
