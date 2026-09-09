from policyreclab.experiments import run_doubly_robust_study


def test_dr_repairs_deliberately_misspecified_dm_with_correct_propensities() -> None:
    study = run_doubly_robust_study(
        behavior_epsilon=0.1,
        n_repetitions=50,
        n_users=150,
        n_items=10,
        latent_dim=4,
        n_rounds=6_000,
        environment_seed=47,
        base_log_seed=19_000,
    )

    # The deliberately misspecified global-mean DM is systematically wrong.
    assert abs(study.misspecified_dm.bias) > 0.04

    # With known correct behavior propensities and support, the DR correction
    # removes most of that systematic model bias across repeated samples.
    assert abs(study.misspecified_dr.bias) < 0.025
    assert abs(study.misspecified_dr.bias) < abs(study.misspecified_dm.bias) * 0.4


def test_oracle_reward_model_reduces_dr_variance_relative_to_ips() -> None:
    study = run_doubly_robust_study(
        behavior_epsilon=0.1,
        n_repetitions=50,
        n_users=150,
        n_items=10,
        latent_dim=4,
        n_rounds=6_000,
        environment_seed=53,
        base_log_seed=23_000,
    )

    # Exact conditional reward means make the residual correction substantially
    # less noisy than weighting raw rewards alone in this controlled setting.
    assert study.oracle_dr.variance < study.ips.variance
    assert study.oracle_dr.rmse < study.ips.rmse
