from policyreclab.experiments import run_estimator_tradeoff_study


def test_clipping_exhibits_bias_variance_tradeoff_under_weak_overlap() -> None:
    study = run_estimator_tradeoff_study(
        epsilons=(0.05,),
        clip_threshold=5.0,
        n_repetitions=40,
        n_users=140,
        n_items=10,
        latent_dim=4,
        n_rounds=5_000,
        environment_seed=23,
        base_log_seed=4_000,
    )
    point = study.points[0]

    # Non-greedy target weights equal 1/epsilon = 20, so threshold 5 clips
    # some observations by construction.
    assert point.mean_clipping_fraction > 0.0

    # Clipping should reduce repeated-sample variance in this controlled setup.
    assert point.clipped_ips.variance < point.ips.variance

    # But it changes the estimand contribution and therefore introduces
    # systematic finite-sample bias relative to ordinary IPS.
    assert abs(point.clipped_ips.bias) > abs(point.ips.bias)


def test_snips_can_reduce_rmse_in_weak_overlap_setting() -> None:
    study = run_estimator_tradeoff_study(
        epsilons=(0.05,),
        clip_threshold=10.0,
        n_repetitions=50,
        n_users=150,
        n_items=10,
        latent_dim=4,
        n_rounds=5_000,
        environment_seed=29,
        base_log_seed=8_000,
    )
    point = study.points[0]

    # This is a controlled empirical demonstration, not a universal theorem.
    assert point.snips.rmse < point.ips.rmse
