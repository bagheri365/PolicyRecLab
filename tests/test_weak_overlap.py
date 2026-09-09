import pytest

from policyreclab.experiments import run_weak_overlap_study


def test_weaker_overlap_increases_weight_concentration() -> None:
    study = run_weak_overlap_study(
        epsilons=(1.0, 0.2, 0.05),
        n_repetitions=20,
        n_users=120,
        n_items=10,
        latent_dim=4,
        n_rounds=6_000,
        environment_seed=9,
        base_log_seed=1_000,
    )

    by_epsilon = {point.epsilon: point for point in study.points}
    strong = by_epsilon[1.0]
    medium = by_epsilon[0.2]
    weak = by_epsilon[0.05]

    # Support remains positive because epsilon > 0, but the smallest behavior
    # probability shrinks exactly with epsilon.
    assert strong.min_behavior_probability == pytest.approx(0.1)
    assert medium.min_behavior_probability == pytest.approx(0.02)
    assert weak.min_behavior_probability == pytest.approx(0.005)

    # Importance weights become more extreme and ESS concentration worsens.
    assert weak.mean_max_weight > medium.mean_max_weight > strong.mean_max_weight
    assert weak.mean_ess_fraction < medium.mean_ess_fraction < strong.mean_ess_fraction


def test_weak_overlap_has_worse_rmse_than_uniform_behavior() -> None:
    study = run_weak_overlap_study(
        epsilons=(1.0, 0.05),
        n_repetitions=30,
        n_users=140,
        n_items=10,
        latent_dim=4,
        n_rounds=5_000,
        environment_seed=17,
        base_log_seed=2_000,
    )

    strong, weak = study.points

    # Both settings identify the uniform target. The difference is estimation
    # quality: weak overlap should materially increase repeated-sample error.
    assert weak.rmse > strong.rmse * 1.5

    # IPS remains approximately centered across repetitions; weak overlap hurts
    # variance rather than introducing a systematic support-related bias.
    assert abs(strong.bias) < 0.02
    assert abs(weak.bias) < 0.03
