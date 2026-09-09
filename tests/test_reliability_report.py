import math

from policyreclab.experiments import run_reliability_system_study


def test_healthy_report_is_identified_without_major_warnings() -> None:
    study = run_reliability_system_study(
        n_users=80, n_items=8, latent_dim=4, n_rounds=4_000,
        environment_seed=31, base_seed=40_000,
    )
    report = study.healthy_overlap

    assert report.identified_nonparametrically
    assert report.full_contextual_support
    assert math.isfinite(report.estimated_value)
    assert report.minimum_behavior_probability_on_target_support > 0.0


def test_weak_overlap_report_flags_weight_instability() -> None:
    study = run_reliability_system_study(
        n_users=80, n_items=8, latent_dim=4, n_rounds=5_000,
        environment_seed=37, base_seed=44_000,
    )
    report = study.weak_overlap

    assert report.full_contextual_support
    assert report.maximum_importance_weight > 20.0
    assert any(
        "weak overlap" in warning.lower()
        for warning in report.reliability_warnings
    )


def test_zero_support_report_refuses_to_present_ips_as_identified() -> None:
    study = run_reliability_system_study(
        n_users=60, n_items=6, latent_dim=3, n_rounds=3_000,
        environment_seed=41, base_seed=48_000,
    )
    report = study.zero_support

    assert not report.identified_nonparametrically
    assert not report.full_contextual_support
    assert math.isnan(report.estimated_value)
    assert any(
        "not nonparametrically identified" in warning.lower()
        for warning in report.reliability_warnings
    )


def test_selection_and_adaptive_logging_are_reported_separately() -> None:
    study = run_reliability_system_study(
        n_users=60, n_items=6, latent_dim=3, n_rounds=3_000,
        environment_seed=43, base_seed=52_000,
    )

    selection_report = study.selected_on_same_data
    adaptive_report = study.adaptive_logging

    assert any(
        "selection optimism" in warning.lower()
        for warning in selection_report.reliability_warnings
    )
    assert any(
        "adaptive" in warning.lower()
        for warning in adaptive_report.reliability_warnings
    )
    assert "ordinary iid" in adaptive_report.inference_validity_note.lower()
