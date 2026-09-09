import numpy as np
import pytest

from policyreclab.diagnostics import analyze_support


def test_uniform_behavior_supports_deterministic_target() -> None:
    behavior = np.full((3, 4), 0.25)
    target = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    report = analyze_support(behavior, target)

    assert report.has_full_support
    assert report.n_violating_contexts == 0
    assert report.n_violating_pairs == 0
    assert report.target_mass_off_support == pytest.approx(0.0)
    assert (
        report.min_positive_behavior_probability_on_target_support
        == pytest.approx(0.25)
    )


def test_deterministic_behavior_does_not_support_uniform_target() -> None:
    behavior = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    target = np.full((2, 4), 0.25)

    report = analyze_support(behavior, target)

    assert not report.has_full_support
    assert report.n_violating_contexts == 2
    assert report.n_violating_pairs == 6
    assert report.violating_context_fraction == pytest.approx(1.0)
    assert report.target_mass_off_support == pytest.approx(0.75)
    assert report.max_context_target_mass_off_support == pytest.approx(0.75)


def test_small_positive_probability_is_support_not_zero_support() -> None:
    behavior = np.array([[0.999, 0.001]])
    target = np.array([[0.0, 1.0]])

    report = analyze_support(behavior, target)

    assert report.has_full_support
    assert (
        report.min_positive_behavior_probability_on_target_support
        == pytest.approx(0.001)
    )


def test_zero_tolerance_can_be_used_for_numerical_support_checks() -> None:
    behavior = np.array([[1.0 - 1e-14, 1e-14]])
    target = np.array([[0.0, 1.0]])

    exact = analyze_support(behavior, target)
    tolerant = analyze_support(behavior, target, zero_tolerance=1e-12)

    assert exact.has_full_support
    assert not tolerant.has_full_support


def test_policy_matrix_validation() -> None:
    behavior = np.array([[0.5, 0.5]])
    bad_target = np.array([[0.8, 0.3]])

    with pytest.raises(ValueError, match="sum to 1"):
        analyze_support(behavior, bad_target)
