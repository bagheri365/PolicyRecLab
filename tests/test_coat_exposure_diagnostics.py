import numpy as np

from policyreclab.datasets.mnar_ratings import RatingTriples
from policyreclab.experiments.coat_exposure_diagnostics import (
    run_coat_exposure_diagnostics,
)


def _toy_observational() -> RatingTriples:
    return RatingTriples(
        users=np.array([0, 0, 1, 2, 2, 3], dtype=np.int64),
        items=np.array([0, 1, 1, 2, 3, 3], dtype=np.int64),
        ratings=np.array([5., 4., 3., 2., 1., 4.]),
    )


def test_diagnostics_report_prediction_quality_and_spread():
    result = run_coat_exposure_diagnostics(
        observational=_toy_observational(),
        n_users=4,
        n_items=4,
        n_folds=2,
        rank=2,
        epochs=8,
        learning_rate=2.0,
        calibration_bins=4,
        seed=13,
    )

    for diag in (result.item_frequency, result.matrix_factorization):
        assert 0.0 <= diag.brier_score <= 1.0
        assert diag.log_loss >= 0.0
        assert 0.0 <= diag.p01_propensity <= diag.p99_propensity <= 1.0
        assert np.isfinite(diag.separation_gap)
        assert diag.user_mean_propensity_std >= 0.0
        assert diag.item_mean_propensity_std >= 0.0
        assert diag.calibration_error >= 0.0


def test_item_frequency_has_no_user_level_variation():
    result = run_coat_exposure_diagnostics(
        observational=_toy_observational(),
        n_users=4,
        n_items=4,
        n_folds=2,
        rank=2,
        epochs=4,
        learning_rate=1.0,
        seed=3,
    )

    assert result.item_frequency.user_mean_propensity_std == 0.0
    assert result.item_frequency.item_mean_propensity_std > 0.0


def test_invalid_fold_count_is_rejected():
    try:
        run_coat_exposure_diagnostics(
            observational=_toy_observational(),
            n_users=4,
            n_items=4,
            n_folds=1,
        )
    except ValueError as exc:
        assert "fold" in str(exc)
    else:
        raise AssertionError("expected invalid fold count")
