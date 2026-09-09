import numpy as np

from policyreclab.datasets.mnar_ratings import RatingTriples
from policyreclab.experiments.coat_exposure_model import (
    cross_fitted_exposure_propensities,
    run_coat_exposure_comparison,
)


def _toy_observational() -> RatingTriples:
    return RatingTriples(
        users=np.array([0, 0, 1, 2, 2, 3], dtype=np.int64),
        items=np.array([0, 1, 1, 2, 3, 3], dtype=np.int64),
        ratings=np.array([5., 4., 3., 2., 1., 4.]),
    )


def test_cross_fitted_propensities_are_valid_and_reproducible():
    obs = _toy_observational()
    p1 = cross_fitted_exposure_propensities(
        obs,
        n_users=4,
        n_items=4,
        n_folds=2,
        rank=2,
        epochs=8,
        learning_rate=2.0,
        seed=7,
    )
    p2 = cross_fitted_exposure_propensities(
        obs,
        n_users=4,
        n_items=4,
        n_folds=2,
        rank=2,
        epochs=8,
        learning_rate=2.0,
        seed=7,
    )

    assert p1.shape == (obs.n_ratings,)
    assert np.all((p1 >= 0.01) & (p1 <= 1.0))
    assert np.allclose(p1, p2)


def test_cross_fitting_rejects_invalid_configuration():
    obs = _toy_observational()
    try:
        cross_fitted_exposure_propensities(
            obs,
            n_users=4,
            n_items=4,
            n_folds=1,
        )
    except ValueError as exc:
        assert "fold" in str(exc)
    else:
        raise AssertionError("expected invalid fold configuration")


def test_comparison_reports_both_models_and_weight_diagnostics():
    obs = _toy_observational()
    rnd = RatingTriples(
        users=np.array([0, 1, 2, 3], dtype=np.int64),
        items=np.array([0, 1, 2, 3], dtype=np.int64),
        ratings=np.array([2., 2., 2., 2.]),
    )

    result = run_coat_exposure_comparison(
        observational=obs,
        randomized=rnd,
        n_users=4,
        n_items=4,
        n_folds=2,
        rank=2,
        epochs=8,
        learning_rate=2.0,
        seed=9,
    )

    assert result.randomized_reference_mean == 2.0
    assert np.isfinite(result.item_frequency_mean)
    assert np.isfinite(result.matrix_factorization_mean)
    assert result.item_frequency_diagnostics.maximum_weight >= 1.0
    assert result.matrix_factorization_diagnostics.maximum_weight >= 1.0
    assert 0.0 < result.matrix_factorization_diagnostics.effective_sample_fraction <= 1.0
