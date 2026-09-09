from pathlib import Path
import numpy as np

from policyreclab.datasets.coat_features import (
    load_coat_feature_matrix,
    load_coat_propensity_matrix,
)
from policyreclab.datasets.mnar_ratings import RatingTriples
from policyreclab.experiments.coat_feature_propensity import (
    cross_fitted_feature_propensity_matrix,
    run_coat_feature_propensity_study,
)


def _obs() -> RatingTriples:
    return RatingTriples(
        users=np.array([0, 0, 1, 2, 3], dtype=np.int64),
        items=np.array([0, 1, 1, 2, 3], dtype=np.int64),
        ratings=np.array([5., 4., 3., 2., 1.]),
    )


def _features():
    users = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    items = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    return users, items


def test_feature_loader_and_propensity_loader(tmp_path: Path):
    feature_path = tmp_path / "features.ascii"
    np.savetxt(feature_path, np.array([[1, 0], [0, 1]]), fmt="%d")
    features = load_coat_feature_matrix(feature_path, expected_rows=2)
    assert features.shape == (2, 2)

    propensity_path = tmp_path / "p.ascii"
    np.savetxt(propensity_path, np.full((2, 3), 0.2), fmt="%.3f")
    propensities = load_coat_propensity_matrix(
        propensity_path, n_users=2, n_items=3
    )
    assert propensities.shape == (2, 3)


def test_cross_fitted_feature_predictions_are_valid():
    users, items = _features()
    predictions = cross_fitted_feature_propensity_matrix(
        _obs(),
        users,
        items,
        n_users=4,
        n_items=4,
        n_folds=2,
        epochs=8,
        learning_rate=1.0,
        seed=11,
    )
    assert predictions.shape == (4, 4)
    assert np.all((predictions > 0.0) & (predictions < 1.0))


def test_feature_study_reports_optional_released_benchmark():
    users, items = _features()
    randomized = RatingTriples(
        users=np.array([0, 1, 2, 3], dtype=np.int64),
        items=np.array([0, 1, 2, 3], dtype=np.int64),
        ratings=np.array([2., 2., 2., 2.]),
    )
    released = np.full((4, 4), 0.25)

    result = run_coat_feature_propensity_study(
        observational=_obs(),
        randomized=randomized,
        user_features=users,
        item_features=items,
        released_propensity_matrix=released,
        n_users=4,
        n_items=4,
        n_folds=2,
        epochs=8,
        learning_rate=1.0,
        seed=5,
    )

    assert result.randomized_reference_mean == 2.0
    assert np.isfinite(result.feature_aware_mean)
    assert result.released_propensity_mean is not None
    assert result.feature_aware_weight_diagnostics.maximum_weight >= 1.0
    assert result.feature_aware_exposure_diagnostics.log_loss >= 0.0


def test_released_propensity_requires_positive_observed_entries():
    users, items = _features()
    randomized = RatingTriples(
        users=np.array([0], dtype=np.int64),
        items=np.array([0], dtype=np.int64),
        ratings=np.array([3.]),
    )
    released = np.full((4, 4), 0.25)
    released[0, 0] = 0.0

    try:
        run_coat_feature_propensity_study(
            observational=_obs(),
            randomized=randomized,
            user_features=users,
            item_features=items,
            released_propensity_matrix=released,
            n_users=4,
            n_items=4,
            n_folds=2,
            epochs=4,
            learning_rate=1.0,
        )
    except ValueError as exc:
        assert "propensities" in str(exc)
    else:
        raise AssertionError("expected zero observed propensity to fail")


def test_official_coat_layout_paths_are_documented():
    from pathlib import Path
    root = Path("data/coat")
    assert root / "user_item_features" / "user_features.ascii"
    assert root / "user_item_features" / "item_features.ascii"
    assert root / "propensities.ascii"
