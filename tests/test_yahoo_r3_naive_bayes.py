from pathlib import Path
import numpy as np

from policyreclab.datasets.mnar_ratings import RatingTriples
from policyreclab.datasets.yahoo_r3 import load_yahoo_r3_triples
from policyreclab.experiments.yahoo_r3_naive_bayes import (
    estimate_naive_bayes_rating_propensities,
    run_yahoo_r3_naive_bayes_study,
    split_randomized_calibration,
)


def _triples(ratings):
    ratings = np.asarray(ratings, dtype=float)
    n = ratings.size
    return RatingTriples(
        users=np.arange(n, dtype=np.int64) % 4,
        items=np.arange(n, dtype=np.int64) % 5,
        ratings=ratings,
    )


def test_yahoo_loader_normalizes_standard_one_based_ids(tmp_path: Path):
    path = tmp_path / "yahoo.txt"
    path.write_text("1 1 5\n15400 1000 1\n", encoding="utf-8")
    triples = load_yahoo_r3_triples(path)
    assert triples.users.tolist() == [0, 15399]
    assert triples.items.tolist() == [0, 999]
    assert triples.ratings.tolist() == [5.0, 1.0]


def test_randomized_split_is_deterministic_and_disjoint():
    randomized = _triples([1, 2, 3, 4, 5] * 20)
    a_cal, a_eval = split_randomized_calibration(
        randomized, calibration_fraction=0.05, seed=7
    )
    b_cal, b_eval = split_randomized_calibration(
        randomized, calibration_fraction=0.05, seed=7
    )
    assert a_cal.ratings.tolist() == b_cal.ratings.tolist()
    assert a_eval.ratings.tolist() == b_eval.ratings.tolist()
    assert a_cal.n_ratings == 5
    assert a_eval.n_ratings == 95


def test_naive_bayes_propensity_formula_is_finite():
    observational = _triples([5, 5, 4, 4, 4, 3, 2, 1])
    calibration = _triples([1, 2, 3, 4, 5] * 4)
    p, obs_dist, marginal_dist, obs_rate = (
        estimate_naive_bayes_rating_propensities(
            observational,
            calibration,
            n_users=4,
            n_items=5,
        )
    )
    assert p.shape == (5,)
    assert np.all((p > 0.0) & (p <= 1.0))
    assert np.isclose(np.sum(obs_dist), 1.0)
    assert np.isclose(np.sum(marginal_dist), 1.0)
    assert np.isclose(obs_rate, 8 / 20)


def test_yahoo_study_uses_evaluation_holdout_as_reference():
    observational = RatingTriples(
        users=np.array([0, 0, 1, 1, 2, 2, 3, 3]),
        items=np.array([0, 1, 0, 2, 1, 3, 2, 4]),
        ratings=np.array([5., 5., 4., 4., 3., 3., 2., 1.]),
    )
    randomized = _triples([1, 2, 3, 4, 5] * 20)
    result = run_yahoo_r3_naive_bayes_study(
        observational=observational,
        randomized=randomized,
        n_users=4,
        n_items=5,
        calibration_fraction=0.05,
        seed=13,
    )
    assert result.n_randomized_calibration == 5
    assert result.n_randomized_evaluation == 95
    assert np.isfinite(result.naive_bayes_mean)
    assert result.naive_bayes_diagnostics.maximum_weight >= 1.0
    assert result.rating_propensities.shape == (5,)


def test_yahoo_loader_accepts_zero_based_csv_and_header(tmp_path: Path):
    path = tmp_path / "mirror.txt"
    path.write_text(
        "uid,iid,rating\n0,48,1\n15399,999,5\n",
        encoding="utf-8",
    )
    triples = load_yahoo_r3_triples(path)
    assert triples.users.tolist() == [0, 15399]
    assert triples.items.tolist() == [48, 999]
    assert triples.ratings.tolist() == [1.0, 5.0]


def test_yahoo_loader_accepts_zero_based_csv_without_item_zero(tmp_path: Path):
    path = tmp_path / "random.txt"
    path.write_text("0,48,1\n0,125,1\n", encoding="utf-8")
    triples = load_yahoo_r3_triples(path)
    assert triples.users.tolist() == [0, 0]
    assert triples.items.tolist() == [48, 125]


def test_naive_bayes_weighted_mean_matches_calibration_distribution_mean():
    observational = RatingTriples(
        users=np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 2]),
        items=np.array([0, 1, 0, 2, 1, 3, 2, 4, 0, 4]),
        ratings=np.array([5., 5., 4., 4., 3., 3., 2., 1., 5., 1.]),
    )
    randomized = _triples([1, 2, 3, 4, 5] * 40)
    result = run_yahoo_r3_naive_bayes_study(
        observational=observational,
        randomized=randomized,
        n_users=4,
        n_items=5,
        calibration_fraction=0.10,
        laplace=0.0,
        seed=3,
    )
    assert np.isclose(
        result.naive_bayes_mean,
        result.randomized_calibration_mean,
        atol=1e-12,
    )
    assert np.isclose(
        result.calibration_evaluation_gap,
        result.randomized_calibration_mean - result.randomized_reference_mean,
    )
