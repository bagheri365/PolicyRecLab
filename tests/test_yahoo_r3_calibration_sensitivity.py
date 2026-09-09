from __future__ import annotations

import numpy as np
import pytest

from policyreclab.datasets.mnar_ratings import RatingTriples
from policyreclab.experiments.yahoo_r3_calibration_sensitivity import (
    run_yahoo_r3_calibration_sensitivity,
)


def _triples(ratings: list[float]) -> RatingTriples:
    n = len(ratings)
    return RatingTriples(
        users=np.arange(n, dtype=np.int64) % 4,
        items=np.arange(n, dtype=np.int64) % 5,
        ratings=np.asarray(ratings, dtype=float),
    )


def test_sensitivity_returns_one_run_per_fraction_seed_pair():
    observational = _triples([1, 2, 3, 4, 5] * 8)
    randomized = _triples([1, 2, 3, 4, 5] * 20)

    runs, summaries = run_yahoo_r3_calibration_sensitivity(
        observational,
        randomized,
        calibration_fractions=(0.10, 0.20),
        seeds=(1, 2, 3),
        n_users=4,
        n_items=5,
        laplace=1.0,
    )

    assert len(runs) == 6
    assert len(summaries) == 2
    assert {s.calibration_fraction for s in summaries} == {0.10, 0.20}
    assert all(s.n_runs == 3 for s in summaries)


def test_sensitivity_summary_metrics_match_underlying_runs():
    observational = _triples([5, 5, 4, 4, 3, 3, 2, 1] * 5)
    randomized = _triples([1, 2, 3, 4, 5] * 20)

    runs, summaries = run_yahoo_r3_calibration_sensitivity(
        observational,
        randomized,
        calibration_fractions=(0.10,),
        seeds=(4, 5, 6),
        n_users=4,
        n_items=5,
        laplace=1.0,
    )

    summary = summaries[0]
    assert np.isclose(
        summary.mean_abs_naive_bayes_bias,
        np.mean([abs(r.naive_bayes_bias) for r in runs]),
    )
    assert np.isclose(
        summary.mean_ess_fraction,
        np.mean([r.naive_bayes_ess_fraction for r in runs]),
    )


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_invalid_fraction_is_rejected(fraction):
    observational = _triples([1, 2, 3, 4, 5] * 4)
    randomized = _triples([1, 2, 3, 4, 5] * 10)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        run_yahoo_r3_calibration_sensitivity(
            observational,
            randomized,
            calibration_fractions=(fraction,),
            seeds=(1,),
            n_users=4,
            n_items=5,
        )


def test_empty_seed_list_is_rejected():
    observational = _triples([1, 2, 3, 4, 5] * 4)
    randomized = _triples([1, 2, 3, 4, 5] * 10)
    with pytest.raises(ValueError, match="at least one seed"):
        run_yahoo_r3_calibration_sensitivity(
            observational,
            randomized,
            calibration_fractions=(0.1,),
            seeds=(),
            n_users=4,
            n_items=5,
        )
