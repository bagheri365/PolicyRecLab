from __future__ import annotations

import numpy as np
import pytest

from policyreclab.experiments.yahoo_r3_calibration_sensitivity import (
    YahooR3CalibrationRun,
    summarize_yahoo_r3_calibration_reliability,
)


def _run(fraction: float, seed: int, bias: float, reduction: float) -> YahooR3CalibrationRun:
    return YahooR3CalibrationRun(
        calibration_fraction=fraction,
        seed=seed,
        n_calibration=int(1000 * fraction),
        n_evaluation=1000 - int(1000 * fraction),
        randomized_calibration_mean=2.0 + bias,
        randomized_reference_mean=2.0,
        calibration_evaluation_gap=bias,
        naive_bayes_mean=2.0 + bias,
        naive_bayes_bias=bias,
        naive_bayes_bias_reduction=reduction,
        naive_bayes_max_weight=90.0,
        naive_bayes_p99_weight=90.0,
        naive_bayes_ess_fraction=0.66,
    )


def test_reliability_groups_by_calibration_budget():
    runs = [
        _run(0.01, 1, -0.03, 0.97),
        _run(0.01, 2, 0.01, 0.99),
        _run(0.05, 1, -0.005, 0.995),
        _run(0.05, 2, 0.004, 0.996),
    ]
    result = summarize_yahoo_r3_calibration_reliability(runs)
    assert [x.calibration_fraction for x in result] == [0.01, 0.05]
    assert [x.mean_n_calibration for x in result] == [10.0, 50.0]


def test_reliability_success_probabilities_are_empirical_frequencies():
    runs = [
        _run(0.05, 1, -0.025, 0.97),
        _run(0.05, 2, -0.015, 0.985),
        _run(0.05, 3, 0.005, 0.995),
        _run(0.05, 4, 0.008, 0.999),
    ]
    result = summarize_yahoo_r3_calibration_reliability(runs)[0]
    assert np.isclose(result.probability_abs_bias_le_0_01, 0.5)
    assert np.isclose(result.probability_abs_bias_le_0_02, 0.75)
    assert np.isclose(result.probability_bias_reduction_ge_0_98, 0.75)


def test_reliability_quantiles_match_numpy():
    biases = np.asarray([-0.04, -0.02, 0.0, 0.01, 0.03])
    runs = [_run(0.10, i, float(b), 0.99) for i, b in enumerate(biases)]
    result = summarize_yahoo_r3_calibration_reliability(runs)[0]
    assert np.isclose(result.q05_naive_bayes_bias, np.quantile(biases, 0.05))
    assert np.isclose(result.median_naive_bayes_bias, np.quantile(biases, 0.50))
    assert np.isclose(result.q95_naive_bayes_bias, np.quantile(biases, 0.95))
    assert np.isclose(result.q95_abs_naive_bayes_bias, np.quantile(np.abs(biases), 0.95))


def test_empty_runs_are_rejected():
    with pytest.raises(ValueError, match="at least one calibration run"):
        summarize_yahoo_r3_calibration_reliability([])
