from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np

from policyreclab.datasets.mnar_ratings import RatingTriples
from policyreclab.experiments.yahoo_r3_naive_bayes import (
    run_yahoo_r3_naive_bayes_study,
)


@dataclass(frozen=True)
class YahooR3CalibrationRun:
    calibration_fraction: float
    seed: int
    n_calibration: int
    n_evaluation: int
    randomized_calibration_mean: float
    randomized_reference_mean: float
    calibration_evaluation_gap: float
    naive_bayes_mean: float
    naive_bayes_bias: float
    naive_bayes_bias_reduction: float
    naive_bayes_max_weight: float
    naive_bayes_p99_weight: float
    naive_bayes_ess_fraction: float


@dataclass(frozen=True)
class YahooR3CalibrationSummary:
    calibration_fraction: float
    n_runs: int
    mean_n_calibration: float
    mean_abs_calibration_evaluation_gap: float
    std_calibration_evaluation_gap: float
    mean_abs_naive_bayes_bias: float
    std_naive_bayes_bias: float
    mean_bias_reduction: float
    min_bias_reduction: float
    max_bias_reduction: float
    mean_max_weight: float
    mean_p99_weight: float
    mean_ess_fraction: float


@dataclass(frozen=True)
class YahooR3CalibrationReliability:
    calibration_fraction: float
    mean_n_calibration: float
    q05_naive_bayes_bias: float
    median_naive_bayes_bias: float
    q95_naive_bayes_bias: float
    q90_abs_naive_bayes_bias: float
    q95_abs_naive_bayes_bias: float
    probability_abs_bias_le_0_01: float
    probability_abs_bias_le_0_02: float
    probability_bias_reduction_ge_0_98: float


def run_yahoo_r3_calibration_sensitivity(
    observational: RatingTriples,
    randomized: RatingTriples,
    *,
    calibration_fractions: Iterable[float] = (0.01, 0.025, 0.05, 0.10, 0.20),
    seeds: Iterable[int] = tuple(range(10)),
    n_users: int = 15400,
    n_items: int = 1000,
    laplace: float = 1.0,
    min_propensity: float = 1e-6,
) -> tuple[list[YahooR3CalibrationRun], list[YahooR3CalibrationSummary]]:
    fractions = tuple(float(x) for x in calibration_fractions)
    seed_values = tuple(int(x) for x in seeds)
    if not fractions:
        raise ValueError("at least one calibration fraction is required")
    if not seed_values:
        raise ValueError("at least one seed is required")

    runs: list[YahooR3CalibrationRun] = []
    for fraction in fractions:
        if not 0.0 < fraction < 1.0:
            raise ValueError("calibration fractions must lie strictly between 0 and 1")
        for seed in seed_values:
            result = run_yahoo_r3_naive_bayes_study(
                observational=observational,
                randomized=randomized,
                n_users=n_users,
                n_items=n_items,
                calibration_fraction=fraction,
                laplace=laplace,
                min_propensity=min_propensity,
                seed=seed,
            )
            diag = result.naive_bayes_diagnostics
            runs.append(
                YahooR3CalibrationRun(
                    calibration_fraction=fraction,
                    seed=seed,
                    n_calibration=result.n_randomized_calibration,
                    n_evaluation=result.n_randomized_evaluation,
                    randomized_calibration_mean=result.randomized_calibration_mean,
                    randomized_reference_mean=result.randomized_reference_mean,
                    calibration_evaluation_gap=result.calibration_evaluation_gap,
                    naive_bayes_mean=result.naive_bayes_mean,
                    naive_bayes_bias=result.naive_bayes_bias,
                    naive_bayes_bias_reduction=result.naive_bayes_bias_reduction,
                    naive_bayes_max_weight=diag.maximum_weight,
                    naive_bayes_p99_weight=diag.p99_weight,
                    naive_bayes_ess_fraction=diag.effective_sample_fraction,
                )
            )

    summaries: list[YahooR3CalibrationSummary] = []
    for fraction in sorted(set(fractions)):
        group = [run for run in runs if run.calibration_fraction == fraction]
        gaps = np.asarray([r.calibration_evaluation_gap for r in group], dtype=float)
        nb_bias = np.asarray([r.naive_bayes_bias for r in group], dtype=float)
        reductions = np.asarray([r.naive_bayes_bias_reduction for r in group], dtype=float)
        max_w = np.asarray([r.naive_bayes_max_weight for r in group], dtype=float)
        p99_w = np.asarray([r.naive_bayes_p99_weight for r in group], dtype=float)
        ess = np.asarray([r.naive_bayes_ess_fraction for r in group], dtype=float)
        n_cal = np.asarray([r.n_calibration for r in group], dtype=float)

        summaries.append(
            YahooR3CalibrationSummary(
                calibration_fraction=fraction,
                n_runs=len(group),
                mean_n_calibration=float(np.mean(n_cal)),
                mean_abs_calibration_evaluation_gap=float(np.mean(np.abs(gaps))),
                std_calibration_evaluation_gap=float(np.std(gaps, ddof=0)),
                mean_abs_naive_bayes_bias=float(np.mean(np.abs(nb_bias))),
                std_naive_bayes_bias=float(np.std(nb_bias, ddof=0)),
                mean_bias_reduction=float(np.mean(reductions)),
                min_bias_reduction=float(np.min(reductions)),
                max_bias_reduction=float(np.max(reductions)),
                mean_max_weight=float(np.mean(max_w)),
                mean_p99_weight=float(np.mean(p99_w)),
                mean_ess_fraction=float(np.mean(ess)),
            )
        )

    return runs, summaries


def summarize_yahoo_r3_calibration_reliability(
    runs: Iterable[YahooR3CalibrationRun],
) -> list[YahooR3CalibrationReliability]:
    """Summarize empirical split-to-split reliability at each calibration budget.

    Quantiles and probabilities are descriptive over the supplied randomized
    splits. They are not confidence intervals for a population parameter.
    """
    run_values = list(runs)
    if not run_values:
        raise ValueError("at least one calibration run is required")

    reliability: list[YahooR3CalibrationReliability] = []
    for fraction in sorted({run.calibration_fraction for run in run_values}):
        group = [run for run in run_values if run.calibration_fraction == fraction]
        bias = np.asarray([run.naive_bayes_bias for run in group], dtype=float)
        abs_bias = np.abs(bias)
        reductions = np.asarray(
            [run.naive_bayes_bias_reduction for run in group], dtype=float
        )
        n_cal = np.asarray([run.n_calibration for run in group], dtype=float)
        reliability.append(
            YahooR3CalibrationReliability(
                calibration_fraction=fraction,
                mean_n_calibration=float(np.mean(n_cal)),
                q05_naive_bayes_bias=float(np.quantile(bias, 0.05)),
                median_naive_bayes_bias=float(np.quantile(bias, 0.50)),
                q95_naive_bayes_bias=float(np.quantile(bias, 0.95)),
                q90_abs_naive_bayes_bias=float(np.quantile(abs_bias, 0.90)),
                q95_abs_naive_bayes_bias=float(np.quantile(abs_bias, 0.95)),
                probability_abs_bias_le_0_01=float(np.mean(abs_bias <= 0.01)),
                probability_abs_bias_le_0_02=float(np.mean(abs_bias <= 0.02)),
                probability_bias_reduction_ge_0_98=float(
                    np.mean(reductions >= 0.98)
                ),
            )
        )
    return reliability
