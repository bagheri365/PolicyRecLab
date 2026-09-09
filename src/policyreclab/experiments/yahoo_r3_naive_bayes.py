from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from policyreclab.datasets.mnar_ratings import RatingTriples
from policyreclab.experiments.coat_exposure_model import ExposureWeightDiagnostics
from policyreclab.experiments.mnar_debiasing import (
    inverse_item_frequency_propensities,
)


@dataclass(frozen=True)
class YahooR3NaiveBayesResult:
    n_observational: int
    n_randomized_total: int
    n_randomized_calibration: int
    n_randomized_evaluation: int
    randomized_reference_mean: float
    randomized_calibration_mean: float
    calibration_evaluation_gap: float
    naive_observational_mean: float
    naive_bias: float
    item_frequency_mean: float
    item_frequency_bias: float
    item_frequency_bias_reduction: float
    naive_bayes_mean: float
    naive_bayes_bias: float
    naive_bayes_bias_reduction: float
    observation_rate: float
    observational_rating_distribution: np.ndarray
    calibration_rating_distribution: np.ndarray
    rating_propensities: np.ndarray
    item_frequency_diagnostics: ExposureWeightDiagnostics
    naive_bayes_diagnostics: ExposureWeightDiagnostics
    calibration_fraction: float
    seed: int


def split_randomized_calibration(
    randomized: RatingTriples,
    *,
    calibration_fraction: float = 0.05,
    seed: int = 2026,
) -> tuple[RatingTriples, RatingTriples]:
    """Create a deterministic calibration/evaluation split of randomized data."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie in (0, 1)")
    n = randomized.n_ratings
    n_calibration = max(1, int(round(calibration_fraction * n)))
    if n_calibration >= n:
        raise ValueError("calibration split must leave evaluation ratings")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    calibration_idx = order[:n_calibration]
    evaluation_idx = order[n_calibration:]

    def take(idx: np.ndarray) -> RatingTriples:
        return RatingTriples(
            users=randomized.users[idx],
            items=randomized.items[idx],
            ratings=randomized.ratings[idx],
        )

    return take(calibration_idx), take(evaluation_idx)


def _rating_distribution(
    ratings: np.ndarray,
    *,
    laplace: float,
) -> np.ndarray:
    if laplace < 0.0:
        raise ValueError("laplace must be nonnegative")
    integer = np.asarray(ratings, dtype=np.int64)
    if np.any((integer < 1) | (integer > 5)):
        raise ValueError("ratings must lie in [1, 5]")
    counts = np.bincount(integer, minlength=6)[1:6].astype(np.float64)
    counts += laplace
    return counts / np.sum(counts)


def estimate_naive_bayes_rating_propensities(
    observational: RatingTriples,
    randomized_calibration: RatingTriples,
    *,
    n_users: int = 15400,
    n_items: int = 1000,
    laplace: float = 1.0,
    min_propensity: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Estimate P(O=1 | Y=r) using Schnabel et al.'s Naive-Bayes formula."""
    if n_users < 1 or n_items < 1:
        raise ValueError("n_users and n_items must be positive")
    if not 0.0 < min_propensity < 1.0:
        raise ValueError("min_propensity must lie in (0, 1)")

    observed_dist = _rating_distribution(
        observational.ratings,
        laplace=0.0,
    )
    marginal_dist = _rating_distribution(
        randomized_calibration.ratings,
        laplace=laplace,
    )
    observation_rate = observational.n_ratings / float(n_users * n_items)

    propensity = observed_dist * observation_rate / marginal_dist
    propensity = np.clip(propensity, min_propensity, 1.0)

    return propensity, observed_dist, marginal_dist, float(observation_rate)


def _weighted_mean(
    ratings: np.ndarray,
    propensities: np.ndarray,
) -> tuple[float, ExposureWeightDiagnostics]:
    weights = 1.0 / propensities
    estimate = float(np.sum(weights * ratings) / np.sum(weights))
    ess = float(np.sum(weights) ** 2 / np.sum(weights ** 2))
    return estimate, ExposureWeightDiagnostics(
        maximum_weight=float(np.max(weights)),
        p99_weight=float(np.quantile(weights, 0.99)),
        effective_sample_size=ess,
        effective_sample_fraction=float(ess / ratings.size),
    )


def run_yahoo_r3_naive_bayes_study(
    *,
    observational: RatingTriples,
    randomized: RatingTriples,
    n_users: int = 15400,
    n_items: int = 1000,
    calibration_fraction: float = 0.05,
    laplace: float = 1.0,
    min_propensity: float = 1e-6,
    seed: int = 2026,
) -> YahooR3NaiveBayesResult:
    """Run a Yahoo R3 MNAR mean-rating debiasing benchmark.

    The randomized calibration subset is used only to estimate P(Y=r). The
    randomized evaluation subset remains untouched until final comparison.
    """
    calibration, evaluation = split_randomized_calibration(
        randomized,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )

    reference = float(np.mean(evaluation.ratings))
    calibration_mean = float(np.mean(calibration.ratings))
    naive = float(np.mean(observational.ratings))
    naive_bias = naive - reference

    def reduction(bias: float) -> float:
        if naive_bias == 0.0:
            return float("nan")
        return float(1.0 - abs(bias) / abs(naive_bias))

    item_by_item = inverse_item_frequency_propensities(
        observational,
        n_users=n_users,
        n_items=n_items,
    )
    item_p = item_by_item[observational.items]
    item_mean, item_diag = _weighted_mean(observational.ratings, item_p)
    item_bias = item_mean - reference

    (
        rating_propensity,
        observational_dist,
        calibration_dist,
        observation_rate,
    ) = estimate_naive_bayes_rating_propensities(
        observational,
        calibration,
        n_users=n_users,
        n_items=n_items,
        laplace=laplace,
        min_propensity=min_propensity,
    )

    nb_p = rating_propensity[
        np.asarray(observational.ratings, dtype=np.int64) - 1
    ]
    nb_mean, nb_diag = _weighted_mean(observational.ratings, nb_p)
    nb_bias = nb_mean - reference

    return YahooR3NaiveBayesResult(
        n_observational=observational.n_ratings,
        n_randomized_total=randomized.n_ratings,
        n_randomized_calibration=calibration.n_ratings,
        n_randomized_evaluation=evaluation.n_ratings,
        randomized_reference_mean=reference,
        randomized_calibration_mean=calibration_mean,
        calibration_evaluation_gap=calibration_mean - reference,
        naive_observational_mean=naive,
        naive_bias=naive_bias,
        item_frequency_mean=item_mean,
        item_frequency_bias=item_bias,
        item_frequency_bias_reduction=reduction(item_bias),
        naive_bayes_mean=nb_mean,
        naive_bayes_bias=nb_bias,
        naive_bayes_bias_reduction=reduction(nb_bias),
        observation_rate=observation_rate,
        observational_rating_distribution=observational_dist,
        calibration_rating_distribution=calibration_dist,
        rating_propensities=rating_propensity,
        item_frequency_diagnostics=item_diag,
        naive_bayes_diagnostics=nb_diag,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
