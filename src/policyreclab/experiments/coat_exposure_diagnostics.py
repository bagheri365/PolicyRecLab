from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from policyreclab.datasets.mnar_ratings import RatingTriples
from policyreclab.experiments.coat_exposure_model import (
    _fit_logistic_matrix_factorization,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ExposureModelDiagnostics:
    model_name: str
    brier_score: float
    log_loss: float
    observed_mean_predicted_propensity: float
    unobserved_mean_predicted_propensity: float
    separation_gap: float
    p01_propensity: float
    p10_propensity: float
    p50_propensity: float
    p90_propensity: float
    p99_propensity: float
    user_mean_propensity_std: float
    item_mean_propensity_std: float
    calibration_error: float


@dataclass(frozen=True)
class CoatExposureDiagnosticsResult:
    item_frequency: ExposureModelDiagnostics
    matrix_factorization: ExposureModelDiagnostics
    n_users: int
    n_items: int
    n_folds: int
    rank: int


def _safe_log_loss(y: FloatArray, p: FloatArray) -> float:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _calibration_error(
    y: FloatArray,
    p: FloatArray,
    *,
    n_bins: int = 10,
) -> float:
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = y.size
    error = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        weight = float(np.sum(mask) / total)
        error += weight * abs(float(np.mean(y[mask]) - np.mean(p[mask])))
    return float(error)


def _summarize(
    *,
    model_name: str,
    labels: FloatArray,
    predictions: FloatArray,
    n_users: int,
    n_items: int,
    calibration_bins: int,
) -> ExposureModelDiagnostics:
    matrix = predictions.reshape(n_users, n_items)
    observed = labels == 1.0
    unobserved = ~observed

    observed_mean = float(np.mean(predictions[observed]))
    unobserved_mean = float(np.mean(predictions[unobserved]))

    return ExposureModelDiagnostics(
        model_name=model_name,
        brier_score=float(np.mean((predictions - labels) ** 2)),
        log_loss=_safe_log_loss(labels, predictions),
        observed_mean_predicted_propensity=observed_mean,
        unobserved_mean_predicted_propensity=unobserved_mean,
        separation_gap=observed_mean - unobserved_mean,
        p01_propensity=float(np.quantile(predictions, 0.01)),
        p10_propensity=float(np.quantile(predictions, 0.10)),
        p50_propensity=float(np.quantile(predictions, 0.50)),
        p90_propensity=float(np.quantile(predictions, 0.90)),
        p99_propensity=float(np.quantile(predictions, 0.99)),
        user_mean_propensity_std=float(np.std(np.mean(matrix, axis=1))),
        item_mean_propensity_std=float(np.std(np.mean(matrix, axis=0))),
        calibration_error=_calibration_error(
            labels,
            predictions,
            n_bins=calibration_bins,
        ),
    )


def _cross_fitted_predictions(
    observation_matrix: FloatArray,
    *,
    n_folds: int,
    rank: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> FloatArray:
    n_users, n_items = observation_matrix.shape
    rng = np.random.default_rng(seed)
    fold_ids = rng.integers(
        0,
        n_folds,
        size=(n_users, n_items),
        dtype=np.int64,
    )
    predictions = np.empty_like(observation_matrix)

    for fold in range(n_folds):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        fitted = _fit_logistic_matrix_factorization(
            observation_matrix,
            train_mask,
            rank=rank,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed + fold + 1,
        )
        predictions[test_mask] = fitted[test_mask]
    return predictions


def run_coat_exposure_diagnostics(
    *,
    observational: RatingTriples,
    n_users: int = 290,
    n_items: int = 300,
    n_folds: int = 5,
    rank: int = 4,
    epochs: int = 80,
    learning_rate: float = 8.0,
    l2: float = 1e-3,
    calibration_bins: int = 10,
    seed: int = 2026,
) -> CoatExposureDiagnosticsResult:
    """Diagnose exposure prediction without using randomized ratings."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    observed = np.zeros((n_users, n_items), dtype=np.float64)
    observed[observational.users, observational.items] = 1.0
    labels = observed.reshape(-1)

    # Item-frequency baseline: estimate item exposure probability from
    # observational counts. This is intentionally simple and not cross-fitted.
    item_rates = np.mean(observed, axis=0)
    item_predictions = np.broadcast_to(
        item_rates[None, :],
        observed.shape,
    ).reshape(-1)

    mf_predictions = _cross_fitted_predictions(
        observed,
        n_folds=n_folds,
        rank=rank,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
    ).reshape(-1)

    return CoatExposureDiagnosticsResult(
        item_frequency=_summarize(
            model_name="item_frequency",
            labels=labels,
            predictions=item_predictions,
            n_users=n_users,
            n_items=n_items,
            calibration_bins=calibration_bins,
        ),
        matrix_factorization=_summarize(
            model_name="cross_fitted_logistic_matrix_factorization",
            labels=labels,
            predictions=mf_predictions,
            n_users=n_users,
            n_items=n_items,
            calibration_bins=calibration_bins,
        ),
        n_users=n_users,
        n_items=n_items,
        n_folds=n_folds,
        rank=rank,
    )
