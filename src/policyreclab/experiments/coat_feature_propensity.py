from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from policyreclab.datasets.mnar_ratings import RatingTriples
from policyreclab.experiments.coat_exposure_model import ExposureWeightDiagnostics
from policyreclab.experiments.mnar_debiasing import (
    inverse_item_frequency_propensities,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class FeatureExposureDiagnostics:
    brier_score: float
    log_loss: float
    observed_mean_predicted_propensity: float
    unobserved_mean_predicted_propensity: float
    separation_gap: float
    p01_propensity: float
    p50_propensity: float
    p99_propensity: float


@dataclass(frozen=True)
class CoatFeaturePropensityResult:
    randomized_reference_mean: float
    naive_observational_mean: float
    naive_bias: float
    item_frequency_mean: float
    item_frequency_bias: float
    item_frequency_bias_reduction: float
    feature_aware_mean: float
    feature_aware_bias: float
    feature_aware_bias_reduction: float
    item_frequency_diagnostics: ExposureWeightDiagnostics
    feature_aware_weight_diagnostics: ExposureWeightDiagnostics
    feature_aware_exposure_diagnostics: FeatureExposureDiagnostics
    released_propensity_mean: float | None
    released_propensity_bias: float | None
    released_propensity_bias_reduction: float | None
    released_propensity_diagnostics: ExposureWeightDiagnostics | None
    n_user_features: int
    n_item_features: int
    n_folds: int
    min_propensity: float


def _sigmoid(values: FloatArray) -> FloatArray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def _fit_feature_logistic(
    observation_matrix: FloatArray,
    user_features: FloatArray,
    item_features: FloatArray,
    train_mask: BoolArray,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> FloatArray:
    n_users, n_items = observation_matrix.shape
    if user_features.shape[0] != n_users:
        raise ValueError("user feature rows must match the number of users")
    if item_features.shape[0] != n_items:
        raise ValueError("item feature rows must match the number of items")

    user_x = user_features - np.mean(user_features, axis=0, keepdims=True)
    item_x = item_features - np.mean(item_features, axis=0, keepdims=True)

    global_rate = float(np.mean(observation_matrix[train_mask]))
    global_rate = float(np.clip(global_rate, 1e-5, 1.0 - 1e-5))
    intercept = float(np.log(global_rate / (1.0 - global_rate)))

    user_coef = np.zeros(user_x.shape[1], dtype=np.float64)
    item_coef = np.zeros(item_x.shape[1], dtype=np.float64)
    interaction = np.zeros(
        (user_x.shape[1], item_x.shape[1]),
        dtype=np.float64,
    )

    mask = train_mask.astype(np.float64)
    denominator = max(float(np.sum(mask)), 1.0)

    for _ in range(epochs):
        logits = (
            intercept
            + (user_x @ user_coef)[:, None]
            + (item_x @ item_coef)[None, :]
            + user_x @ interaction @ item_x.T
        )
        probabilities = _sigmoid(logits)
        residual = (probabilities - observation_matrix) * mask

        grad_intercept = float(np.sum(residual) / denominator)
        grad_user = (
            user_x.T @ np.sum(residual, axis=1) / denominator
            + l2 * user_coef
        )
        grad_item = (
            item_x.T @ np.sum(residual, axis=0) / denominator
            + l2 * item_coef
        )
        grad_interaction = (
            user_x.T @ residual @ item_x / denominator
            + l2 * interaction
        )

        intercept -= learning_rate * grad_intercept
        user_coef -= learning_rate * grad_user
        item_coef -= learning_rate * grad_item
        interaction -= learning_rate * grad_interaction

    logits = (
        intercept
        + (user_x @ user_coef)[:, None]
        + (item_x @ item_coef)[None, :]
        + user_x @ interaction @ item_x.T
    )
    return _sigmoid(logits)


def cross_fitted_feature_propensity_matrix(
    observational: RatingTriples,
    user_features: FloatArray,
    item_features: FloatArray,
    *,
    n_users: int = 290,
    n_items: int = 300,
    n_folds: int = 5,
    epochs: int = 160,
    learning_rate: float = 4.0,
    l2: float = 1e-3,
    seed: int = 2026,
) -> FloatArray:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0.0:
        raise ValueError("l2 must be nonnegative")

    observed = np.zeros((n_users, n_items), dtype=np.float64)
    observed[observational.users, observational.items] = 1.0

    rng = np.random.default_rng(seed)
    fold_ids = rng.integers(
        0, n_folds, size=(n_users, n_items), dtype=np.int64
    )
    predictions = np.empty_like(observed)

    for fold in range(n_folds):
        held_out = fold_ids == fold
        fitted = _fit_feature_logistic(
            observed,
            user_features,
            item_features,
            ~held_out,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        predictions[held_out] = fitted[held_out]

    return predictions


def _weighted_mean(
    ratings: FloatArray,
    propensities: FloatArray,
) -> tuple[float, ExposureWeightDiagnostics]:
    if propensities.shape != ratings.shape:
        raise ValueError("propensities and ratings must have identical shape")
    if np.any((propensities <= 0.0) | (propensities > 1.0)):
        raise ValueError("propensities on observed ratings must lie in (0, 1]")
    weights = 1.0 / propensities
    estimate = float(np.sum(weights * ratings) / np.sum(weights))
    ess = float(np.sum(weights) ** 2 / np.sum(weights ** 2))
    return estimate, ExposureWeightDiagnostics(
        maximum_weight=float(np.max(weights)),
        p99_weight=float(np.quantile(weights, 0.99)),
        effective_sample_size=ess,
        effective_sample_fraction=float(ess / ratings.size),
    )


def _exposure_diagnostics(
    labels: FloatArray,
    predictions: FloatArray,
) -> FeatureExposureDiagnostics:
    flat_y = labels.reshape(-1)
    flat_p = predictions.reshape(-1)
    safe_p = np.clip(flat_p, 1e-12, 1.0 - 1e-12)
    observed = flat_y == 1.0
    unobserved = ~observed
    obs_mean = float(np.mean(flat_p[observed]))
    unobs_mean = float(np.mean(flat_p[unobserved]))
    return FeatureExposureDiagnostics(
        brier_score=float(np.mean((flat_p - flat_y) ** 2)),
        log_loss=float(
            -np.mean(
                flat_y * np.log(safe_p)
                + (1.0 - flat_y) * np.log(1.0 - safe_p)
            )
        ),
        observed_mean_predicted_propensity=obs_mean,
        unobserved_mean_predicted_propensity=unobs_mean,
        separation_gap=obs_mean - unobs_mean,
        p01_propensity=float(np.quantile(flat_p, 0.01)),
        p50_propensity=float(np.quantile(flat_p, 0.50)),
        p99_propensity=float(np.quantile(flat_p, 0.99)),
    )


def run_coat_feature_propensity_study(
    *,
    observational: RatingTriples,
    randomized: RatingTriples,
    user_features: FloatArray,
    item_features: FloatArray,
    released_propensity_matrix: FloatArray | None = None,
    n_users: int = 290,
    n_items: int = 300,
    n_folds: int = 5,
    epochs: int = 160,
    learning_rate: float = 4.0,
    l2: float = 1e-3,
    min_propensity: float = 0.01,
    seed: int = 2026,
) -> CoatFeaturePropensityResult:
    if not 0.0 < min_propensity < 1.0:
        raise ValueError("min_propensity must lie in (0, 1)")

    reference = float(np.mean(randomized.ratings))
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

    feature_matrix = cross_fitted_feature_propensity_matrix(
        observational,
        user_features,
        item_features,
        n_users=n_users,
        n_items=n_items,
        n_folds=n_folds,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
    )
    feature_matrix = np.clip(feature_matrix, min_propensity, 1.0)
    feature_p = feature_matrix[observational.users, observational.items]
    feature_mean, feature_weight_diag = _weighted_mean(
        observational.ratings,
        feature_p,
    )
    feature_bias = feature_mean - reference

    observed_matrix = np.zeros((n_users, n_items), dtype=np.float64)
    observed_matrix[observational.users, observational.items] = 1.0
    feature_exposure_diag = _exposure_diagnostics(
        observed_matrix,
        feature_matrix,
    )

    released_mean = None
    released_bias = None
    released_reduction = None
    released_diag = None
    if released_propensity_matrix is not None:
        if released_propensity_matrix.shape != (n_users, n_items):
            raise ValueError("released propensity matrix has the wrong shape")
        released_p = released_propensity_matrix[
            observational.users,
            observational.items,
        ]
        released_mean, released_diag = _weighted_mean(
            observational.ratings,
            released_p,
        )
        released_bias = released_mean - reference
        released_reduction = reduction(released_bias)

    return CoatFeaturePropensityResult(
        randomized_reference_mean=reference,
        naive_observational_mean=naive,
        naive_bias=naive_bias,
        item_frequency_mean=item_mean,
        item_frequency_bias=item_bias,
        item_frequency_bias_reduction=reduction(item_bias),
        feature_aware_mean=feature_mean,
        feature_aware_bias=feature_bias,
        feature_aware_bias_reduction=reduction(feature_bias),
        item_frequency_diagnostics=item_diag,
        feature_aware_weight_diagnostics=feature_weight_diag,
        feature_aware_exposure_diagnostics=feature_exposure_diag,
        released_propensity_mean=released_mean,
        released_propensity_bias=released_bias,
        released_propensity_bias_reduction=released_reduction,
        released_propensity_diagnostics=released_diag,
        n_user_features=int(user_features.shape[1]),
        n_item_features=int(item_features.shape[1]),
        n_folds=n_folds,
        min_propensity=min_propensity,
    )
