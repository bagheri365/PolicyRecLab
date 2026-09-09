from __future__ import annotations

import argparse
from pathlib import Path

from policyreclab.datasets.coat_features import (
    load_coat_feature_matrix,
    load_coat_propensity_matrix,
)
from policyreclab.datasets.mnar_ratings import load_coat_matrix
from policyreclab.experiments.coat_feature_propensity import (
    run_coat_feature_propensity_study,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/coat"))
    parser.add_argument(
        "--released-propensities",
        type=Path,
        default=None,
        help="Optional 290x300 released/authorized Coat propensity matrix.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=4.0)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--min-propensity", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    observational = load_coat_matrix(args.data_root / "train.ascii")
    randomized = load_coat_matrix(args.data_root / "test.ascii")
    feature_root = args.data_root / "user_item_features"
    user_features = load_coat_feature_matrix(
        feature_root / "user_features.ascii", expected_rows=290
    )
    item_features = load_coat_feature_matrix(
        feature_root / "item_features.ascii", expected_rows=300
    )

    released_path = args.released_propensities
    if released_path is None:
        candidate = args.data_root / "propensities.ascii"
        if candidate.exists():
            released_path = candidate

    released = None
    if released_path is not None:
        released = load_coat_propensity_matrix(
            released_path,
            n_users=290,
            n_items=300,
        )

    result = run_coat_feature_propensity_study(
        observational=observational,
        randomized=randomized,
        user_features=user_features,
        item_features=item_features,
        released_propensity_matrix=released,
        n_folds=args.folds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        min_propensity=args.min_propensity,
        seed=args.seed,
    )

    print(f"n_user_features: {result.n_user_features}")
    print(f"n_item_features: {result.n_item_features}")
    print(f"randomized_reference_mean: {result.randomized_reference_mean}")
    print(f"naive_observational_mean: {result.naive_observational_mean}")
    print(f"naive_bias: {result.naive_bias}")
    print(f"item_frequency_mean: {result.item_frequency_mean}")
    print(f"item_frequency_bias: {result.item_frequency_bias}")
    print(f"item_frequency_bias_reduction: {result.item_frequency_bias_reduction:.4%}")
    print(f"feature_aware_mean: {result.feature_aware_mean}")
    print(f"feature_aware_bias: {result.feature_aware_bias}")
    print(f"feature_aware_bias_reduction: {result.feature_aware_bias_reduction:.4%}")
    print(
        "feature_aware_max_weight: "
        f"{result.feature_aware_weight_diagnostics.maximum_weight:.6f}"
    )
    print(
        "feature_aware_p99_weight: "
        f"{result.feature_aware_weight_diagnostics.p99_weight:.6f}"
    )
    print(
        "feature_aware_ess_fraction: "
        f"{result.feature_aware_weight_diagnostics.effective_sample_fraction:.4%}"
    )
    exposure = result.feature_aware_exposure_diagnostics
    print(f"feature_aware_brier_score: {exposure.brier_score}")
    print(f"feature_aware_log_loss: {exposure.log_loss}")
    print(
        "feature_aware_observed_mean_predicted_propensity: "
        f"{exposure.observed_mean_predicted_propensity}"
    )
    print(
        "feature_aware_unobserved_mean_predicted_propensity: "
        f"{exposure.unobserved_mean_predicted_propensity}"
    )
    print(f"feature_aware_separation_gap: {exposure.separation_gap}")
    print(f"feature_aware_p01_propensity: {exposure.p01_propensity}")
    print(f"feature_aware_p50_propensity: {exposure.p50_propensity}")
    print(f"feature_aware_p99_propensity: {exposure.p99_propensity}")

    if result.released_propensity_mean is None:
        print("released_propensity_benchmark: not supplied")
    else:
        print(f"released_propensity_mean: {result.released_propensity_mean}")
        print(f"released_propensity_bias: {result.released_propensity_bias}")
        print(
            "released_propensity_bias_reduction: "
            f"{result.released_propensity_bias_reduction:.4%}"
        )
        assert result.released_propensity_diagnostics is not None
        print(
            "released_propensity_max_weight: "
            f"{result.released_propensity_diagnostics.maximum_weight:.6f}"
        )
        print(
            "released_propensity_ess_fraction: "
            f"{result.released_propensity_diagnostics.effective_sample_fraction:.4%}"
        )


if __name__ == "__main__":
    main()
