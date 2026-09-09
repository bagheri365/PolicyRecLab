from __future__ import annotations

import argparse
from pathlib import Path

from policyreclab.datasets.mnar_ratings import load_coat_matrix
from policyreclab.experiments.coat_exposure_model import (
    run_coat_exposure_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Coat item-frequency IPW with cross-fitted "
            "user-item exposure matrix factorization."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/coat"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--min-propensity", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    observational = load_coat_matrix(args.data_root / "train.ascii")
    randomized = load_coat_matrix(args.data_root / "test.ascii")

    result = run_coat_exposure_comparison(
        observational=observational,
        randomized=randomized,
        n_folds=args.folds,
        rank=args.rank,
        epochs=args.epochs,
        min_propensity=args.min_propensity,
        seed=args.seed,
    )

    print(f"randomized_reference_mean: {result.randomized_reference_mean}")
    print(f"naive_observational_mean: {result.naive_observational_mean}")
    print(f"naive_bias: {result.naive_bias}")
    print(f"item_frequency_mean: {result.item_frequency_mean}")
    print(f"item_frequency_bias: {result.item_frequency_bias}")
    print(
        "item_frequency_bias_reduction: "
        f"{result.item_frequency_bias_reduction:.4%}"
    )
    print(f"matrix_factorization_mean: {result.matrix_factorization_mean}")
    print(
        "matrix_factorization_bias: "
        f"{result.matrix_factorization_bias}"
    )
    print(
        "matrix_factorization_bias_reduction: "
        f"{result.matrix_factorization_bias_reduction:.4%}"
    )
    print(
        "item_frequency_max_weight: "
        f"{result.item_frequency_diagnostics.maximum_weight:.4f}"
    )
    print(
        "item_frequency_p99_weight: "
        f"{result.item_frequency_diagnostics.p99_weight:.4f}"
    )
    print(
        "item_frequency_ess_fraction: "
        f"{result.item_frequency_diagnostics.effective_sample_fraction:.4%}"
    )
    print(
        "matrix_factorization_max_weight: "
        f"{result.matrix_factorization_diagnostics.maximum_weight:.4f}"
    )
    print(
        "matrix_factorization_p99_weight: "
        f"{result.matrix_factorization_diagnostics.p99_weight:.4f}"
    )
    print(
        "matrix_factorization_ess_fraction: "
        f"{result.matrix_factorization_diagnostics.effective_sample_fraction:.4%}"
    )


if __name__ == "__main__":
    main()
