from __future__ import annotations

import argparse
from pathlib import Path

from policyreclab.datasets.mnar_ratings import load_coat_matrix
from policyreclab.experiments.coat_exposure_diagnostics import (
    run_coat_exposure_diagnostics,
)


def _print(prefix: str, diagnostics) -> None:
    print(f"{prefix}_brier_score: {diagnostics.brier_score}")
    print(f"{prefix}_log_loss: {diagnostics.log_loss}")
    print(
        f"{prefix}_observed_mean_predicted_propensity: "
        f"{diagnostics.observed_mean_predicted_propensity}"
    )
    print(
        f"{prefix}_unobserved_mean_predicted_propensity: "
        f"{diagnostics.unobserved_mean_predicted_propensity}"
    )
    print(f"{prefix}_separation_gap: {diagnostics.separation_gap}")
    print(f"{prefix}_p01_propensity: {diagnostics.p01_propensity}")
    print(f"{prefix}_p10_propensity: {diagnostics.p10_propensity}")
    print(f"{prefix}_p50_propensity: {diagnostics.p50_propensity}")
    print(f"{prefix}_p90_propensity: {diagnostics.p90_propensity}")
    print(f"{prefix}_p99_propensity: {diagnostics.p99_propensity}")
    print(
        f"{prefix}_user_mean_propensity_std: "
        f"{diagnostics.user_mean_propensity_std}"
    )
    print(
        f"{prefix}_item_mean_propensity_std: "
        f"{diagnostics.item_mean_propensity_std}"
    )
    print(
        f"{prefix}_calibration_error: "
        f"{diagnostics.calibration_error}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose Coat observational exposure models."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/coat"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    observational = load_coat_matrix(args.data_root / "train.ascii")
    result = run_coat_exposure_diagnostics(
        observational=observational,
        n_folds=args.folds,
        rank=args.rank,
        epochs=args.epochs,
        calibration_bins=args.calibration_bins,
        seed=args.seed,
    )

    _print("item_frequency", result.item_frequency)
    _print("matrix_factorization", result.matrix_factorization)


if __name__ == "__main__":
    main()
