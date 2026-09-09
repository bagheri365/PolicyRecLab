from __future__ import annotations

import argparse
from pathlib import Path

from policyreclab.datasets.yahoo_r3 import load_yahoo_r3_triples
from policyreclab.experiments.yahoo_r3_naive_bayes import (
    run_yahoo_r3_naive_bayes_study,
)


STANDARD_TRAIN = "ydata-ymusic-rating-study-v1_0-train.txt"
STANDARD_TEST = "ydata-ymusic-rating-study-v1_0-test.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Yahoo R3 Naive-Bayes MNAR propensity benchmark."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/yahoo_r3"),
    )
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--randomized", type=Path, default=None)
    parser.add_argument("--calibration-fraction", type=float, default=0.05)
    parser.add_argument("--laplace", type=float, default=1.0)
    parser.add_argument("--min-propensity", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.train is not None:
        train_path = args.train
    else:
        standard_train = args.data_root / STANDARD_TRAIN
        mirror_train = args.data_root / "user.txt"
        train_path = standard_train if standard_train.exists() else mirror_train

    if args.randomized is not None:
        randomized_path = args.randomized
    else:
        standard_test = args.data_root / STANDARD_TEST
        mirror_test = args.data_root / "random.txt"
        randomized_path = standard_test if standard_test.exists() else mirror_test

    observational = load_yahoo_r3_triples(train_path)
    randomized = load_yahoo_r3_triples(randomized_path)

    result = run_yahoo_r3_naive_bayes_study(
        observational=observational,
        randomized=randomized,
        calibration_fraction=args.calibration_fraction,
        laplace=args.laplace,
        min_propensity=args.min_propensity,
        seed=args.seed,
    )

    print(f"n_observational: {result.n_observational}")
    print(f"n_randomized_total: {result.n_randomized_total}")
    print(f"n_randomized_calibration: {result.n_randomized_calibration}")
    print(f"n_randomized_evaluation: {result.n_randomized_evaluation}")
    print(f"randomized_reference_mean: {result.randomized_reference_mean}")
    print(
        f"randomized_calibration_mean: {result.randomized_calibration_mean}"
    )
    print(
        "calibration_evaluation_gap: "
        f"{result.calibration_evaluation_gap}"
    )
    print(f"naive_observational_mean: {result.naive_observational_mean}")
    print(f"naive_bias: {result.naive_bias}")
    print(f"item_frequency_mean: {result.item_frequency_mean}")
    print(f"item_frequency_bias: {result.item_frequency_bias}")
    print(
        "item_frequency_bias_reduction: "
        f"{result.item_frequency_bias_reduction:.4%}"
    )
    print(f"naive_bayes_mean: {result.naive_bayes_mean}")
    print(f"naive_bayes_bias: {result.naive_bayes_bias}")
    print(
        "naive_bayes_bias_reduction: "
        f"{result.naive_bayes_bias_reduction:.4%}"
    )
    print(f"observation_rate: {result.observation_rate}")
    for rating, value in enumerate(
        result.observational_rating_distribution, start=1
    ):
        print(f"observational_P_Y{rating}: {value}")
    for rating, value in enumerate(
        result.calibration_rating_distribution, start=1
    ):
        print(f"calibration_P_Y{rating}: {value}")
    for rating, value in enumerate(result.rating_propensities, start=1):
        print(f"naive_bayes_propensity_rating_{rating}: {value}")
    print(
        "item_frequency_max_weight: "
        f"{result.item_frequency_diagnostics.maximum_weight:.6f}"
    )
    print(
        "item_frequency_ess_fraction: "
        f"{result.item_frequency_diagnostics.effective_sample_fraction:.4%}"
    )
    print(
        "naive_bayes_max_weight: "
        f"{result.naive_bayes_diagnostics.maximum_weight:.6f}"
    )
    print(
        "naive_bayes_p99_weight: "
        f"{result.naive_bayes_diagnostics.p99_weight:.6f}"
    )
    print(
        "naive_bayes_ess_fraction: "
        f"{result.naive_bayes_diagnostics.effective_sample_fraction:.4%}"
    )
    print(
        "naive_bayes_minus_calibration_mean: "
        f"{result.naive_bayes_mean - result.randomized_calibration_mean}"
    )
    print(
        "interpretation: Naive-Bayes inverse weighting reconstructs the rating "
        "distribution estimated from the randomized calibration subset, so its "
        "weighted mean is structurally tied to the calibration mean; the held-"
        "out randomized evaluation subset is the independent empirical "
        "reference; this is MNAR debiasing, not contextual-bandit OPE"
    )


if __name__ == "__main__":
    main()
