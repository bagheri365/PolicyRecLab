from __future__ import annotations

import argparse
from pathlib import Path

from policyreclab.datasets.yahoo_r3 import load_yahoo_r3_triples
from policyreclab.experiments.yahoo_r3_calibration_sensitivity import (
    run_yahoo_r3_calibration_sensitivity,
    summarize_yahoo_r3_calibration_reliability,
)

STANDARD_TRAIN = "ydata-ymusic-rating-study-v1_0-train.txt"
STANDARD_TEST = "ydata-ymusic-rating-study-v1_0-test.txt"


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/yahoo_r3"))
    parser.add_argument("--train", type=Path)
    parser.add_argument("--randomized", type=Path)
    parser.add_argument(
        "--fractions",
        type=_parse_float_list,
        default=(0.01, 0.025, 0.05, 0.10, 0.20),
    )
    parser.add_argument(
        "--seeds",
        type=_parse_int_list,
        default=tuple(range(50)),
        help="Comma-separated split seeds; default uses 50 seeds for stable quantiles.",
    )
    parser.add_argument("--laplace", type=float, default=1.0)
    parser.add_argument("--min-propensity", type=float, default=1e-6)
    args = parser.parse_args()

    if args.train is not None:
        train_path = args.train
    else:
        standard_train = args.data_root / STANDARD_TRAIN
        train_path = (
            standard_train if standard_train.exists() else args.data_root / "user.txt"
        )

    if args.randomized is not None:
        randomized_path = args.randomized
    else:
        standard_test = args.data_root / STANDARD_TEST
        randomized_path = (
            standard_test if standard_test.exists() else args.data_root / "random.txt"
        )

    observational = load_yahoo_r3_triples(train_path)
    randomized = load_yahoo_r3_triples(randomized_path)

    runs, _ = run_yahoo_r3_calibration_sensitivity(
        observational,
        randomized,
        calibration_fractions=args.fractions,
        seeds=args.seeds,
        laplace=args.laplace,
        min_propensity=args.min_propensity,
    )
    reliability = summarize_yahoo_r3_calibration_reliability(runs)

    print(f"n_observational: {len(observational.ratings)}")
    print(f"n_randomized_total: {len(randomized.ratings)}")
    print(f"n_runs: {len(runs)}")
    print("reliability:")
    for r in reliability:
        print(
            "fraction={:.4f} mean_n_cal={:.1f} "
            "bias_q05={:.6f} bias_median={:.6f} bias_q95={:.6f} "
            "abs_bias_q90={:.6f} abs_bias_q95={:.6f} "
            "p_abs_bias_le_0.01={:.2%} p_abs_bias_le_0.02={:.2%} "
            "p_bias_reduction_ge_98pct={:.2%}".format(
                r.calibration_fraction,
                r.mean_n_calibration,
                r.q05_naive_bayes_bias,
                r.median_naive_bayes_bias,
                r.q95_naive_bayes_bias,
                r.q90_abs_naive_bayes_bias,
                r.q95_abs_naive_bayes_bias,
                r.probability_abs_bias_le_0_01,
                r.probability_abs_bias_le_0_02,
                r.probability_bias_reduction_ge_0_98,
            )
        )

    print(
        "interpretation: these quantiles and success frequencies describe "
        "randomized-split sensitivity at each calibration budget; they are "
        "empirical reliability summaries, not inferential confidence intervals"
    )


if __name__ == "__main__":
    main()
