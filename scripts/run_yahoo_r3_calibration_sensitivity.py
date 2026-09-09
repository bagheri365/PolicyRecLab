from __future__ import annotations

import argparse
from pathlib import Path

from policyreclab.datasets.yahoo_r3 import load_yahoo_r3_triples
from policyreclab.experiments.yahoo_r3_calibration_sensitivity import (
    run_yahoo_r3_calibration_sensitivity,
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
    parser.add_argument("--fractions", type=_parse_float_list, default=(0.01, 0.025, 0.05, 0.10, 0.20))
    parser.add_argument("--seeds", type=_parse_int_list, default=tuple(range(10)))
    parser.add_argument("--laplace", type=float, default=1.0)
    parser.add_argument("--min-propensity", type=float, default=1e-6)
    args = parser.parse_args()

    if args.train is not None:
        train_path = args.train
    else:
        standard_train = args.data_root / STANDARD_TRAIN
        train_path = standard_train if standard_train.exists() else args.data_root / "user.txt"

    if args.randomized is not None:
        randomized_path = args.randomized
    else:
        standard_test = args.data_root / STANDARD_TEST
        randomized_path = standard_test if standard_test.exists() else args.data_root / "random.txt"

    observational = load_yahoo_r3_triples(train_path)
    randomized = load_yahoo_r3_triples(randomized_path)

    runs, summaries = run_yahoo_r3_calibration_sensitivity(
        observational,
        randomized,
        calibration_fractions=args.fractions,
        seeds=args.seeds,
        laplace=args.laplace,
        min_propensity=args.min_propensity,
    )

    print(f"n_observational: {len(observational.ratings)}")
    print(f"n_randomized_total: {len(randomized.ratings)}")
    print(f"n_runs: {len(runs)}")
    print("summary:")
    for s in summaries:
        print(
            "fraction={:.4f} n_runs={} mean_n_cal={:.1f} "
            "mean_abs_cal_eval_gap={:.6f} std_cal_eval_gap={:.6f} "
            "mean_abs_nb_bias={:.6f} std_nb_bias={:.6f} "
            "mean_bias_reduction={:.4%} min_bias_reduction={:.4%} "
            "max_bias_reduction={:.4%} mean_max_weight={:.3f} "
            "mean_p99_weight={:.3f} mean_ess_fraction={:.4%}".format(
                s.calibration_fraction,
                s.n_runs,
                s.mean_n_calibration,
                s.mean_abs_calibration_evaluation_gap,
                s.std_calibration_evaluation_gap,
                s.mean_abs_naive_bayes_bias,
                s.std_naive_bayes_bias,
                s.mean_bias_reduction,
                s.min_bias_reduction,
                s.max_bias_reduction,
                s.mean_max_weight,
                s.mean_p99_weight,
                s.mean_ess_fraction,
            )
        )

    print(
        "interpretation: sensitivity is driven by how well the randomized "
        "calibration subset estimates the target rating distribution; the "
        "held-out randomized subset remains the empirical evaluation reference"
    )


if __name__ == "__main__":
    main()
