# v1.3.6 — Yahoo R3 calibration-budget reliability

## Question

How reliable is the Yahoo R3 randomized-calibration correction at each
randomized-feedback budget, beyond the mean error reported in v1.3.5?

v1.3.5 established a diminishing-returns curve across calibration fractions.
v1.3.6 adds empirical split-to-split reliability summaries so that a budget is
not judged only by its average performance.

## Protocol

The default reliability run uses the same calibration fractions as v1.3.5:

- 1%
- 2.5%
- 5%
- 10%
- 20%

It increases the randomized split repetitions from 10 to 50 seeds, yielding
250 calibration/evaluation experiments.

For each calibration budget, PolicyRecLab reports:

- 5th, median, and 95th percentile Naive-Bayes bias;
- 90th and 95th percentile absolute Naive-Bayes bias;
- fraction of splits with absolute bias at most 0.01;
- fraction of splits with absolute bias at most 0.02;
- fraction of splits with at least 98% apparent bias reduction.

These thresholds are descriptive operating criteria for this benchmark, not
universal guarantees.

## Important inference boundary

The across-seed quantiles are empirical summaries of sensitivity to the random
calibration/evaluation split. They are **not** confidence intervals for the
population rating mean and should not be labeled as such.

The experiment continues to use randomized ratings for calibration and
held-out empirical evaluation. It does not show that the observational Yahoo
R3 ratings alone identify the MNAR exposure mechanism.

## Run

```bash
python scripts/run_yahoo_r3_calibration_reliability.py \
  --data-root data/yahoo_r3
```

The default 50-seed run is intentionally larger than v1.3.5 so that tail
quantiles and empirical success frequencies are less dependent on only a few
splits.
