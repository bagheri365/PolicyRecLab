# PolicyRecLab

A research lab for off-policy evaluation, exploration, and policy-generated bias
in contextual-bandit recommendation systems.

## Current milestone: v0.0

The first milestone builds a finite synthetic contextual-bandit environment with
known conditional reward probabilities and verifies that Monte Carlo policy
rollouts converge to the exact finite-population policy value.

No off-policy estimator is introduced yet.

## Install

```bash
python -m pip install -e ".[dev]"
pytest
```

See:

- `docs/ASSUMPTIONS.md`
- `docs/ESTIMAND.md`
- `docs/v0_0_simulator_protocol.md`

## Yahoo R3 MNAR calibration results

PolicyRecLab's Yahoo R3 branch studies self-selected rating exposure rather
than contextual-bandit logging. The observational mean rating is **2.892**
versus **1.820** on held-out randomized ratings. Item-frequency weighting
removes only **13.9%** of this gap.

A rating-dependent Naive-Bayes propensity model calibrated with randomized
ratings nearly reconstructs the randomized rating distribution, but this
success is structurally tied to the calibration subset rather than showing
that observational data alone identify the MNAR mechanism.

Calibration-budget experiments show diminishing average error and a stronger
reliability improvement between 5% and 10% randomized calibration. Across 50
splits, 10% calibration yields **94%** of runs with absolute bias at most 0.02
and **94%** with at least 98% apparent gap reduction. The 10% level is a useful
empirical operating point for this benchmark, not a universal threshold.

See [`docs/v1_3_7_yahoo_r3_results_synthesis.md`](docs/v1_3_7_yahoo_r3_results_synthesis.md)
for the full retained interpretation and identification caveat.

