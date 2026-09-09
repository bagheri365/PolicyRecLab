# v1.3.2 — Coat Exposure-Model Diagnostics

v1.3.1 rejected the cross-fitted logistic user-item exposure model as a
debiasing correction: it produced very stable inverse weights but removed less
of the observational-versus-randomized rating gap than the simpler
item-frequency baseline.

v1.3.2 does not add another correction model. It diagnoses why.

## Question

> Does the richer exposure model actually discriminate observed from
> unobserved user-item pairs, and how much propensity variation does it learn?

The randomized Coat ratings are not used anywhere in this diagnostic milestone.

## Metrics

For item-frequency and cross-fitted logistic matrix factorization, report:

- Brier score,
- binary log loss,
- mean predicted propensity for observed pairs,
- mean predicted propensity for unobserved pairs,
- observed-minus-unobserved separation gap,
- p01/p10/p50/p90/p99 propensity quantiles,
- standard deviation of user-average propensities,
- standard deviation of item-average propensities,
- equal-width-bin expected calibration error.

These metrics separate three ideas that should not be conflated:

1. **discrimination** — can the model distinguish observed from unobserved pairs?
2. **calibration** — do predicted probabilities agree with empirical frequencies?
3. **useful reweighting variation** — are propensity differences large enough to
   materially alter the observational rating distribution?

A model can have stable weights and reasonable probability loss while still
being too flat to correct MNAR rating bias.

## Cross-fitting

Matrix-factorization predictions remain pair-level cross-fitted: each
user-item pair is scored by a model that did not train on that pair's
observation indicator.

The item-frequency baseline is intentionally not cross-fitted because its role
is descriptive: per-item observation rate from the full observational matrix.
Its predictive scores should therefore not be interpreted as a perfectly fair
out-of-sample benchmark against the cross-fitted model. The primary purpose is
to understand structure and propensity spread, not declare a leaderboard.

## Run

```bash
python scripts/run_coat_exposure_diagnostics.py \
  --data-root data/coat
```

## Decision rule

Do not tune the exposure model against randomized rating outcomes.

If the MF model shows weak observed/unobserved separation and narrow propensity
spread, the v1.3.1 failure is consistent with under-differentiated exposure
scores.

If it predicts exposure well but still fails to debias ratings, the stronger
lesson is that predictive exposure fit alone is insufficient for the MNAR
correction required by the rating estimand.

Either outcome is retained.
