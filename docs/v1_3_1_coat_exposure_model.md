# v1.3.1 — Cross-Fitted Coat Exposure Modeling

v1.3 established a simple MNAR baseline:

- naive observational mean: biased upward relative to randomized Coat ratings,
- item-frequency inverse-observation weighting: moved in the right direction,
  but removed only a minority of the observed gap.

v1.3.1 asks whether a richer exposure model can explain more of the
self-selection process without using the randomized ratings to fit that model.

## Exposure model

The observational Coat matrix induces a binary user-item observation matrix

```math
O_{ui} =
\begin{cases}
1 & \text{if user }u\text{ self-selected and rated item }i,\\
0 & \text{otherwise}
\end{cases}
```

We model

```math
P(O_{ui}=1)
=
\sigma\left(
\alpha
+b_u
+c_i
+p_u^\top q_i
\right)
```

This is a regularized logistic matrix-factorization exposure model.

It is strictly an **estimated observation propensity model**. These
propensities are not known logging probabilities.

## Cross-fitting

The full user-item pair grid is randomly partitioned into folds.

For each fold, the observation model is fitted on the other pair labels and
predicts the held-out pairs. The final propensity assigned to an observed
rating therefore comes from a model that did not train on that pair's
observation indicator.

Cross-fitting reduces same-pair overfitting of the nuisance exposure model. It
does not make the MNAR identification assumptions automatically true.

The randomized Coat ratings are never used to fit the exposure model.

## Estimator

For observed ratings, v1.3.1 uses the self-normalized inverse-observation
weighted mean

```math
\widehat\mu_{\mathrm{IPW}}
=
\frac{
\sum_{(u,i):O_{ui}=1} R_{ui}/\widehat p_{ui}
}{
\sum_{(u,i):O_{ui}=1} 1/\widehat p_{ui}
}
```

The comparison reports:

- naive observational mean,
- item-frequency weighted mean,
- cross-fitted matrix-factorization weighted mean,
- bias against the randomized empirical reference,
- fraction of the naive absolute bias removed,
- max weight,
- p99 weight,
- ESS fraction.

A minimum propensity floor is configurable and defaults to `0.01`. This is an
explicit stabilization choice and therefore part of the estimator definition.

## Run

```bash
python scripts/run_coat_exposure_model.py \
  --data-root data/coat
```

## Retain/reject rule

The matrix-factorization exposure model is retained only if it materially
reduces the absolute discrepancy from the randomized reference without creating
pathological weight concentration.

If it performs worse than item-frequency weighting, that is a valid rejection
result and should be preserved.

## Scientific boundary

This remains an MNAR rating/debiasing experiment.

It must not be presented as contextual-bandit IPS, and the estimated exposure
probabilities must not be described as known behavior-policy propensities.
