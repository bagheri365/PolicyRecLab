# v1.3.3 — Feature-Aware Coat Debiasing

v1.3.1 showed that a latent user-item exposure model produced stable but nearly
constant inverse weights. v1.3.2 diagnosed that failure directly. v1.3.3
changes the information set rather than increasing latent-model complexity.

The original Coat study released user and item features alongside the ratings
and estimated propensities. The paper describes propensity estimation with
regularized logistic regression using user covariates (gender, age group,
location, fashion awareness) and item covariates (gender, coat type, color,
and promotion/front-page status).

PolicyRecLab loads the official Coat layout:

- `user_item_features/user_features.ascii`
- `user_item_features/item_features.ascii`
- `propensities.ascii` when present.

## Feature-aware exposure model

For raw released user features `u` and item features `v`:

\[
P(O_{ui}=1)=
\sigma(\alpha + u^\top\beta_u + v^\top\beta_i + u^\top Wv).
\]

The interaction term includes all user-feature × item-feature interactions
without materializing a huge pairwise design matrix.

This is scientifically close to the feature-interaction logistic model
described in the Coat paper, but it is **not claimed to reproduce the original
authors' exact preprocessing, regularization search, or fitted propensities**.

## Protocol

The observational pair grid is split into folds. Each pair receives a
propensity from a model that did not train on that pair's observation label.
The randomized Coat ratings are never used for propensity fitting or
hyperparameter selection; they remain an empirical reference only after the
observational estimator is computed.

The study compares:

1. naive observational mean,
2. item-frequency self-normalized inverse-observation weighting,
3. cross-fitted feature-aware weighting,
4. optionally, a separately supplied released propensity matrix.

All propensity quantities here are estimated/supplied observation
probabilities, not known contextual-bandit behavior-policy probabilities.

## Run

```bash
python scripts/run_coat_feature_propensity.py \
  --data-root data/coat
```

The local official Coat layout includes `propensities.ascii`, so PolicyRecLab
auto-detects and benchmarks it when present. You can still override that path
explicitly:

```bash
python scripts/run_coat_feature_propensity.py \
  --data-root data/coat \
  --released-propensities /path/to/propensity-matrix.ascii
```

For the feature-aware model, report Brier score, log loss, observed/unobserved
propensity separation, p01/p50/p99 propensities, maximum and p99 inverse
weights, and ESS fraction.

Do not tune against the randomized rating mean. Retain the feature-aware model
only if its pre-specified fit materially reduces the absolute
observational-versus-randomized discrepancy without pathological weight
concentration. A failure is preserved.

A supplied released-propensity result must be labeled an **estimated
propensity benchmark from the Coat release**, not a known propensity.
