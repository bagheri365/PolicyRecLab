# v0.7 — Direct Reward Modeling

## Research question

Can we avoid unstable importance weights by modeling rewards directly?

The direct method (DM) estimates the conditional reward surface

```math
\mu(x,a)=\mathbb E[R\mid X=x,A=a]
```

with a model \(\hat\mu(x,a)\), then evaluates the target policy as

```math
\hat V_{\mathrm{DM}}
=
\frac{1}{N}
\sum_i
\sum_a
\pi_e(a\mid x_i)\hat\mu(x_i,a)
```

DM does not multiply observed rewards by inverse propensities. That can make it
far less variable than IPS. Its vulnerability is different: it relies on the
reward model being accurate for the context-action regions used by the target.

## Two deliberately separated branches

v0.7 uses two reward-model branches.

### Oracle simulator model

The oracle model returns the synthetic environment's exact conditional reward
means.

It is not a learned production estimator. It is a diagnostic upper bound that
answers:

> What would DM do if the reward surface were known exactly?

Because PolicyRecLab's simulator has a finite fixed context population, oracle
DM must equal the exact finite-population target value up to numerical
precision.

### Intentionally misspecified global-mean model

The second model is fitted from each logged dataset but ignores both context and
action. It predicts the behavior log's mean reward for every context-action
pair.

This is deliberately misspecified.

When the behavior policy is exploitative and the target is uniform, the
behavior-policy reward mean is generally not the uniform target's value.
Applying that constant prediction to every target action therefore creates
persistent model bias.

## Controlled comparison

The environment and target policy are fixed.

The behavior policy is epsilon-greedy with positive support, so IPS remains
identified under the baseline assumptions.

Across repeated independent logs, v0.7 compares:

- IPS,
- oracle DM,
- misspecified DM.

For each branch it reports:

- mean estimate,
- bias,
- repeated-sample variance,
- RMSE.

## Interpretation

The experiment is designed to expose an important failure mode:

```math
\text{low variance}
\not\Rightarrow
\text{accurate}
```

A misspecified reward model can produce estimates that are very stable across
datasets while remaining systematically wrong.

Conversely, the oracle branch demonstrates that DM is exact when the
conditional reward surface is exact.

The oracle result should not be interpreted as evidence that real reward models
are exact or easy to learn.

## Model fitting and evaluation discipline

The misspecified model is fitted independently inside each repeated logging
sample.

This milestone intentionally uses a simple model to isolate misspecification.
When more flexible learned nuisance models are introduced, PolicyRecLab will
use held-out fitting or cross-fitting where appropriate rather than silently
training and evaluating nuisance predictions on the same observations.

## Identification caveat

A reward model can output predictions even for actions with no logged support.
That does not make an otherwise unsupported target nonparametrically
identified.

Such predictions are model-based extrapolation and require additional
assumptions.

v0.7 therefore retains positive behavior support and does not use DM to hide
the zero-support failure studied in v0.2.

## Next milestone

v0.8 introduces doubly robust estimation and combines reward modeling with
importance weighting. The primary branch will keep known behavior propensities
and stress reward-model misspecification before adding any separate
estimated-propensity experiment.
