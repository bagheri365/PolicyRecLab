# v0.8 — Doubly Robust Estimation

## Research question

Can a reward model reduce IPS variance without inheriting all of the direct
method's sensitivity to reward-model misspecification?

For one-step contextual bandits, PolicyRecLab implements

```math
\hat V_{\mathrm{DR}}
=
\frac{1}{n}
\sum_i
\left[
\hat r(x_i,\pi_e)
+
\frac{
\pi_e(a_i\mid x_i)
}{
\pi_b(a_i\mid x_i)
}
\left(
r_i-\hat r(x_i,a_i)
\right)
\right]
```

where

```math
\hat r(x,\pi_e)
=
\sum_a
\pi_e(a\mid x)\hat r(x,a)
```

The first term is the reward-model prediction under the target policy. The
second uses importance weighting to correct model residuals on logged actions.

## Scope of the v0.8 robustness claim

The primary v0.8 branch uses behavior propensities supplied by the known
synthetic logging policy.

They are treated as correct.

Therefore this milestone tests one side of the doubly robust story:

> With support and correct behavior propensities, DR can remain valid even when
> the reward model is misspecified.

It does **not** yet demonstrate the complementary case in which an estimated
behavior model is misspecified but the reward model is correct.

That requires a separate estimated-propensity branch and should not be implied
by this experiment.

## Controlled reward-model branches

v0.8 reuses the two reward-model constructions from v0.7.

### Oracle reward model

The simulator's exact conditional reward means are supplied to DR.

This is a synthetic diagnostic, not a realistic learned production model.

With an accurate reward model, the importance-weighted correction acts on
reward residuals rather than raw rewards, which can reduce variance.

### Deliberately misspecified reward model

The global-mean model ignores context and action and is fitted from each
behavior-policy log.

Its direct-method estimate is systematically biased in the controlled setup.

DR applies the propensity-weighted residual correction to this wrong model.
Because the behavior propensities are correct and support holds, repeated
sampling should center the DR estimate near the target value despite the
misspecified reward surface.

## Experimental comparison

Across repeated independent logs, v0.8 compares:

- IPS,
- misspecified DM,
- DR with oracle reward means,
- DR with the misspecified global-mean reward model.

Every estimator is compared against the exact finite-population target value.

Reported metrics are:

- mean estimate,
- bias,
- repeated-sample variance,
- RMSE.

## Interpretation

The misspecified-DM comparison illustrates the robustness benefit:

```math
\text{wrong reward model}
+
\text{correct propensities}
\;\Longrightarrow\;
\text{DR can still be valid}
```

This statement requires the baseline assumptions, including contextual support
and valid behavior propensities.

It does not mean DR is immune to weak overlap. Its correction term still
contains

```math
\frac{\pi_e(a\mid x)}{\pi_b(a\mid x)}
```

so poor overlap can still produce high variance.

Nor does it mean arbitrary nuisance-model errors cancel.

## Support remains mandatory

v0.8 refuses exact support violations.

A reward model can produce predictions for unsupported actions, but those
predictions do not create nonparametric identification. Using them beyond
support would be model-based extrapolation requiring additional assumptions.

## Nuisance-model fitting caveat

The intentionally simple global-mean reward model is fitted on each logged
sample and reused in the DR calculation. This milestone is designed to isolate
the algebraic robustness mechanism, not to establish inference guarantees for
flexible learned nuisance models.

Later learned-model experiments should use sample splitting or cross-fitting
where appropriate.

## Next milestone

v0.9 separates policy selection from policy evaluation. It will show why
searching over many candidate policies and reporting the largest OPE estimate
on the same data creates selection optimism, even when the underlying OPE
estimator is otherwise valid.
