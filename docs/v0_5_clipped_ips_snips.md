# v0.5 — Clipped IPS and SNIPS

## Research question

When weak overlap makes ordinary IPS unstable, can simple importance-weight
modifications improve finite-sample error?

v0.5 compares:

1. ordinary IPS,
2. clipped IPS,
3. self-normalized IPS (SNIPS).

The goal is not to crown a universally best estimator. It is to measure their
bias-variance tradeoffs under the same controlled logging process.

## Clipped IPS

For threshold \(c > 0\),

\[
\tilde w_i = \min(w_i, c),
\]

and

\[
\hat V_{\mathrm{CIPS}}
=
\frac1n
\sum_i
\tilde w_i r_i.
\]

Clipping limits the influence of extreme weights and can substantially reduce
variance.

However, clipping generally introduces bias because it changes the contribution
of observations whose correct importance ratio exceeds the threshold.

The threshold is therefore an explicit tuning choice, not a free correction.

## SNIPS

SNIPS normalizes by the observed total importance weight:

\[
\hat V_{\mathrm{SNIPS}}
=
\frac{\sum_i w_i r_i}
     {\sum_i w_i}.
\]

SNIPS is generally finite-sample biased. Under suitable conditions it can be
consistent and may reduce variance relative to ordinary IPS.

It is not universally superior to IPS.

## Experimental protocol

The target remains uniform and the behavior policy remains epsilon-greedy, as
in v0.4.

For each weak-overlap setting, the complete logging-and-estimation process is
repeated many times. Every estimator is evaluated against the same exact
finite-population target value.

For each estimator v0.5 reports:

- mean estimate,
- bias,
- repeated-sample variance,
- RMSE.

For clipped IPS it also records the fraction of observed importance weights
that exceed the clipping threshold.

## Interpretation rules

A lower variance estimate is not automatically better if the resulting bias
increases enough to worsen RMSE.

Likewise, a single run that happens to land close to ground truth is not
evidence that one estimator dominates another. The comparison is based on
repeated-sample behavior.

The central quantities are

\[
\mathrm{Bias}(\hat V)
=
\mathbb E[\hat V] - V,
\]

\[
\mathrm{Var}(\hat V)
=
\mathbb E[(\hat V-\mathbb E[\hat V])^2],
\]

and

\[
\mathrm{RMSE}(\hat V)
=
\sqrt{\mathbb E[(\hat V-V)^2]}.
\]

## Scientific message

v0.4 showed:

\[
\text{identified}
\not\Rightarrow
\text{well estimated}.
\]

v0.5 adds:

\[
\text{variance reduction}
\not\Rightarrow
\text{unbiasedness}.
\]

Clipping and self-normalization are estimator choices with measurable
tradeoffs, not generic fixes for weak overlap.

## Non-goals

v0.5 does not yet introduce:

- learned reward models,
- direct-method estimation,
- doubly robust estimation,
- confidence intervals,
- adaptive-logging inference.

## Next milestone

v0.6 turns from estimator repair to data collection: it studies why exploration
can improve future policy evaluability.
