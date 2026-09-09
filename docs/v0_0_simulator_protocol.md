# v0.0 — Ground-Truth Simulator Protocol

## Research question

Can PolicyRecLab construct a one-step contextual-bandit environment where the
policy value is known exactly for the fixed synthetic context population and a
Monte Carlo rollout converges to that value?

## Data-generating process

For user/context vector `u_i`, item/action vector `v_a`, latent dimension `d`,
and item intercept `b_a`,

\[
\mu(x_i,a)
=
\sigma\left(
\frac{u_i^\top v_a}{\sqrt d}
+
b_a
\right).
\]

Rewards are then sampled as

\[
R \sim \mathrm{Bernoulli}(\mu(x_i,a)).
\]

The `sqrt(d)` scaling prevents logit variance from increasing mechanically with
latent dimension.

## v0.0 target policy

Uniform over all actions:

\[
\pi(a \mid x)=\frac{1}{|A|}.
\]

## Validation criterion

Compute the exact finite-population policy value and compare it against a Monte
Carlo rollout.

The automated test uses 120,000 decision rounds and requires an absolute error
below 0.006.

This threshold is intentionally loose relative to the worst-case Bernoulli
standard error so that the test checks correctness without becoming flaky.

## Explicit non-goals

v0.0 does not study:

- policy-induced selection bias,
- off-policy evaluation,
- propensity weighting,
- support violations,
- adaptive data collection,
- policy learning,
- feedback loops.

Those are introduced only after the simulator baseline is validated.
