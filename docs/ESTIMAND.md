# Baseline Estimand

For a fixed finite population of contexts `x_1, ..., x_N`, action set `A`, target
policy `pi`, and conditional mean reward `mu(x, a)`, PolicyRecLab v0.0 targets the
finite-population one-step policy value

\[
V_{\mathrm{finite}}(\pi)
=
\frac{1}{N}
\sum_{i=1}^{N}
\sum_{a \in A}
\pi(a \mid x_i)\mu(x_i,a).
\]

In the synthetic environment, `mu(x_i, a)` is known by construction, so this
quantity can be computed directly without Monte Carlo approximation.

The v0.0 Monte Carlo experiment is a validation exercise: sampled policy reward
should converge to this finite-population value as the number of simulated
decision rounds grows.

This is not yet an off-policy estimator. No IPS, SNIPS, direct-method, or doubly
robust claim is made in v0.0.
