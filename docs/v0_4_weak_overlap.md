# v0.4 — Weak Overlap and Importance-Weight Variance

## Research question

Does satisfying positivity guarantee that inverse propensity scoring is
practically reliable?

No. Exact support is an identification condition. It does not guarantee good
finite-sample estimation.

## Controlled design

The target policy is uniform.

The behavior policy is epsilon-greedy:

```math
\pi_b(a \mid x)
=
(1-\epsilon)\mathbf{1}\{a=a^*(x)\}
+
\frac{\epsilon}{|\mathcal A|}
```

For every `epsilon > 0`, every action has positive behavior probability, so the
uniform target remains supported.

For non-greedy actions,

```math
\pi_b(a \mid x)=\frac{\epsilon}{|\mathcal A|}
```

Under the uniform target,

```math
\pi_e(a \mid x)=\frac{1}{|\mathcal A|}
```

so their importance weights are

```math
w
=
\frac{
1/|\mathcal A|
}{
\epsilon/|\mathcal A|
}
=
\frac{1}{\epsilon}
```

As epsilon shrinks, these observations become rarer but much more influential.

## Repeated-sample protocol

For each epsilon value, v0.4 repeats the full logging-and-estimation experiment
many times while holding the reward environment fixed.

It reports:

- true finite-population target value,
- mean IPS estimate,
- repeated-sample bias,
- repeated-sample variance,
- RMSE,
- average maximum importance weight,
- average 99th-percentile weight,
- average ESS heuristic,
- average ESS fraction.

This separates identification from estimator stability.

## Effective sample size diagnostic

The project reports the common importance-weight heuristic

```math
\widehat{\mathrm{ESS}}
=
\frac{
\left(\sum_i w_i\right)^2
}{
\sum_i w_i^2
}
```

This quantity is useful for describing weight concentration. It is **not** a
literal inferential sample size, a confidence-interval guarantee, or proof that
an OPE estimate is reliable.

PolicyRecLab therefore labels ESS as a diagnostic only.

## Scientific interpretation

When epsilon becomes small:

1. support still holds,
2. the target remains nonparametrically identifiable under the baseline
   assumptions,
3. importance weights become more concentrated,
4. repeated-sample variance and RMSE can grow sharply.

The key lesson is

```math
\text{identified}
\not\Rightarrow
\text{well estimated}
```

This is different from v0.2. There, exact zero support made the target value
unidentified. Here, propensities remain positive but estimation becomes fragile.

## Non-goals

v0.4 does not yet repair the problem with:

- weight clipping,
- self-normalization,
- direct reward modeling,
- doubly robust estimation.

Those interventions come later so their bias/variance tradeoffs can be measured
against the unmodified IPS baseline.

## Next milestone

v0.5 introduces clipped IPS and SNIPS and compares bias, variance, and RMSE
against standard IPS under weak overlap.
