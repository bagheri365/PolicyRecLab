# v0.6 — Why Exploration Matters

## Research question

How does the logging policy's exploration rate affect both:

1. its immediate online reward, and
2. our ability to evaluate a different future policy from its logs?

Earlier milestones treated the logging policy as given. v0.6 makes data
collection itself part of the experimental design.

## Controlled setup

The synthetic environment is fixed.

The behavior policy is epsilon-greedy with respect to the simulator's oracle
reward means:

```math
\pi_\epsilon(a\mid x)
=
(1-\epsilon)\mathbf{1}\{a=a^*(x)\}
+
\frac{\epsilon}{|\mathcal A|}
```

The future evaluation target is the uniform policy.

This target is intentionally different from the increasingly exploitative
behavior policy. It creates a clean setting in which less exploration can
improve immediate behavior reward while degrading overlap for future OPE.

The oracle scores are a synthetic intervention. They do not imply that a real
production recommender knows the true reward function.

## Two objectives

For every exploration rate, v0.6 measures the behavior policy's exact
finite-population value:

```math
V(\pi_b)
```

This is the immediate data-collection objective.

It separately evaluates the fixed future target with IPS and records:

- exact target value,
- repeated-sample IPS bias,
- repeated-sample IPS variance,
- IPS RMSE,
- minimum behavior probability on target support,
- maximum importance-weight diagnostics,
- ESS fraction heuristic.

This is the future evaluability objective.

## Why the tradeoff appears here

For a non-greedy action under epsilon-greedy logging,

```math
\pi_b(a\mid x)
=
\frac{\epsilon}{|\mathcal A|}
```

For the uniform future target,

```math
\pi_e(a\mid x)
=
\frac{1}{|\mathcal A|}
```

Therefore its importance ratio is

```math
w=\frac{1}{\epsilon}
```

As epsilon falls, those target-relevant observations become rarer and more
influential.

At the same time, because the behavior policy exploits the simulator's oracle
best action more often, its immediate expected reward increases in this
controlled environment.

## Interpretation

This experiment demonstrates a data-collection tradeoff:

```math
\text{immediate behavior reward}
\quad\leftrightarrow\quad
\text{future evaluability}
```

It does **not** establish a universal theorem that more exploration always
improves every OPE problem, or that exploitation always improves online reward.

The result depends on:

- the environment,
- the behavior-policy family,
- the future target policy,
- sample size,
- estimator,
- and reward structure.

The scientific claim is narrower: exploration changes the support and weight
distribution available to future evaluation, so its consequences should be
measured rather than treated as an implementation detail.

## Identification versus estimation

All epsilon values in this milestone satisfy

```math
\epsilon > 0
```

Thus the uniform target retains exact contextual support.

v0.6 studies estimation quality under changing overlap, not the zero-support
identification failure from v0.2.

## Practical implication

A logging policy can be attractive for immediate reward yet produce data that
are poor for evaluating plausible future policies.

This motivates treating exploration as part of the measurement system, not
only as an online reward optimization choice.

## Next milestone

v0.7 introduces direct reward modeling and studies a different failure mode:
low-variance estimates that can become biased when the reward model is
misspecified.
