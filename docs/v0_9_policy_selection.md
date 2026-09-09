# v0.9 — Policy Selection vs. Policy Evaluation

## Research question

What goes wrong when the same logged sample is used both to choose a policy and
to report that policy's offline value?

Even when an OPE estimator is valid for each fixed candidate policy, searching
over many candidates and reporting the largest estimate creates a new source of
bias: selection optimism.

## Fixed-policy validity is not enough

Suppose candidate policies

```math
\pi_1,\ldots,\pi_M
```

are fixed before the evaluation data are observed.

For each fixed policy, an OPE estimator may satisfy

```math
\mathbb E[\hat V(\pi_j)] = V(\pi_j)
```

under its assumptions.

But if we select

```math
\hat j
=
\arg\max_j \hat V(\pi_j)
```

and then report

```math
\hat V(\pi_{\hat j})
```

using the same sample, the reported value has been selected partly because its
estimation error was favorable.

In general,

```math
\mathbb E\left[
\hat V(\pi_{\hat j}) - V(\pi_{\hat j})
\right]
> 0
```

can occur even though each candidate's estimator is unbiased before selection.

## Controlled experiment

v0.9 creates a fixed pool of stochastic candidate policies before any logged
evaluation samples are generated.

Candidates use noisy versions of the simulator reward surface with different
noise levels and softmax temperatures. They have different exact
finite-population values.

The logging policy is uniform, giving positive contextual support to every
candidate.

Within each repetition:

1. generate a selection log,
2. estimate every candidate's value with IPS,
3. choose the candidate with the largest IPS estimate,
4. record that same selection-set estimate,
5. compare it with the selected policy's exact simulator value,
6. generate a fresh independent holdout log,
7. evaluate only the already-selected policy on that holdout.

The holdout is never used to choose or re-choose the policy.

## Primary diagnostics

For each candidate-pool size, v0.9 reports:

- mean selection-set estimate,
- mean exact value of selected policies,
- mean fresh-holdout estimate,
- mean selection optimism,
- mean holdout error,
- mean regret of the selected policy relative to the best true candidate.

Selection optimism is

```math
\hat V_{\text{selection}}(\pi_{\hat j})
-
V(\pi_{\hat j})
```

Holdout error is

```math
\hat V_{\text{holdout}}(\pi_{\hat j})
-
V(\pi_{\hat j})
```

## Why the fresh holdout works

The selected policy is random from the perspective of the selection sample.

But after selection is complete, it is fixed before the independent holdout is
observed.

Therefore the holdout evaluates an already-selected policy rather than
participating in the selection procedure.

This restores the ordinary fixed-policy interpretation for the final
evaluation sample, subject to the usual OPE assumptions.

## Candidate count and winner's curse

As more candidates are searched, there are more opportunities for one policy to
receive a favorable estimation error.

The controlled experiment is designed to make this effect visible.

PolicyRecLab does not claim that selection optimism must increase monotonically
for every realized dataset or every possible candidate family. The scientific
point is that repeated search on the same evaluation data invalidates the naive
interpretation of the winning estimate.

## Selection quality versus evaluation honesty

Two distinct questions must remain separate:

1. Did the selection procedure choose a genuinely good policy?
2. Is the reported value of the chosen policy an honest estimate?

A fresh holdout addresses the second question. It does not guarantee that noisy
selection found the truly best candidate.

That is why v0.9 also reports selected-policy regret.

## Cross-fitting caveat

Cross-fitting is useful for nuisance estimation in DM/DR.

It does not, by itself, make it valid to repeatedly search over target policies
using the same final evaluation observations and then report the maximum as if
the policy had been fixed in advance.

Final policy selection and final policy evaluation must be separated.

## Project rule strengthened

Every reported final policy value should state how the target policy was
selected and whether the evaluation sample was independent of that selection.

## Next milestone

v1.0 introduces repeated policy feedback loops: policies generate exposure,
logged exposure changes future training data, and successive policies can
amplify or alter the data distribution they learn from.
