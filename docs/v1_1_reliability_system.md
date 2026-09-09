# v1.1 — Integrated OPE Reliability System

## Research question

How should PolicyRecLab report an offline policy value so readers can judge
whether the number is identified, estimable, and inferentially defensible?

The answer is: never report the value alone.

A final OPE result should carry:

- estimator,
- logging process,
- behavior-propensity source,
- contextual support status,
- overlap diagnostics,
- target-policy selection procedure,
- inference method and validity caveat.

## Identification, estimation, and inference

v1.1 keeps three questions separate.

### Identification

For IPS, contextual support requires

```math
\pi_e(a\mid x)>0
\Rightarrow
\pi_b(a\mid x)>0
```

If support fails, the report marks the target as not nonparametrically
identified and refuses to present a finite IPS value as though it were valid.

### Estimation

When support holds, weak overlap can still make estimation unstable.

The report includes:

- minimum positive behavior probability on target support,
- maximum importance weight,
- p99 importance weight,
- ESS,
- ESS fraction.

ESS remains a descriptive weight-concentration heuristic, not a literal
inferential sample size.

### Inference

The report records whether logging is:

- IID/static,
- adaptive,
- unknown.

Adaptive logging receives an explicit warning that ordinary IID intervals
should not automatically be assumed valid across the adaptive sequence.

## Policy selection is part of reliability

A policy selected on the same final evaluation data receives a
selection-optimism warning.

Cross-fitting nuisance models does not by itself repair repeated target-policy
search on the same final evaluation sample.

## Propensity source is part of reliability

Propensities are labeled as:

- known,
- estimated,
- unknown.

Estimated propensities trigger a model-error caveat. Unknown propensity source
prevents a clean IPS validity claim.

## Controlled failure-mode study

v1.1 creates five reports:

1. healthy overlap,
2. weak overlap,
3. zero support,
4. same-data target-policy selection,
5. adaptive logging paired with an IID-style inference label.

These are intentionally not collapsed into one universal reliability score.

Different failures mean different things:

- zero support: identification failure,
- weak overlap: estimation-stability problem,
- same-data selection: selection-validity problem,
- adaptive logging with IID inference: inference problem.

## Reporting contract

Every final policy value should state:

1. estimand,
2. logging process,
3. propensity source,
4. support condition,
5. target-policy selection procedure,
6. estimator,
7. overlap diagnostics,
8. inference method.

## Next milestone

v1.2 moves the framework to the Open Bandit Dataset, complementing exact
synthetic truth with real production contextual-bandit logs and supplied
behavior-policy action probabilities.
