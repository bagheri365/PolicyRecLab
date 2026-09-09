# D1 documentation audit

## Scope decision

**Experimental scope is frozen at v1.3.7.**

New estimators, datasets, or experimental branches are out of scope for the
finishing phase unless documentation or reproducibility work reveals a genuine
correctness defect.

## What is already strong

The repository already contains a complete experimental progression:

- exact-value synthetic simulation;
- support and positivity failures;
- IPS, clipped IPS, SNIPS, DM, and DR behavior;
- exploration/evaluation tradeoffs;
- policy-selection reuse and feedback loops;
- reliability diagnostics;
- Open Bandit Dataset replication;
- Coat MNAR exposure-model comparisons, including a retained negative result;
- Yahoo R3 randomized-calibration sensitivity and reliability analysis.

The project therefore has enough experimental evidence to support a final
research narrative without adding another estimator.

## Documentation risks to resolve before release

### 1. Fragmented narrative

Milestone documents preserve the research history well, but a reader should
not need to reconstruct the central argument from many files. D2 should
provide one final report that explains the progression from identification to
estimation to reliability.

### 2. Benchmark-category confusion

The final documentation must keep Open Bandit Dataset separate from Coat and
Yahoo R3 at the data-generating-process level. The common theme is
policy/exposure-generated observation bias, not an assertion that all three
datasets are contextual-bandit logs.

### 3. Numerical-claim traceability

Headline real-data numbers should be traceable to the experiment that produced
them. D2 should avoid introducing new numerical claims that are not already
supported by retained experiment output.

### 4. Diagnostic overinterpretation

Weight diagnostics, including ESS, should remain diagnostics. They must not be
used as proof of identification, estimator correctness, or inferential
precision.

### 5. Yahoo structural interpretation

The final report must state that the Naive-Bayes correction reconstructs the
rating distribution supplied by randomized calibration data. Its near-match
to held-out randomized ratings validates calibration transfer, not
observational-only identification.

### 6. Reproduction path

The repository needs one authoritative D3 guide for environment setup,
synthetic runs, optional real-data runs, local data layout, and test commands.

### 7. External-data boundary

Raw external datasets should not be bundled merely to simplify reproduction.
The release guide must distinguish code from locally obtained data and record
provenance/licensing uncertainty where relevant.

## Release criterion

PolicyRecLab is ready for a finished research release when a new reader can:

1. understand the estimand and identification assumptions;
2. follow the experimental progression without reading every milestone file;
3. distinguish synthetic truth, empirical references, and estimated values;
4. reproduce the code/tests from a fresh clone;
5. understand which real datasets must be obtained separately;
6. see failed and rejected experiments alongside retained ones;
7. understand the limitations of every headline real-data conclusion.

D1 establishes this checklist. D2 and D3 should close it without expanding the
experimental scope.
