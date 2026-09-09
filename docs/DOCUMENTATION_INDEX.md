# PolicyRecLab documentation index

PolicyRecLab is experimentally frozen at v1.3.7. The remaining work is
documentation, reproducibility, and release preparation unless a correctness
issue is discovered.

## Start here

1. [`../README.md`](../README.md) — project overview and headline findings.
2. [`DOCUMENTATION_INDEX.md`](DOCUMENTATION_INDEX.md) — map of the research record.
3. [`FINAL_REPORT.md`](FINAL_REPORT.md) — final paper-style research narrative.
4. [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — fresh-clone and data-reproduction guide.
5. [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) — final release audit.

The final research report and reproducibility guide are complete. D3 closes the documentation phase with a release checklist.

## Research progression

### Synthetic contextual-bandit foundation

| Milestone | Question | Status |
|---|---|---|
| v0.0 | Can the simulator expose exact policy value? | Retained |
| v0.1 | How does logging policy change observed feedback? | Retained |
| v0.2 | When does support make a target value identifiable? | Retained |
| v0.3 | Does IPS recover supported target-policy value? | Retained |
| v0.4 | What happens under weak overlap? | Retained |
| v0.5 | How do clipping and SNIPS trade bias for variance? | Retained |
| v0.6 | How does exploration affect online reward and later evaluation? | Retained |
| v0.7 | How does DM behave under reward-model misspecification? | Retained |
| v0.8 | When does doubly robust estimation help? | Retained |
| v0.9 | What changes when policy selection and evaluation reuse data? | Retained |
| v1.0 | How do policy-generated feedback loops change future data? | Retained |
| v1.1 | Can estimator output be paired with reliability diagnostics? | Retained |

### Real-data validation

| Milestone | Benchmark | Main role | Status |
|---|---|---|---|
| v1.2 | Open Bandit Dataset | Production contextual-bandit replication | Retained |
| v1.2.1 | Open Bandit Dataset | Overlap and estimator-sensitivity audit | Retained |
| v1.3 | Coat / Yahoo R3 | MNAR benchmark scaffold | Retained |
| v1.3.1 | Coat | Cross-fitted latent exposure model | Rejected as effective debiaser |
| v1.3.2 | Coat | Diagnose latent exposure-model failure | Retained diagnostic |
| v1.3.3 | Coat | Feature-aware propensity model | Retained |
| v1.3.4 | Yahoo R3 | Rating-dependent Naive-Bayes calibration | Retained with identification caveat |
| v1.3.5 | Yahoo R3 | Randomized-calibration sensitivity | Retained |
| v1.3.6 | Yahoo R3 | Calibration-budget reliability | Retained |
| v1.3.7 | Yahoo R3 | Results synthesis | Retained |

## Evidence boundaries

The documentation must preserve these distinctions.

### Contextual-bandit OPE versus MNAR rating exposure

Open Bandit Dataset is used as a one-step production contextual-bandit
benchmark with supplied logging propensities. Coat and Yahoo R3 are MNAR
rating-exposure benchmarks. Coat/Yahoo results must not be described as if
they were production contextual-bandit logs.

### Identification versus estimation

A numerically stable estimator does not establish identification. Unsupported
target actions remain nonparametrically unidentified from the logged data.
Weak overlap can produce severe weight concentration even when a point
estimate happens to agree with a reference.

### ESS interpretation

Effective sample size is used as a weight-concentration diagnostic. It is not
a literal inferential sample size and must not be presented as one.

### Yahoo R3 calibration dependence

The Yahoo Naive-Bayes weighted mean is structurally tied to the randomized
calibration rating distribution. The result demonstrates the value of
randomized calibration information; it does not show that the observational
ratings alone identify the MNAR mechanism.

### Failed experiments are evidence

The v1.3.1 Coat latent exposure model is intentionally retained as a negative
result. Its stable weights and high ESS did not imply successful debiasing.

## Documentation finishing plan

### D1 — audit and structure

- freeze the experimental scope;
- establish this documentation map;
- preserve interpretation boundaries;
- identify release-blocking documentation gaps.

### D2 — final research narrative — complete

- README rewritten as the project entry point;
- `docs/FINAL_REPORT.md` added;
- simulator, OBD, Coat, and Yahoo evidence synthesized;
- retained, rejected, and caveated results kept explicit.

### D3 — reproducibility and release audit — complete

- `docs/REPRODUCIBILITY.md` added;
- fresh-clone setup and experiment commands documented;
- external-data provenance and local-only raw-data rules documented;
- `docs/RELEASE_CHECKLIST.md` added for the final metadata/link/test audit.

## Release-blocking gaps after D1

D1 intentionally does not fill these gaps:

- no single final paper-style report yet;
- no single fresh-clone reproduction guide yet;
- real-data acquisition/provenance instructions need a final consolidated pass;
- final release metadata and version/tag policy need a consistency audit;
- numerical claims in the final narrative still need a last traceability pass
  against experiment outputs.

These are D2/D3 tasks rather than reasons to add new experiments.
