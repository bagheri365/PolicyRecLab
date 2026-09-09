# v1.3.5 — Yahoo R3 randomized-calibration sensitivity

## Question

How much randomized feedback is needed for the Yahoo R3 rating-dependent
Naive-Bayes MNAR correction to generalize reliably to an untouched randomized
evaluation sample?

v1.3.4 showed that a 5% randomized calibration subset produced a Naive-Bayes
weighted observational mean close to the remaining 95% randomized reference.
Because the Naive-Bayes inverse-weighted mean is structurally tied to the
calibration rating distribution, the scientifically relevant quantity is the
stability of calibration-to-evaluation agreement as randomized calibration
size changes.

## Protocol

For every calibration fraction and seed:

1. split the 54,000 randomized ratings into calibration and evaluation;
2. estimate the randomized marginal rating distribution using calibration only;
3. estimate rating-dependent Naive-Bayes observation propensities;
4. compute the self-normalized inverse-weighted observational mean;
5. compare it with the untouched randomized evaluation mean;
6. retain inverse-weight diagnostics.

Defaults are calibration fractions 1%, 2.5%, 5%, 10%, and 20%, with seeds
0 through 9.

## Reported metrics

Each fraction reports mean calibration size, mean absolute
calibration-to-evaluation gap, across-seed gap variability, mean absolute
Naive-Bayes bias, Naive-Bayes bias variability, mean/min/max apparent bias
reduction, and mean max-weight/p99-weight/ESS diagnostics.

The experiment does not assume monotonic improvement with calibration size.
It reports the empirical sensitivity curve.

## Interpretation boundary

This is an MNAR rating-exposure calibration experiment, not contextual-bandit
OPE. Randomized ratings supply identifying information about the marginal
rating distribution. The observational data alone do not nonparametrically
identify the missingness mechanism.
