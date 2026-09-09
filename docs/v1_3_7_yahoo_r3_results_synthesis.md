# v1.3.7 — Yahoo R3 results synthesis

## Research question

How much randomized feedback is required to reliably correct the exposure bias
in Yahoo R3's self-selected rating data?

Yahoo R3 is used here as an MNAR rating-exposure benchmark. It is not a production contextual-bandit log, so its results should not be described as standard contextual-bandit off-policy evaluation.

## Benchmark scale

PolicyRecLab uses 311,704 self-selected observational ratings and 54,000
randomized ratings over 15,400 users and 1,000 songs on a 1–5 scale.

The randomized ratings provide empirical information about the target marginal
rating distribution. That information is not identified from the observational
ratings alone.

## v1.3.4 — Naive-Bayes MNAR correction

The observational mean is severely upward biased relative to held-out
randomized ratings:

| Quantity | Value |
|---|---:|
| Observational mean | 2.89199 |
| Held-out randomized reference | 1.81961 |
| Naive observational gap | +1.07238 |
| Item-frequency weighted mean | 2.74342 |
| Item-frequency gap reduction | 13.85% |
| Naive-Bayes weighted mean | 1.81072 |
| Naive-Bayes gap reduction | 99.17% |

The estimated observation propensities increase sharply with rating:

| Rating | Estimated observation propensity |
|---|---:|
| 1 | 0.0120 |
| 2 | 0.0106 |
| 3 | 0.0222 |
| 4 | 0.0516 |
| 5 | 0.2039 |

This explains the upward observational bias: five-star outcomes are far more
likely to appear in the self-selected sample than one- or two-star outcomes.

### Structural identity

The near-perfect Naive-Bayes correction must not be interpreted as the
observational data alone learning the missingness mechanism.

With

```math
\hat p_r =
\frac{\hat P(Y=r\mid O=1)\hat P(O=1)}
{\hat P_{\mathrm{cal}}(Y=r)}
```

self-normalized inverse weighting reconstructs the rating distribution
estimated from the randomized calibration subset. Without smoothing or active
clipping,

```math
\hat\mu_{\mathrm{NB-IPW}}
=
\sum_r r\,\hat P_{\mathrm{cal}}(Y=r)
=
\hat\mu_{\mathrm{cal}}
```

In the real Yahoo run, the calibration mean is 1.80852 and the Naive-Bayes
weighted mean is 1.81072, a difference of only 0.00220. The scientific test is
therefore whether the randomized calibration subset generalizes to the
untouched randomized evaluation subset.

## v1.3.5 — Calibration-size sensitivity

Across 10 randomized splits, increasing the randomized calibration fraction
reduces both average error and split-to-split variability.

| Calibration | Mean n calibration | Mean abs NB bias | Mean bias reduction |
|---|---:|---:|---:|
| 1% | 540 | 0.03413 | 96.82% |
| 2.5% | 1,350 | 0.01810 | 98.31% |
| 5% | 2,700 | 0.01424 | 98.67% |
| 10% | 5,400 | 0.01143 | 98.94% |
| 20% | 10,800 | 0.00775 | 99.28% |

The largest efficiency gain occurs early, especially from 1% to 2.5%.
However, mean performance alone understates tail risk.

Weight concentration remains nearly unchanged across calibration fractions,
with ESS fractions around 66–67%. The sensitivity curve is therefore driven
primarily by uncertainty in estimating the randomized marginal rating
distribution rather than by inverse-weight instability.

## v1.3.6 — Calibration reliability

Fifty randomized splits per calibration fraction reveal a stronger practical
distinction between 5% and 10% calibration.

| Calibration | P(|bias| ≤ .01) | P(|bias| ≤ .02) | 95th pct abs bias | P(≥98% reduction) |
|---|---:|---:|---:|---:|
| 1% | 18% | 28% | 0.0902 | 32% |
| 2.5% | 22% | 58% | 0.0554 | 60% |
| 5% | 44% | 72% | 0.0413 | 72% |
| 10% | 66% | 94% | 0.0220 | 94% |
| 20% | 74% | 96% | 0.0170 | 98% |

The median bias moves toward zero as calibration size grows:

```math
0.0279
\rightarrow 0.0125
\rightarrow 0.0078
\rightarrow 0.0026
\rightarrow -0.0005
```

## Retained conclusion

The Yahoo R3 branch supports four conclusions:

1. Self-selected ratings exhibit severe MNAR exposure bias.
2. Simple item-frequency weighting is inadequate.
3. Rating-dependent Naive-Bayes weighting can recover the randomized rating
   distribution when supplied with randomized calibration data, but this
   success is structurally tied to that calibration distribution.
4. Average error improves smoothly with calibration size, while empirical
   reliability improves especially between 5% and 10%.

For this benchmark, 10% randomized calibration is a defensible empirical operating point: 94% of randomized splits achieve absolute bias below 0.02, and 94% achieve at least 98% apparent gap reduction. This is not a universal threshold and should not be presented as one.

## Identification boundary

The observational Yahoo R3 ratings alone do not nonparametrically identify the MNAR exposure mechanism. Randomized calibration data supply identifying information about the target rating distribution.

The across-seed quantiles and success frequencies reported in v1.3.6 are
empirical split-sensitivity summaries, not confidence intervals for a
population parameter.
