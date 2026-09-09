# v1.3.4 — Yahoo R3 Naive-Bayes MNAR Benchmark

PolicyRecLab now extends the real-data MNAR branch from Coat to Yahoo R3.

Yahoo R3 is **not** treated as contextual-bandit logging. Its observational
training ratings are self-selected, while its separate test data were elicited
by asking a subset of users to rate randomly selected songs. The purpose of
this milestone is to measure and partially correct observation bias.

## Dataset protocol

The standard Yahoo R3 release uses:

- `ydata-ymusic-rating-study-v1_0-train.txt`
- `ydata-ymusic-rating-study-v1_0-test.txt`

with rows of the form:

```text
<user_id> <song_id> <rating>
```

The benchmark contains 15,400 users and 1,000 songs. The self-selected
training set contains more than 300,000 ratings. The randomized test subset
contains 5,400 users with 10 randomly selected songs per user, i.e. 54,000
ratings.

PolicyRecLab never downloads or redistributes Yahoo R3. Supply locally
authorized files.


A commonly mirrored zero-based CSV layout is also supported automatically:

- `user.txt` — 311,704 self-selected observational ratings;
- `random.txt` — 54,000 randomized ratings.

`sampling_data.txt` is not used by this benchmark because it is a separate
derived/sample file rather than the complete observational rating log.

## Naive-Bayes propensity model

Following Schnabel et al. (2016), assume observation depends on the latent
rating category and tie parameters across users/items:

\[
P(O=1 \mid Y=r)
=
\frac{
P(Y=r \mid O=1) P(O=1)
}{
P(Y=r)
}.
\]

The terms are estimated as follows:

- `P(Y=r | O=1)`: empirical rating distribution in self-selected training data;
- `P(O=1)`: observed-training density over the 15,400 × 1,000 matrix;
- `P(Y=r)`: rating distribution from a small randomized calibration sample.

The paper reserves 5% of randomized Yahoo ratings to estimate the marginal
rating distribution and evaluates on the remaining 95%. PolicyRecLab follows
that separation with a deterministic seeded split. Laplace smoothing is used
for the randomized calibration rating distribution.

## Evaluation

The randomized calibration subset is used only for propensity estimation.
The untouched 95% randomized subset is the empirical reference for the final
mean-rating comparison.

Report:

1. naive observational mean,
2. item-frequency self-normalized inverse-observation mean,
3. Naive-Bayes self-normalized inverse-observation mean,
4. absolute discrepancy from the randomized evaluation reference,
5. percentage reduction in the naive discrepancy,
6. per-rating propensities,
7. maximum/p99 inverse weights and ESS fraction.

This is an MNAR missingness/debiasing experiment, not contextual-bandit OPE.

## Run

If your files use the standard names:

```bash
python scripts/run_yahoo_r3_naive_bayes.py \
  --data-root data/yahoo_r3
```

If they live elsewhere:

```bash
python scripts/run_yahoo_r3_naive_bayes.py \
  --train /path/to/ydata-ymusic-rating-study-v1_0-train.txt \
  --randomized /path/to/ydata-ymusic-rating-study-v1_0-test.txt
```

## Interpretation boundary

The Naive-Bayes propensity model makes a strong simplification: observation
probability is modeled only through the rating category, with parameters tied
across users and songs. A successful correction is evidence that rating-level
selection matters; it is not proof that the MNAR mechanism is fully identified.

Do not tune the 5% split, smoothing, clipping floor, or other choices against
the 95% randomized reference.


## Structural identity and interpretation

This benchmark has an important algebraic property. With the rating-only
Naive-Bayes propensity

\[
\hat p_r =
\frac{\hat P(Y=r\mid O=1)\hat P(O=1)}
{\hat P_{\mathrm{cal}}(Y=r)},
\]

self-normalized inverse weighting of the observational ratings reconstructs
the randomized calibration rating distribution. Without smoothing or active
propensity clipping,

\[
\hat\mu_{\mathrm{NB-IPW}}
=
\sum_r r\,\hat P_{\mathrm{cal}}(Y=r)
=
\hat\mu_{\mathrm{cal}}.
\]

Therefore, near agreement between the Naive-Bayes weighted mean and the
held-out randomized evaluation mean should not be described as the
observational log independently recovering the target mean. The meaningful
validation is whether the small randomized calibration sample generalizes to
the untouched randomized evaluation sample.

PolicyRecLab reports the calibration mean, its gap to the evaluation mean, and
the Naive-Bayes-minus-calibration difference explicitly to make this
interpretation auditable. Laplace smoothing can make the equality approximate
rather than exact.
