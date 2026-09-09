# v1.2 — Open Bandit Dataset Replication

## Why this milestone matters

PolicyRecLab now leaves the exact-truth synthetic setting and runs the OPE
framework on real production contextual-bandit logs.

The Open Bandit Dataset (OBD) was collected in an A/B test on ZOZOTOWN using
two behavior policies:

- Uniform Random,
- Bernoulli Thompson Sampling (BTS).

Each impression contains the selected item, recommendation position, click
outcome, user-side features, and a logged behavior-policy action probability.

## What counts as "ground truth" here?

Nothing in OBD gives PolicyRecLab exact counterfactual truth of the kind
available in the synthetic simulator.

Instead, OBD gives a valuable real-world validation protocol because the two
policies were actually deployed.

For v1.2 we use:

\[
\text{BTS logs}
\rightarrow
\widehat V_{\mathrm{IPS}}(\pi_{\mathrm{Random}})
\]

and compare that estimate with:

\[
\text{empirical CTR from the separately collected Random logs}.
\]

The Random CTR is an **on-policy empirical reference**.

It is not exact counterfactual ground truth.

## Why start with BTS -> Random?

Uniform Random has a simple known evaluation probability at each position:

\[
\pi_{\mathrm{Random}}(a\mid x,\ell)=1/K,
\]

where \(K\) is the number of available item actions for the campaign.

That lets PolicyRecLab evaluate Random from BTS logs using only:

- observed reward,
- logged BTS propensity,
- Random target probability on the logged action.

No dense \(n \times K\) policy matrix is necessary.

The reverse Random -> BTS direction requires reconstructing BTS action
probabilities from the production policy parameters. Open Bandit Pipeline
supports that with Monte Carlo action-distribution computation, and it is a
natural extension after this first real-data path is validated.

## Estimator

For impression \(i\),

\[
w_i =
\frac{\pi_e(a_i\mid x_i,\ell_i)}
     {\pi_b(a_i\mid x_i,\ell_i)}
\]

and

\[
\widehat V_{\mathrm{IPS}}
=
\frac{1}{n}\sum_i w_i r_i.
\]

v1.2 adds a logged-action IPS interface so real logs do not need to be expanded
into a dense context-action probability matrix.

## Propensity interpretation

PolicyRecLab treats the OBD propensity field as the **supplied logged behavior
propensity**.

The OBD/OBP documentation describes these behavior-policy action choice
probabilities as being produced from the production policy parameters, with
Monte Carlo computation used for the deployed Bernoulli-TS probabilities.

Therefore the project should not casually describe every recorded BTS
propensity as an analytically exact probability.

## Position

OBD has three recommendation positions.

The supplied propensity is the probability of the selected item at its recorded
position. v1.2 preserves the position column and uses the per-impression logged
propensity.

This remains a one-step, slot-level contextual-bandit OPE analysis. It should
not be generalized into arbitrary slate-level causal identification.

## Files expected locally

The runner expects the official directory layout:

```text
open_bandit_dataset/
├── bts/
│   └── all/
│       └── all.csv
└── random/
    └── all/
        └── all.csv
```

The same structure works for `men` and `women`.

Example:

```bash
python scripts/run_obd_bts_to_random.py \
  --data-root data/open_bandit_dataset \
  --campaign all
```

For a partial or custom extract that does not contain the maximum item id, pass
the known campaign action count explicitly with `--n-actions`.

## What v1.2 reports

The real-data experiment reports:

- number of BTS behavior impressions,
- number of Random reference impressions,
- IPS estimate of Random from BTS logs,
- separate Random on-policy empirical CTR,
- signed and relative error,
- max and p99 importance weights,
- ESS and ESS fraction as weight-concentration heuristics.

The error is an empirical replication discrepancy, not error relative to exact
truth.

## Scientific interpretation

Agreement supports the practical usefulness of the estimator/protocol under
this real logging setup.

Disagreement does not automatically imply that IPS is "wrong." It can arise
from finite-sample variance, weak overlap, differences between the two logged
samples, propensity approximation, or violations of the simplifying OPE model.

Synthetic experiments remain necessary because only there does PolicyRecLab
know the exact finite-population policy value.

Real and synthetic validation answer complementary questions.
