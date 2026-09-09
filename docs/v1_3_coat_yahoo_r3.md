# v1.3 — Coat and Yahoo! R3 Debiasing Study

v1.3 deliberately changes benchmark type.

Open Bandit Dataset is a production contextual-bandit logging benchmark.
Coat and Yahoo! R3 are MNAR rating datasets with self-selected observational
ratings and separately collected randomly selected ratings. They should not be
described as if they contained production action propensities.

## Coat

Coat contains 290 users and 300 coats. Each participant rated 24 self-selected
coats and 16 randomly selected coats. The randomized portion is an empirical
reference for evaluating bias from self-selection.

Run:

```bash
python scripts/run_coat_debiasing.py --data-root data/coat
```

## Yahoo! R3

Yahoo! R3 contains ratings over 1,000 songs from 15,400 users. Its observational
portion contains user-selected ratings. For the first 5,400 users, exactly 10
randomly selected songs were rated, giving 54,000 randomized ratings.

Because Yahoo! Webscope access/distribution conditions may vary, PolicyRecLab
does not download or redistribute Yahoo! R3. Point the runner at locally
authorized copies:

```bash
python scripts/run_yahoo_r3_debiasing.py \
  --train /path/to/yahoo-r3-train.txt \
  --randomized /path/to/yahoo-r3-random.txt
```

## First v1.3 estimand

The initial benchmark intentionally uses a transparent population-level target:
the mean 1--5 rating under the randomized rating sample.

It compares:
1. naive mean of self-selected observational ratings,
2. a self-normalized inverse-observation weighted mean using an explicitly
   estimated item-frequency exposure model,
3. the separate randomized empirical mean.

The estimated propensity model is intentionally simple and must be labeled
`estimated_item_frequency`. It is not a known logging propensity. Failure to
match the randomized reference is evidence about the inadequacy of the
observational estimator/model, not proof that the randomized sample is exact
counterfactual truth.

This first step establishes loaders, provenance, and the naive MNAR gap. More
faithful learned propensity models and MF-IPS prediction experiments can build
on the same protocol.

## Scientific boundary

Do not merge these results with OBD as if they estimate the same object.

OBD asks whether a target recommendation policy can be evaluated from another
policy's logged actions.

Coat/Yahoo! R3 ask whether learning/evaluation from self-selected ratings can be
corrected toward performance measured on randomly elicited ratings.

The common theme is policy/exposure-generated observation bias; the data
generating processes and identification assumptions differ.
