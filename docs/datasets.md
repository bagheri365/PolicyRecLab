# Datasets

## Synthetic environment

The synthetic finite-context environment remains PolicyRecLab's primary
identification and estimator-validation instrument because exact
finite-population policy values are available.

## Open Bandit Dataset

Open Bandit Dataset is the first real logged-policy benchmark integrated into
PolicyRecLab.

Official release:
https://research.zozo.com/data.html

The full release contains Random and Bernoulli Thompson Sampling logs for the
`all`, `men`, and `women` campaigns.

PolicyRecLab does not vendor the dataset in Git.

Place downloaded data under:

```text
data/open_bandit_dataset/
```

Then run the v1.2 replication script described in
`docs/v1_2_open_bandit_dataset.md`.

The separate on-policy log of the evaluation policy is treated as an empirical
reference, not exact counterfactual truth.
