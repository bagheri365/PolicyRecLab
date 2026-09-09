# v1.2.1 — OBD Overlap and Estimator Sensitivity

The first full Open Bandit Dataset run showed that close IPS point estimates can coexist with extreme importance-weight concentration.

This follow-up records IPS, SNIPS, clipped IPS, p99/p99.9/max weights, ESS, and top-tail contribution shares for `all`, `men`, and `women`.

Run:

```bash
python scripts/run_obd_sensitivity.py --data-root data/open_bandit_dataset
```

It writes `summary.csv`, `metrics.json`, and `RESULTS.md` under `artifacts/v1_2_open_bandit_dataset/`.

Clipping is a bias-variance sensitivity analysis, not a free improvement. SNIPS is generally finite-sample biased and is not assumed to dominate IPS. ESS is a descriptive weight-concentration heuristic, not a literal inferential sample size. The separately deployed Random CTR remains an empirical on-policy reference, not exact counterfactual truth.
