# PolicyRecLab reproducibility guide

This guide is the release-time path for reproducing PolicyRecLab from a fresh
clone. The experimental program is frozen at **v1.3.7**; reproduction work
should not require adding new experiments.

## 1. Fresh clone

```bash
git clone <PolicyRecLab repository URL>
cd PolicyRecLab

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
pytest
```

A clean release candidate should pass the complete test suite before any
real-data files are added.

## 2. Reproducibility layers

PolicyRecLab has two reproduction layers.

### Synthetic experiments

Synthetic experiments are self-contained and expose finite-population exact
policy value. They are the preferred first reproduction target because they do
not depend on external dataset access.

### Real-data experiments

OBD, Coat, and Yahoo R3 require datasets obtained separately. Raw external
datasets are local inputs and should not be committed merely to make a run
convenient.

The benchmarks also have different scientific roles:

- **Open Bandit Dataset:** one-step production contextual-bandit logs with
  supplied logging propensities and an independent Random deployment.
- **Coat:** MNAR self-selected ratings plus randomized ratings.
- **Yahoo R3:** MNAR self-selected ratings plus randomized ratings.

Coat and Yahoo R3 must not be described as production contextual-bandit logs.

## 3. Available experiment scripts

The repository currently contains these runnable scripts:

- `python scripts/run_coat_debiasing.py`
- `python scripts/run_coat_exposure_diagnostics.py`
- `python scripts/run_coat_exposure_model.py`
- `python scripts/run_coat_feature_propensity.py`
- `python scripts/run_obd_bts_to_random.py`
- `python scripts/run_obd_sensitivity.py`
- `python scripts/run_yahoo_r3_calibration_reliability.py`
- `python scripts/run_yahoo_r3_calibration_sensitivity.py`
- `python scripts/run_yahoo_r3_debiasing.py`
- `python scripts/run_yahoo_r3_naive_bayes.py`

Use `python scripts/<name>.py --help` when a script exposes command-line
arguments. Dataset-dependent scripts expect the corresponding local data
layout documented by that milestone.

## 4. Open Bandit Dataset

Use the official Open Bandit Dataset / OBP distribution and keep the raw data
outside version control unless its redistribution terms are explicitly
satisfied.

The v1.2/v1.2.1 experiments evaluate Uniform Random from BTS logs using the
propensity information supplied with the benchmark and compare against the
independent Random-policy deployment as an **empirical reference**, not exact
ground truth.

Do not rewrite the supplied propensities as analytically exact propensities in
the final documentation.

Relevant milestone documentation:

- `docs/v1_2_open_bandit_dataset.md` when present;
- `docs/v1_2_1_obd_sensitivity.md` when present;
- the OBD section of `docs/FINAL_REPORT.md`.

## 5. Coat

Expected local layout:

```text
data/coat/
├── README.txt
├── propensities.ascii
├── test.ascii
├── train.ascii
└── user_item_features/
    ├── item_features.ascii
    ├── item_features_map.txt
    ├── user_features.ascii
    └── user_features_map.txt
```

The observational and randomized matrices serve different roles. The
feature-aware propensity experiment should use the released user/item features
rather than treating item popularity alone as a sufficient exposure model.

Raw Coat files should remain local unless their redistribution terms are
explicitly checked.

## 6. Yahoo R3

The project uses a local Yahoo R3 mirror layout:

```text
data/yahoo_r3/
├── random.txt
├── sampling_data.txt
└── user.txt
```

For the benchmark implemented here:

- `random.txt` is the 54,000-row randomized-rating file;
- `user.txt` is the 311,704-row self-selected observational-rating file;
- `sampling_data.txt` is not the full observational benchmark.

The locally used mirror has uncertain redistribution provenance/licensing.
**Do not commit or redistribute these raw Yahoo R3 files through this
repository.**

The Naive-Bayes experiment uses randomized ratings for calibration. Its
success is therefore not observational-only identification.

## 7. Generated artifacts

Generated experiment outputs are reproducible products, not source code.
The repository's ignore policy should be checked before release so generated
artifacts do not accidentally enter version control.

Do not change the artifact-tracking policy during release preparation without
an explicit decision.

## 8. Verification matrix

Before release, verify all of the following:

| Check | Command / evidence |
|---|---|
| Environment installs | `python -m pip install -e .` |
| Full tests | `pytest` |
| Working tree clean | `git status` |
| Synthetic experiments | run the relevant scripts without external data |
| OBD | run only after local OBD data are available |
| Coat | run only after the expected local Coat layout is available |
| Yahoo R3 | run only after the expected local Yahoo layout is available |
| Final narrative | inspect `README.md` and `docs/FINAL_REPORT.md` |
| Documentation map | inspect `docs/DOCUMENTATION_INDEX.md` |

## 9. Scientific release checks

A release is not reproducible merely because the code executes. Confirm that:

- the estimand is stated;
- support/positivity assumptions are explicit;
- propensity provenance is stated;
- OBD empirical references are not called exact truth;
- Coat/Yahoo are not mislabeled as contextual-bandit logs;
- ESS is described as a concentration diagnostic;
- rejected experiments remain visible;
- Yahoo calibration dependence is explicit;
- target-policy selection and final evaluation are distinguished.

## 10. Release procedure

From a clean checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
pytest
git status
```

Then review:

```bash
git log --oneline -10
git diff --check
```

The release commit should have a clean working tree and all tests passing.

Tagging is a separate explicit decision. The internal research milestones
(v0.x through v1.3.7) do not require the public release tag to use the same
numbering scheme.
