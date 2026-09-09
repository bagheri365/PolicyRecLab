# PolicyRecLab release checklist

## Scope

- [ ] Experimental scope remains frozen at v1.3.7.
- [ ] No new estimator/dataset was added during release preparation.
- [ ] Any post-freeze code change is documented as a correctness or
      reproducibility fix.

## Code

- [ ] Fresh virtual environment created successfully.
- [ ] `python -m pip install -e .` succeeds.
- [ ] Full `pytest` suite passes.
- [ ] `git diff --check` is clean.
- [ ] `git status` is clean after verification.

## Documentation

- [ ] README states the research question and central reliability rule.
- [ ] `docs/FINAL_REPORT.md` matches retained experimental conclusions.
- [ ] `docs/DOCUMENTATION_INDEX.md` marks D1/D2/D3 accurately.
- [ ] `docs/REPRODUCIBILITY.md` matches actual repository paths.
- [ ] Failed/rejected experiments remain visible.
- [ ] Internal documentation links have been checked.

## Scientific wording

- [ ] Identification, estimation, and inference remain distinct.
- [ ] Zero support is described as an identification failure.
- [ ] ESS is a weight-concentration diagnostic, not inferential sample size.
- [ ] OBD uses supplied propensity wording.
- [ ] OBD Random deployment is an empirical reference, not exact truth.
- [ ] Coat/Yahoo are described as MNAR rating-exposure benchmarks.
- [ ] Yahoo randomized-calibration dependence is explicit.
- [ ] The 10% Yahoo calibration operating point is benchmark-specific.

## External data

- [ ] Raw OBD data are not accidentally committed.
- [ ] Raw Coat data are not accidentally committed.
- [ ] Raw Yahoo R3 data are not committed or redistributed.
- [ ] Yahoo mirror provenance/licensing uncertainty is documented.
- [ ] Generated-artifact tracking policy has not changed accidentally.

## Release

- [ ] `CITATION.cff` reviewed if present.
- [ ] `pyproject.toml` package metadata/version reviewed.
- [ ] Repository description/topics reviewed.
- [ ] Final commit hash recorded.
- [ ] Release/tag naming decision made explicitly.
- [ ] Final tag created only after all checks above pass.
