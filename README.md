# PolicyRecLab

A research lab for off-policy evaluation, exploration, and policy-generated bias
in contextual-bandit recommendation systems.

## Current milestone: v0.0

The first milestone builds a finite synthetic contextual-bandit environment with
known conditional reward probabilities and verifies that Monte Carlo policy
rollouts converge to the exact finite-population policy value.

No off-policy estimator is introduced yet.

## Install

```bash
python -m pip install -e ".[dev]"
pytest
```

See:

- `docs/ASSUMPTIONS.md`
- `docs/ESTIMAND.md`
- `docs/v0_0_simulator_protocol.md`
