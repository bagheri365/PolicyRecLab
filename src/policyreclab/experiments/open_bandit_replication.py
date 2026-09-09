from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from policyreclab.datasets import load_open_bandit_csv
from policyreclab.diagnostics import WeightDiagnostics, summarize_weights
from policyreclab.estimators import estimate_logged_action_ips


@dataclass(frozen=True)
class OpenBanditReplicationResult:
    campaign: str
    behavior_policy: str
    target_policy: str
    n_behavior_rounds: int
    n_reference_rounds: int
    n_actions: int
    ips_policy_value: float
    on_policy_reference_ctr: float
    absolute_error: float
    relative_error: float
    weight_diagnostics: WeightDiagnostics


def run_bts_to_random_replication(
    *,
    bts_csv: str | Path,
    random_csv: str | Path,
    campaign: str,
    n_actions: int | None = None,
) -> OpenBanditReplicationResult:
    """Evaluate Uniform Random from BTS logs and compare with Random logs.

    The Random-policy empirical CTR is an independent on-policy empirical
    reference, not exact counterfactual ground truth.
    """
    bts = load_open_bandit_csv(bts_csv, n_actions=n_actions)
    random = load_open_bandit_csv(random_csv, n_actions=n_actions)

    resolved_n_actions = max(bts.n_actions, random.n_actions)
    if n_actions is not None:
        resolved_n_actions = n_actions

    if np.max(bts.actions) >= resolved_n_actions:
        raise ValueError("BTS action id is outside the resolved action space")
    if np.max(random.actions) >= resolved_n_actions:
        raise ValueError("Random action id is outside the resolved action space")

    # Uniform Random chooses every item with equal probability at each
    # recommendation position.
    random_target_propensity = 1.0 / resolved_n_actions
    target_propensities = np.full(
        bts.n_rounds,
        random_target_propensity,
        dtype=np.float64,
    )

    estimate = estimate_logged_action_ips(
        rewards=bts.rewards,
        behavior_propensities=bts.behavior_propensities,
        target_propensities=target_propensities,
    )
    reference = random.empirical_ctr

    absolute_error = float(estimate.value - reference)
    relative_error = (
        float(absolute_error / reference)
        if reference != 0.0
        else float("nan")
    )

    return OpenBanditReplicationResult(
        campaign=campaign,
        behavior_policy="bts",
        target_policy="random",
        n_behavior_rounds=bts.n_rounds,
        n_reference_rounds=random.n_rounds,
        n_actions=resolved_n_actions,
        ips_policy_value=estimate.value,
        on_policy_reference_ctr=reference,
        absolute_error=absolute_error,
        relative_error=relative_error,
        weight_diagnostics=summarize_weights(estimate.importance_weights),
    )
