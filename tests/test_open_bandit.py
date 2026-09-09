from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from policyreclab.datasets import load_open_bandit_csv
from policyreclab.estimators import estimate_logged_action_ips
from policyreclab.experiments import run_bts_to_random_replication


def _write_obd_csv(
    path: Path,
    rows: list[tuple[int, int, int, float]],
    *,
    propensity_name: str = "propensity_score",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item_id", "position", "click", propensity_name],
        )
        writer.writeheader()
        for item_id, position, click, propensity in rows:
            writer.writerow(
                {
                    "item_id": item_id,
                    "position": position,
                    "click": click,
                    propensity_name: propensity,
                }
            )


def test_loader_accepts_current_and_legacy_propensity_names(tmp_path: Path) -> None:
    current = tmp_path / "current.csv"
    legacy = tmp_path / "legacy.csv"

    rows = [(0, 1, 1, 0.25), (3, 2, 0, 0.5)]
    _write_obd_csv(current, rows, propensity_name="propensity_score")
    _write_obd_csv(legacy, rows, propensity_name="action_prob")

    current_log = load_open_bandit_csv(current)
    legacy_log = load_open_bandit_csv(legacy)

    assert current_log.n_actions == 4
    assert legacy_log.n_rounds == 2
    assert np.allclose(
        current_log.behavior_propensities,
        legacy_log.behavior_propensities,
    )


def test_logged_action_ips_matches_hand_calculation() -> None:
    estimate = estimate_logged_action_ips(
        rewards=np.array([1.0, 0.0, 1.0]),
        behavior_propensities=np.array([0.5, 0.25, 0.5]),
        target_propensities=np.array([0.25, 0.25, 0.25]),
    )

    expected = np.mean([0.5, 0.0, 0.5])
    assert estimate.value == expected


def test_bts_to_random_replication_uses_separate_empirical_reference(
    tmp_path: Path,
) -> None:
    bts = tmp_path / "bts" / "all" / "all.csv"
    random = tmp_path / "random" / "all" / "all.csv"

    # Four actions. The Random target probability at each position is 1/4.
    _write_obd_csv(
        bts,
        [
            (0, 1, 1, 0.5),
            (1, 2, 0, 0.25),
            (2, 3, 1, 0.5),
            (3, 1, 0, 0.25),
        ],
    )
    _write_obd_csv(
        random,
        [
            (0, 1, 1, 0.25),
            (1, 2, 0, 0.25),
            (2, 3, 1, 0.25),
            (3, 1, 1, 0.25),
        ],
    )

    result = run_bts_to_random_replication(
        bts_csv=bts,
        random_csv=random,
        campaign="all",
        n_actions=4,
    )

    # BTS-side IPS: mean([0.5, 0, 0.5, 0]) = 0.25.
    assert result.ips_policy_value == 0.25
    # Separate Random log CTR: 3/4.
    assert result.on_policy_reference_ctr == 0.75
    assert result.absolute_error == -0.5
    assert math.isclose(result.relative_error, -2.0 / 3.0)


def test_loader_rejects_nonpositive_logged_propensity(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    _write_obd_csv(path, [(0, 1, 1, 0.0)])

    try:
        load_open_bandit_csv(path)
    except ValueError as exc:
        assert "propensities" in str(exc)
    else:
        raise AssertionError("expected invalid propensity to be rejected")
