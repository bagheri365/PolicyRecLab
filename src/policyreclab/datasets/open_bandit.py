from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class OpenBanditLog:
    """Minimal OBD fields required for one-step OPE replication."""

    actions: IntArray
    positions: IntArray
    rewards: FloatArray
    behavior_propensities: FloatArray
    n_actions: int
    source_path: Path

    @property
    def n_rounds(self) -> int:
        return int(self.rewards.size)

    @property
    def empirical_ctr(self) -> float:
        return float(np.mean(self.rewards))


def _resolve_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        f"missing required column; expected one of {candidates}, got {fieldnames}"
    )


def load_open_bandit_csv(
    path: str | Path,
    *,
    n_actions: int | None = None,
) -> OpenBanditLog:
    """Load the OBD impression-level fields needed by PolicyRecLab.

    Both the current official ``propensity_score`` name and the older small-data
    ``action_prob`` name are accepted.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    actions: list[int] = []
    positions: list[int] = []
    rewards: list[float] = []
    propensities: list[float] = []

    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV must contain a header")

        fieldnames = list(reader.fieldnames)
        action_col = _resolve_column(fieldnames, ("item_id", "action"))
        position_col = _resolve_column(fieldnames, ("position",))
        reward_col = _resolve_column(fieldnames, ("click", "reward"))
        propensity_col = _resolve_column(
            fieldnames,
            ("propensity_score", "action_prob", "pscore"),
        )

        for row in reader:
            actions.append(int(row[action_col]))
            # Official raw OBD positions are 1,2,3. Preserve the raw label; OPE
            # here only needs the per-row logged propensity.
            positions.append(int(row[position_col]))
            rewards.append(float(row[reward_col]))
            propensities.append(float(row[propensity_col]))

    if not actions:
        raise ValueError("Open Bandit CSV must contain at least one row")

    action_array = np.asarray(actions, dtype=np.int64)
    position_array = np.asarray(positions, dtype=np.int64)
    reward_array = np.asarray(rewards, dtype=np.float64)
    propensity_array = np.asarray(propensities, dtype=np.float64)

    if np.any(action_array < 0):
        raise ValueError("item/action ids must be non-negative")
    if np.any((reward_array < 0.0) | (reward_array > 1.0)):
        raise ValueError("click/reward values must lie in [0, 1]")
    if not np.all(np.isfinite(propensity_array)):
        raise ValueError("behavior propensities must be finite")
    if np.any((propensity_array <= 0.0) | (propensity_array > 1.0)):
        raise ValueError("behavior propensities must lie in (0, 1]")

    inferred_n_actions = int(np.max(action_array)) + 1
    if n_actions is None:
        n_actions = inferred_n_actions
    elif n_actions < inferred_n_actions:
        raise ValueError(
            "n_actions is smaller than the largest observed action id plus one"
        )
    if n_actions <= 0:
        raise ValueError("n_actions must be positive")

    return OpenBanditLog(
        actions=action_array,
        positions=position_array,
        rewards=reward_array,
        behavior_propensities=propensity_array,
        n_actions=int(n_actions),
        source_path=source,
    )
