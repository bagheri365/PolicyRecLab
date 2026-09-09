from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SupportReport:
    """Contextual positivity/support diagnostics for two policies.

    A support violation occurs at context-action pair ``(x, a)`` when the
    target policy assigns positive probability but the behavior policy assigns
    zero probability:

        pi_e(a | x) > 0  and  pi_b(a | x) = 0.

    In the standard nonparametric one-step contextual-bandit setting, positive
    target mass on such pairs means the full target policy value is not
    identified from the behavior-policy logs alone.
    """

    has_full_support: bool
    n_contexts: int
    n_actions: int
    n_violating_contexts: int
    n_violating_pairs: int
    target_mass_off_support: float
    max_context_target_mass_off_support: float
    min_positive_behavior_probability_on_target_support: float | None

    @property
    def violating_context_fraction(self) -> float:
        return self.n_violating_contexts / self.n_contexts


def _validate_policy_matrix(
    probabilities: FloatArray,
    *,
    name: str,
) -> FloatArray:
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.ndim != 2:
        raise ValueError(f"{name} must be a 2D probability matrix")
    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(probabilities < 0.0):
        raise ValueError(f"{name} probabilities must be non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError(f"each row of {name} must sum to 1")

    return probabilities


def analyze_support(
    behavior_probabilities: FloatArray,
    target_probabilities: FloatArray,
    *,
    zero_tolerance: float = 0.0,
) -> SupportReport:
    """Analyze exact contextual support for a target policy.

    Parameters
    ----------
    behavior_probabilities:
        Matrix with entries ``pi_b(a | x)``.
    target_probabilities:
        Matrix with entries ``pi_e(a | x)``.
    zero_tolerance:
        Optional numerical threshold. Behavior probabilities less than or equal
        to this value are treated as zero for the purpose of support checks.

    Notes
    -----
    This function intentionally separates exact support failure from weak
    overlap. v0.2 asks only whether target-policy mass lies completely inside
    behavior-policy support. Very small but positive behavior probabilities are
    still classified as supported here; their variance consequences are studied
    in v0.4.
    """
    if zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be non-negative")

    behavior = _validate_policy_matrix(
        behavior_probabilities,
        name="behavior_probabilities",
    )
    target = _validate_policy_matrix(
        target_probabilities,
        name="target_probabilities",
    )

    if behavior.shape != target.shape:
        raise ValueError(
            "behavior_probabilities and target_probabilities must have the same shape"
        )

    behavior_zero = behavior <= zero_tolerance
    target_positive = target > 0.0
    violation_mask = behavior_zero & target_positive

    off_support_mass_by_context = np.sum(
        np.where(violation_mask, target, 0.0),
        axis=1,
    )

    on_target_support = target_positive & ~behavior_zero
    positive_behavior_values = behavior[on_target_support]
    min_positive_behavior = (
        float(np.min(positive_behavior_values))
        if positive_behavior_values.size
        else None
    )

    violating_contexts = np.any(violation_mask, axis=1)

    return SupportReport(
        has_full_support=not bool(np.any(violation_mask)),
        n_contexts=int(behavior.shape[0]),
        n_actions=int(behavior.shape[1]),
        n_violating_contexts=int(np.sum(violating_contexts)),
        n_violating_pairs=int(np.sum(violation_mask)),
        target_mass_off_support=float(np.mean(off_support_mass_by_context)),
        max_context_target_mass_off_support=float(
            np.max(off_support_mass_by_context)
        ),
        min_positive_behavior_probability_on_target_support=min_positive_behavior,
    )
