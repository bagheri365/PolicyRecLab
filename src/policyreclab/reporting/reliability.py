from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from policyreclab.diagnostics import analyze_support, summarize_weights
from policyreclab.estimators import estimate_ips
from policyreclab.logging import LoggedBanditData


FloatArray = NDArray[np.float64]

EstimatorName = Literal["ips"]
PropensitySource = Literal["known", "estimated", "unknown"]
LoggingProcess = Literal["iid_static", "adaptive", "unknown"]
SelectionProcedure = Literal[
    "fixed_independently",
    "selected_on_separate_data",
    "selected_on_evaluation_data",
    "unknown",
]


@dataclass(frozen=True)
class ReliabilityReport:
    estimator: EstimatorName
    estimated_value: float
    identified_nonparametrically: bool
    full_contextual_support: bool
    minimum_behavior_probability_on_target_support: float
    maximum_importance_weight: float
    p99_importance_weight: float
    effective_sample_size: float
    effective_sample_fraction: float
    propensity_source: PropensitySource
    logging_process: LoggingProcess
    target_policy_selection: SelectionProcedure
    inference_method: str
    inference_validity_note: str
    reliability_warnings: tuple[str, ...]


def _inference_note(logging_process: LoggingProcess) -> str:
    if logging_process == "adaptive":
        return (
            "Adaptive logging detected or declared. Ordinary IID intervals "
            "should not be assumed valid across the adaptive sequence; use an "
            "adaptive/sequential inference method appropriate to the design."
        )
    if logging_process == "iid_static":
        return (
            "IID/static-policy interpretation declared. Any interval still "
            "depends on estimator-specific regularity assumptions and overlap."
        )
    return (
        "Logging-process structure is unknown. Inferential validity cannot be "
        "established from the report alone."
    )


def build_ips_reliability_report(
    logged_data: LoggedBanditData,
    *,
    behavior_probabilities: FloatArray,
    target_probabilities: FloatArray,
    propensity_source: PropensitySource = "known",
    logging_process: LoggingProcess = "iid_static",
    target_policy_selection: SelectionProcedure = "fixed_independently",
    inference_method: str = "none",
) -> ReliabilityReport:
    """Build an integrated reliability report for an IPS policy value."""
    behavior = np.asarray(behavior_probabilities, dtype=np.float64)
    target = np.asarray(target_probabilities, dtype=np.float64)

    support = analyze_support(behavior, target)
    warnings: list[str] = []

    if not support.has_full_support:
        warnings.append(
            "Target policy is not fully supported by the behavior policy; "
            "the target value is not nonparametrically identified by IPS."
        )

    if propensity_source == "unknown":
        warnings.append(
            "Behavior propensity source is unknown; IPS validity cannot be "
            "established."
        )
    elif propensity_source == "estimated":
        warnings.append(
            "Behavior propensities are estimated; propensity-model error must "
            "be included in the validity assessment."
        )

    if target_policy_selection == "selected_on_evaluation_data":
        warnings.append(
            "Target policy was selected on the same evaluation data; the "
            "reported value is vulnerable to selection optimism."
        )
    elif target_policy_selection == "unknown":
        warnings.append("Target-policy selection procedure is unknown.")

    if logging_process == "adaptive":
        warnings.append(
            "Logging is adaptive; ordinary IID inference is not automatically "
            "valid."
        )
    elif logging_process == "unknown":
        warnings.append("Logging-process structure is unknown.")

    min_behavior = float(
        support.min_positive_behavior_probability_on_target_support
    )

    if not support.has_full_support:
        return ReliabilityReport(
            estimator="ips",
            estimated_value=float("nan"),
            identified_nonparametrically=False,
            full_contextual_support=False,
            minimum_behavior_probability_on_target_support=min_behavior,
            maximum_importance_weight=float("inf"),
            p99_importance_weight=float("inf"),
            effective_sample_size=0.0,
            effective_sample_fraction=0.0,
            propensity_source=propensity_source,
            logging_process=logging_process,
            target_policy_selection=target_policy_selection,
            inference_method=inference_method,
            inference_validity_note=_inference_note(logging_process),
            reliability_warnings=tuple(warnings),
        )

    estimate = estimate_ips(
        logged_data,
        behavior_probabilities=behavior,
        target_probabilities=target,
    )
    weight_summary = summarize_weights(estimate.importance_weights)

    if weight_summary.maximum > 20.0:
        warnings.append(
            "Importance weights are highly concentrated; weak overlap may make "
            "the estimate unstable."
        )
    if weight_summary.effective_sample_fraction < 0.2:
        warnings.append(
            "ESS fraction is low. This is a descriptive weight-concentration "
            "warning, not a literal inferential sample-size calculation."
        )

    return ReliabilityReport(
        estimator="ips",
        estimated_value=float(estimate.value),
        identified_nonparametrically=True,
        full_contextual_support=True,
        minimum_behavior_probability_on_target_support=min_behavior,
        maximum_importance_weight=float(weight_summary.maximum),
        p99_importance_weight=float(weight_summary.p99),
        effective_sample_size=float(weight_summary.effective_sample_size),
        effective_sample_fraction=float(weight_summary.effective_sample_fraction),
        propensity_source=propensity_source,
        logging_process=logging_process,
        target_policy_selection=target_policy_selection,
        inference_method=inference_method,
        inference_validity_note=_inference_note(logging_process),
        reliability_warnings=tuple(warnings),
    )
