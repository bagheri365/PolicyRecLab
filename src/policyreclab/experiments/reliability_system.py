from __future__ import annotations

from dataclasses import dataclass

from policyreclab.logging import simulate_logged_bandit_data
from policyreclab.policies import EpsilonGreedyPolicy, UniformPolicy
from policyreclab.reporting import ReliabilityReport, build_ips_reliability_report
from policyreclab.simulation import SyntheticBanditEnvironment


@dataclass(frozen=True)
class ReliabilitySystemStudy:
    healthy_overlap: ReliabilityReport
    weak_overlap: ReliabilityReport
    zero_support: ReliabilityReport
    selected_on_same_data: ReliabilityReport
    adaptive_logging: ReliabilityReport


def run_reliability_system_study(
    *,
    n_users: int = 120,
    n_items: int = 10,
    latent_dim: int = 4,
    n_rounds: int = 6_000,
    environment_seed: int = 121,
    base_seed: int = 90_000,
) -> ReliabilitySystemStudy:
    env = SyntheticBanditEnvironment.generate(
        n_users=n_users,
        n_items=n_items,
        latent_dim=latent_dim,
        seed=environment_seed,
    )
    target = UniformPolicy().probabilities(
        n_contexts=env.n_users,
        n_actions=env.n_items,
    )

    healthy_behavior = EpsilonGreedyPolicy(0.5).probabilities(
        env.reward_probabilities
    )
    weak_behavior = EpsilonGreedyPolicy(0.02).probabilities(
        env.reward_probabilities
    )
    deterministic_behavior = EpsilonGreedyPolicy(0.0).probabilities(
        env.reward_probabilities
    )

    healthy_logs = simulate_logged_bandit_data(
        env, healthy_behavior, n_rounds=n_rounds, seed=base_seed
    )
    weak_logs = simulate_logged_bandit_data(
        env, weak_behavior, n_rounds=n_rounds, seed=base_seed + 1
    )
    deterministic_logs = simulate_logged_bandit_data(
        env, deterministic_behavior, n_rounds=n_rounds, seed=base_seed + 2
    )

    healthy = build_ips_reliability_report(
        healthy_logs,
        behavior_probabilities=healthy_behavior,
        target_probabilities=target,
        propensity_source="known",
        logging_process="iid_static",
        target_policy_selection="fixed_independently",
        inference_method="none",
    )
    weak = build_ips_reliability_report(
        weak_logs,
        behavior_probabilities=weak_behavior,
        target_probabilities=target,
        propensity_source="known",
        logging_process="iid_static",
        target_policy_selection="fixed_independently",
        inference_method="none",
    )
    zero_support = build_ips_reliability_report(
        deterministic_logs,
        behavior_probabilities=deterministic_behavior,
        target_probabilities=target,
        propensity_source="known",
        logging_process="iid_static",
        target_policy_selection="fixed_independently",
        inference_method="none",
    )
    selected_on_same_data = build_ips_reliability_report(
        healthy_logs,
        behavior_probabilities=healthy_behavior,
        target_probabilities=target,
        propensity_source="known",
        logging_process="iid_static",
        target_policy_selection="selected_on_evaluation_data",
        inference_method="none",
    )
    adaptive_logging = build_ips_reliability_report(
        healthy_logs,
        behavior_probabilities=healthy_behavior,
        target_probabilities=target,
        propensity_source="known",
        logging_process="adaptive",
        target_policy_selection="fixed_independently",
        inference_method="iid_normal_approximation",
    )

    return ReliabilitySystemStudy(
        healthy_overlap=healthy,
        weak_overlap=weak,
        zero_support=zero_support,
        selected_on_same_data=selected_on_same_data,
        adaptive_logging=adaptive_logging,
    )
