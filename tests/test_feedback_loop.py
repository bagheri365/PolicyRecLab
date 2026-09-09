from policyreclab.experiments import run_feedback_loop_comparison
from policyreclab.feedback import run_feedback_loop
from policyreclab.simulation import SyntheticBanditEnvironment


def test_feedback_loop_records_all_deployments() -> None:
    env = SyntheticBanditEnvironment.generate(
        n_users=30,
        n_items=6,
        latent_dim=3,
        seed=5,
    )
    result = run_feedback_loop(
        env,
        n_deployments=5,
        n_rounds_per_deployment=800,
        epsilon=0.1,
        base_seed=9_000,
    )

    assert len(result.rounds) == 5
    assert result.rounds[0].round_index == 0
    assert result.rounds[-1].round_index == 4


def test_lower_exploration_creates_weaker_future_uniform_support() -> None:
    study = run_feedback_loop_comparison(
        exploratory_epsilon=0.25,
        exploitative_epsilon=0.01,
        n_deployments=6,
        n_rounds_per_deployment=2_000,
        n_users=80,
        n_items=8,
        latent_dim=4,
        environment_seed=17,
        base_seed=12_000,
    )

    exploratory = study.exploratory.rounds[-1]
    exploitative = study.exploitative.rounds[-1]

    assert (
        exploitative.uniform_target_min_support
        < exploratory.uniform_target_min_support
    )


def test_lower_exploration_concentrates_exposure_more_in_controlled_setup() -> None:
    study = run_feedback_loop_comparison(
        exploratory_epsilon=0.25,
        exploitative_epsilon=0.01,
        n_deployments=7,
        n_rounds_per_deployment=3_000,
        n_users=90,
        n_items=8,
        latent_dim=4,
        environment_seed=23,
        base_seed=18_000,
    )

    exploratory = study.exploratory.rounds[-1]
    exploitative = study.exploitative.rounds[-1]

    assert exploitative.exposure_entropy < exploratory.exposure_entropy
    assert exploitative.exposure_gini > exploratory.exposure_gini
    assert exploitative.maximum_action_share > exploratory.maximum_action_share


def test_feedback_loop_does_not_claim_monotonic_value_improvement() -> None:
    env = SyntheticBanditEnvironment.generate(
        n_users=50,
        n_items=6,
        latent_dim=3,
        seed=29,
    )
    result = run_feedback_loop(
        env,
        n_deployments=6,
        n_rounds_per_deployment=1_500,
        epsilon=0.05,
        base_seed=22_000,
    )

    values = [round_.policy_value for round_ in result.rounds]

    # The contract is only that the loop runs and records policy values. The
    # learned policy may improve or worsen between finite noisy deployments.
    assert len(values) == 6
    assert all(0.0 <= value <= 1.0 for value in values)
