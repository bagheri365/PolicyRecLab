from policyreclab.experiments import run_policy_selection_study


def test_same_sample_selection_creates_optimism() -> None:
    study = run_policy_selection_study(
        candidate_counts=(2, 8, 20),
        n_repetitions=50,
        n_users=120,
        n_items=8,
        latent_dim=4,
        n_rounds_selection=1_500,
        n_rounds_holdout=5_000,
        environment_seed=59,
        candidate_seed=61,
        base_log_seed=27_000,
    )
    by_count = {point.n_candidates: point for point in study.points}

    small = by_count[2]
    large = by_count[20]

    # Selecting the largest noisy OPE estimate and reporting that same estimate
    # creates positive average optimism. More candidates create more opportunity
    # to select a favorable estimation error in this controlled experiment.
    # A very small candidate pool can still have near-zero or slightly
    # negative realized average optimism at finite repetition count. The
    # controlled claim is that searching the larger pool creates materially
    # more optimism, not that every finite small-pool realization is positive.
    assert large.mean_selection_optimism > small.mean_selection_optimism
    assert large.mean_selection_optimism > 0.001


def test_fresh_holdout_evaluates_selected_policy_without_selection_optimism() -> None:
    study = run_policy_selection_study(
        candidate_counts=(20,),
        n_repetitions=60,
        n_users=120,
        n_items=8,
        latent_dim=4,
        n_rounds_selection=1_500,
        n_rounds_holdout=6_000,
        environment_seed=67,
        candidate_seed=71,
        base_log_seed=31_000,
    )
    point = study.points[0]

    # The holdout is generated after the policy is selected and is never used
    # to re-select. Its average error should therefore remain near zero.
    assert abs(point.mean_holdout_error) < 0.015
    assert (
        abs(point.mean_holdout_error)
        < point.mean_selection_optimism * 0.6
    )


def test_policy_selection_can_choose_a_suboptimal_true_policy() -> None:
    study = run_policy_selection_study(
        candidate_counts=(12,),
        n_repetitions=40,
        n_users=100,
        n_items=8,
        latent_dim=4,
        n_rounds_selection=1_000,
        n_rounds_holdout=3_000,
        environment_seed=73,
        candidate_seed=79,
        base_log_seed=35_000,
    )

    assert study.points[0].mean_selected_regret > 0.0
