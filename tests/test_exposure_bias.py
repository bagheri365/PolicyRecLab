import pytest

from policyreclab.experiments import run_exposure_bias_experiment


def test_behavior_policy_changes_exposure_and_logged_reward_distribution() -> None:
    result = run_exposure_bias_experiment(
        n_users=250,
        n_items=20,
        latent_dim=6,
        n_rounds=40_000,
        environment_seed=17,
        log_seed=23,
    )

    # Under uniform logging, the context-wise greedy action is exposed about
    # 1 / n_items of the time. Under deterministic greedy logging, it is always
    # exposed.
    assert result.greedy_action_share_uniform_logs == pytest.approx(0.05, abs=0.01)
    assert result.greedy_action_share_greedy_logs == pytest.approx(1.0)

    # Each on-policy logged mean should track its own policy value.
    assert result.uniform_logged_mean == pytest.approx(
        result.uniform_true_value, abs=0.012
    )
    assert result.greedy_logged_mean == pytest.approx(
        result.greedy_true_value, abs=0.012
    )

    # Same reward environment, different exposure policy, different observed
    # reward distribution. This is the v0.1 phenomenon.
    assert result.greedy_logged_mean > result.uniform_logged_mean + 0.10
