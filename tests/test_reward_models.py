import numpy as np
import pytest

from policyreclab.logging import LoggedBanditData
from policyreclab.reward_models import GlobalMeanRewardModel
from policyreclab.simulation import SyntheticBanditEnvironment
from policyreclab.reward_models import OracleRewardModel


def test_oracle_reward_model_matches_environment_means() -> None:
    env = SyntheticBanditEnvironment.generate(
        n_users=8,
        n_items=5,
        latent_dim=3,
        seed=3,
    )
    predictions = OracleRewardModel.from_environment(env).predict_all()
    np.testing.assert_allclose(predictions, env.reward_probabilities)


def test_global_mean_model_ignores_context_and_action() -> None:
    logs = LoggedBanditData(
        context_indices=np.array([0, 1, 1]),
        actions=np.array([0, 1, 0]),
        rewards=np.array([1, 0, 1]),
        behavior_propensities=np.array([0.5, 0.5, 0.5]),
    )
    model = GlobalMeanRewardModel.fit(
        logs,
        n_contexts=2,
        n_actions=2,
    )

    np.testing.assert_allclose(
        model.predict_all(),
        np.full((2, 2), 2.0 / 3.0),
    )
    assert model.mean_reward == pytest.approx(2.0 / 3.0)
