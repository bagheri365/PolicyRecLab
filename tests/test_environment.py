import numpy as np
import pytest

from policyreclab.simulation import SyntheticBanditEnvironment


def test_generate_is_reproducible() -> None:
    a = SyntheticBanditEnvironment.generate(seed=7)
    b = SyntheticBanditEnvironment.generate(seed=7)

    np.testing.assert_array_equal(a.user_features, b.user_features)
    np.testing.assert_array_equal(a.item_features, b.item_features)
    np.testing.assert_array_equal(a.item_bias, b.item_bias)
    np.testing.assert_array_equal(a.reward_probabilities, b.reward_probabilities)


def test_reward_probabilities_are_valid() -> None:
    env = SyntheticBanditEnvironment.generate(
        n_users=50,
        n_items=20,
        latent_dim=4,
        seed=1,
    )

    assert env.reward_probabilities.shape == (50, 20)
    assert np.all(env.reward_probabilities > 0.0)
    assert np.all(env.reward_probabilities < 1.0)


@pytest.mark.parametrize(
    ("n_users", "n_items", "latent_dim"),
    [(0, 10, 3), (10, 0, 3), (10, 10, 0)],
)
def test_invalid_dimensions_fail(
    n_users: int,
    n_items: int,
    latent_dim: int,
) -> None:
    with pytest.raises(ValueError):
        SyntheticBanditEnvironment.generate(
            n_users=n_users,
            n_items=n_items,
            latent_dim=latent_dim,
        )
