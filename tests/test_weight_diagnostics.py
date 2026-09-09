import numpy as np
import pytest

from policyreclab.diagnostics import effective_sample_size, summarize_weights


def test_equal_weights_have_full_effective_sample_size() -> None:
    weights = np.ones(10)
    assert effective_sample_size(weights) == pytest.approx(10.0)

    diagnostics = summarize_weights(weights)
    assert diagnostics.effective_sample_fraction == pytest.approx(1.0)
    assert diagnostics.maximum == pytest.approx(1.0)


def test_concentrated_weights_reduce_effective_sample_size() -> None:
    balanced = np.ones(10)
    concentrated = np.array([10.0] + [0.0] * 9)

    assert effective_sample_size(concentrated) == pytest.approx(1.0)
    assert effective_sample_size(concentrated) < effective_sample_size(balanced)


def test_weight_validation() -> None:
    with pytest.raises(ValueError):
        effective_sample_size(np.array([]))
    with pytest.raises(ValueError):
        effective_sample_size(np.array([1.0, -1.0]))
