import numpy as np
import pytest
from numpy.typing import NDArray

from bbstat.estimate import Estimate


@pytest.fixture(scope="module")
def samples() -> NDArray[np.floating]:
    return np.linspace(0, 1, 101)


def test_estimate(samples: NDArray[np.floating]) -> None:
    estimate = Estimate(key="key", samples=samples)
    assert estimate.n_samples == len(samples)
    np.testing.assert_allclose(estimate.samples, samples)


def test_estimate_mean(samples: NDArray[np.floating]) -> None:
    estimate = Estimate(key="key", samples=samples)
    actual = estimate.mean()
    assert isinstance(actual, float)
    np.testing.assert_allclose(actual, 0.5)


@pytest.mark.parametrize(
    "width, expected_lo, expected_hi",
    [
        pytest.param(0.5, 0.25, 0.75),
        pytest.param(0.85, 0.075, 0.925),
    ],
)
def test_estimate_ci(
    samples: NDArray[np.floating],
    width: float,
    expected_lo: float,
    expected_hi: float,
) -> None:
    estimate = Estimate(key="key", samples=samples)
    actual = estimate.ci(width=width)
    assert isinstance(actual, tuple)
    assert len(actual) == 2
    actual_lo, actual_hi = actual
    assert isinstance(actual_lo, float)
    assert isinstance(actual_hi, float)
    np.testing.assert_allclose(actual_lo, expected_lo)
    np.testing.assert_allclose(actual_hi, expected_hi)


@pytest.mark.parametrize(
    "n_1, n_2, n",
    [
        pytest.param(2, 3, 5),
        pytest.param(15, 26, 41),
    ],
)
def test_estimate_add(n_1: int, n_2: int, n: int) -> None:
    samples_1 = np.random.rand(n_1)
    samples_2 = np.random.rand(n_2)
    estimate_1 = Estimate(key="key", samples=samples_1)
    estimate_2 = Estimate(key="key", samples=samples_2)
    estimate = estimate_1 + estimate_2
    assert estimate.n_samples == n
    np.testing.assert_allclose(
        n * estimate.mean(),
        n_1 * estimate_1.mean() + n_2 * estimate_2.mean(),
    )


def test_estimate_add_fail() -> None:
    samples_1 = np.random.rand(2)
    samples_2 = np.random.rand(3)
    estimate_1 = Estimate(key="key_1", samples=samples_1)
    estimate_2 = Estimate(key="key_2", samples=samples_2)
    with pytest.raises(ValueError):
        _ = estimate_1 + estimate_2
