from typing import Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from bbstat.evaluate import BootstrapResult, credibility_interval


@pytest.fixture(scope="module")
def estimates() -> NDArray[np.floating]:
    return np.linspace(0, 1, 101)


@pytest.mark.parametrize(
    "coverage, expected",
    [
        pytest.param(0.5, (0.25, 0.75)),
        pytest.param(0.85, (0.075, 0.925)),
    ],
)
def test_credibility_interval(
    estimates: NDArray[np.floating],
    coverage: float,
    expected: Tuple[float, float],
) -> None:
    actual = credibility_interval(estimates=estimates, coverage=coverage)
    assert isinstance(actual, tuple)
    assert len(actual) == 2
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "coverage, expected_ci",
    [
        pytest.param(0.5, (0.25, 0.75)),
        pytest.param(0.85, (0.075, 0.925)),
    ],
)
def test_bootstrap_result(
    estimates: NDArray[np.floating],
    coverage: float,
    expected_ci: Tuple[float, float],
) -> None:
    actual = BootstrapResult(estimates=estimates, coverage=coverage)
    assert isinstance(actual, BootstrapResult)
    assert actual.n_boot == len(estimates)
    assert actual.coverage == coverage
    assert np.all(actual.estimates == estimates)
    np.testing.assert_allclose(actual.mean, 0.5)
    np.testing.assert_allclose(actual.ci, expected_ci)
