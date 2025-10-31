from typing import Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from bbstat.evaluate import BootstrapResult


@pytest.fixture(scope="module")
def estimates() -> NDArray[np.floating]:
    return np.linspace(0, 1, 101)


@pytest.mark.parametrize(
    "level, expected_ci",
    [
        pytest.param(0.5, (0.25, 0.75)),
        pytest.param(0.85, (0.075, 0.925)),
    ],
)
def test_bootstrap_result(
    estimates: NDArray[np.floating],
    level: float,
    expected_ci: Tuple[float, float],
) -> None:
    actual = BootstrapResult(estimates=estimates, level=level)
    assert isinstance(actual, BootstrapResult)
    assert actual.n_boot == len(estimates)
    assert actual.level == level
    assert np.all(actual.estimates == estimates)
    np.testing.assert_allclose(actual.mean, 0.5)
    np.testing.assert_allclose(actual.ci, expected_ci)
    lo, hi = actual.ci
    assert lo <= actual.mean <= hi


@pytest.mark.parametrize(
    "level, expected_ci",
    [
        pytest.param(0.5, (0.25, 0.75)),
        pytest.param(0.85, (0.075, 0.925)),
    ],
)
def test_bootstrap_result_credible_interval(
    estimates: NDArray[np.floating],
    level: float,
    expected_ci: Tuple[float, float],
) -> None:
    bootstrap_result = BootstrapResult(estimates=estimates, level=0.1)
    np.testing.assert_allclose(
        bootstrap_result.credible_interval(level=level),
        expected_ci,
    )


def test_bootstrap_result_str() -> None:
    bootstrap_result = BootstrapResult(estimates=np.array([1, 1, 1]), level=0.87)
    actual = str(bootstrap_result)
    expected = "BootstrapResult(mean=1.0, ci=(1.0, 1.0), level=0.87, n_boot=3)"
    assert actual == expected
