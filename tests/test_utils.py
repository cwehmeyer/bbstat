from typing import Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from bbstat.utils import (
    compute_credibility_interval,
    get_precision_from_credibility_interval,
)


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
    actual = compute_credibility_interval(estimates=estimates, coverage=coverage)
    assert isinstance(actual, tuple)
    assert len(actual) == 2
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "estimates",
    [
        pytest.param(np.array(1)),
        pytest.param(np.array([[1]])),
    ],
)
def test_compute_credibility_interval_fail_on_ndim(
    estimates: NDArray[np.floating],
) -> None:
    with pytest.raises(ValueError):
        _ = compute_credibility_interval(
            estimates=estimates,
            coverage=0.87,
        )


@pytest.mark.parametrize(
    "coverage",
    [
        pytest.param(-1),
        pytest.param(0),
        pytest.param(1),
    ],
)
def test_compute_credibility_interval_fail_on_coverage(
    estimates: NDArray[np.floating],
    coverage: float,
) -> None:
    with pytest.raises(ValueError):
        _ = compute_credibility_interval(
            estimates=estimates,
            coverage=coverage,
        )


@pytest.mark.parametrize(
    "credibility_interval, expected",
    [
        pytest.param((0.0, 0.0), 0),
        pytest.param((1.0, 1.0), 0),
        pytest.param((0.0, 1.0), 1),
        pytest.param((0.0, 9.9), 1),
        pytest.param((0.0, 0.1), 2),
        pytest.param((0.0, 0.999), 2),
        pytest.param((0.0, 0.01), 3),
        pytest.param((0.0, 0.0999), 3),
        pytest.param((0.0, 10.0), 0),
        pytest.param((0.0, 99.9), 0),
        pytest.param((0.0, 100.0), -1),
        pytest.param((9.9, 0.0), 1),
        pytest.param((0.1, 0.0), 2),
    ],
)
def test_bootstrap_result_ndigits(
    credibility_interval: Tuple[float, float], expected: int
) -> None:
    actual = get_precision_from_credibility_interval(credibility_interval)
    assert actual == expected
