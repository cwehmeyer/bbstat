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
    lo, hi = actual.ci
    assert lo <= actual.mean <= hi


@pytest.mark.parametrize(
    "coverage, expected_ci",
    [
        pytest.param(0.5, (0.25, 0.75)),
        pytest.param(0.85, (0.075, 0.925)),
    ],
)
def test_bootstrap_result_credibility_interval(
    estimates: NDArray[np.floating],
    coverage: float,
    expected_ci: Tuple[float, float],
) -> None:
    bootstrap_result = BootstrapResult(estimates=estimates, coverage=0.1)
    np.testing.assert_allclose(
        bootstrap_result.credibility_interval(coverage=coverage),
        expected_ci,
    )


@pytest.mark.parametrize(
    "estimates",
    [
        pytest.param(np.array(1)),
        pytest.param(np.array([[1]])),
    ],
)
def test_credibility_interval_fail_on_ndim(estimates: NDArray[np.floating]) -> None:
    with pytest.raises(ValueError):
        _ = credibility_interval(
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
def test_credibility_interval_fail_on_coverage(
    estimates: NDArray[np.floating],
    coverage: float,
) -> None:
    with pytest.raises(ValueError):
        _ = credibility_interval(
            estimates=estimates,
            coverage=coverage,
        )


def test_bootstrap_result_str() -> None:
    bootstrap_result = BootstrapResult(estimates=np.array([1, 1, 1]), coverage=0.87)
    actual = str(bootstrap_result)
    expected = "BootstrapResult(mean=1.0, ci=(1.0, 1.0), coverage=0.87, n_boot=3)"
    assert actual == expected


@pytest.mark.parametrize(
    "ci, expected",
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
    ],
)
def test_bootstrap_result_ndigits(ci: Tuple[float, float], expected: int) -> None:
    actual = BootstrapResult.ndigits(ci=ci)
    assert actual == expected
