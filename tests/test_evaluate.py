from typing import Optional

import numpy as np
import pytest
from numpy.typing import NDArray

from bbstat.evaluate import BootstrapDistribution, BootstrapSummary


@pytest.fixture(scope="module")
def estimates() -> NDArray[np.floating]:
    return np.linspace(0, 1, 101)


@pytest.mark.parametrize(
    "mean, ci_low, ci_high, level",
    [
        pytest.param(np.nan, 0.1, 0.9, 0.87),
        pytest.param(0.5, np.nan, 0.9, 0.87),
        pytest.param(0.5, 0.1, np.nan, 0.87),
        pytest.param(0.5, 0.1, 0.9, np.nan),
        pytest.param(0.5, 0.9, 0.1, 0.87),
        pytest.param(0.0, 0.1, 0.9, 0.87),
        pytest.param(1.0, 0.1, 0.9, 0.87),
        pytest.param(0.5, 0.1, 0.9, 0.0),
        pytest.param(0.5, 0.1, 0.9, 1.0),
    ],
)
def test_bootstrap_summary_fail(
    mean: float,
    ci_low: float,
    ci_high: float,
    level: float,
) -> None:
    with pytest.raises(ValueError):
        _ = BootstrapSummary(mean=mean, ci_low=ci_low, ci_high=ci_high, level=level)


@pytest.mark.parametrize(
    "ci_low, ci_high, expected",
    [
        pytest.param(0.1, 0.9, 0.8),
        pytest.param(0.2, 0.9, 0.7),
        pytest.param(0.1, 0.8, 0.7),
        pytest.param(0.5, 0.5, 0.0),
    ],
)
def test_bootstrap_summary_ci_width(
    ci_low: float,
    ci_high: float,
    expected: float,
) -> None:
    bootstrap_summary = BootstrapSummary(
        mean=0.5,
        ci_low=ci_low,
        ci_high=ci_high,
        level=0.87,
    )
    np.testing.assert_allclose(bootstrap_summary.ci_width, expected)


@pytest.mark.parametrize(
    "precision, expected_mean, expected_ci_low, expected_ci_high",
    [
        pytest.param(1, 0.5, 0.1, 0.9),
        pytest.param(2, 0.51, 0.11, 0.91),
        pytest.param(3, 0.511, 0.111, 0.911),
        pytest.param(None, 0.51, 0.11, 0.91),
    ],
)
def test_bootstrap_summary_round(
    precision: Optional[int],
    expected_mean: float,
    expected_ci_low: float,
    expected_ci_high: float,
) -> None:
    bootstrap_summary = BootstrapSummary(
        mean=0.511111,
        ci_low=0.11111,
        ci_high=0.91111,
        level=0.87,
    )
    bootstrap_summary_rounded = bootstrap_summary.round(precision)
    assert bootstrap_summary_rounded.mean != bootstrap_summary.mean
    assert bootstrap_summary_rounded.ci_low != bootstrap_summary.ci_low
    assert bootstrap_summary_rounded.ci_high != bootstrap_summary.ci_high
    assert bootstrap_summary_rounded.level == bootstrap_summary.level
    np.testing.assert_allclose(bootstrap_summary_rounded.mean, expected_mean)
    np.testing.assert_allclose(bootstrap_summary_rounded.ci_low, expected_ci_low)
    np.testing.assert_allclose(bootstrap_summary_rounded.ci_high, expected_ci_high)


@pytest.mark.parametrize(
    "level, expected_ci_low, expected_ci_high",
    [
        pytest.param(0.87, 0.065, 0.935),
        pytest.param(0.5, 0.25, 0.75),
    ],
)
def test_bootstrap_summary_from_estimates(
    estimates: NDArray[np.floating],
    level: float,
    expected_ci_low: float,
    expected_ci_high: float,
) -> None:
    bootstrap_summary = BootstrapSummary.from_estimates(estimates, level=level)
    np.testing.assert_allclose(bootstrap_summary.mean, 0.5)
    np.testing.assert_allclose(bootstrap_summary.ci_low, expected_ci_low)
    np.testing.assert_allclose(bootstrap_summary.ci_high, expected_ci_high)
    np.testing.assert_allclose(bootstrap_summary.level, level)


@pytest.mark.parametrize(
    "estimates, level",
    [
        pytest.param(np.array([0.4, 0.5, 0.6]), 0.0),
        pytest.param(np.array([0.4, 0.5, 0.6]), 1.0),
        pytest.param(np.array([0.4, 0.5, 0.6]), np.nan),
        pytest.param(np.array([]), 0.87),
        pytest.param(np.array([[0.4, 0.5, 0.6]]), 0.87),
        pytest.param(np.array([0.4, 0.5, np.nan]), 0.87),
    ],
)
def test_bootstrap_summary_from_estimates_fail(
    estimates: NDArray[np.floating],
    level: float,
) -> None:
    with pytest.raises(ValueError):
        _ = BootstrapSummary.from_estimates(estimates, level=level)


def test_bootstrap_distribution(estimates: NDArray[np.floating]) -> None:
    bootstrap_distribution = BootstrapDistribution(estimates)
    assert len(bootstrap_distribution) == len(estimates)
    assert (
        str(bootstrap_distribution)
        == f"BootstrapDistribution(mean={np.mean(estimates)}, size={len(estimates)})"
    )


@pytest.mark.parametrize(
    "level",
    [
        pytest.param(0.2),
        pytest.param(0.8),
    ],
)
def test_bootstrap_distribution_summarize(
    estimates: NDArray[np.floating],
    level: float,
) -> None:
    bootstrap_distribution = BootstrapDistribution(estimates)
    bootstrap_summary = bootstrap_distribution.summarize(level=level)
    assert isinstance(bootstrap_summary, BootstrapSummary)
    np.testing.assert_allclose(bootstrap_summary.mean, 0.5)
    np.testing.assert_allclose(bootstrap_summary.level, level)


@pytest.mark.parametrize(
    "estimates",
    [
        pytest.param(np.array([])),
        pytest.param(np.array([[0.4, 0.5, 0.6]])),
        pytest.param(np.array([0.4, 0.5, np.nan])),
    ],
)
def test_bootstrap_distribution_fail(
    estimates: NDArray[np.floating],
) -> None:
    with pytest.raises(ValueError):
        _ = BootstrapDistribution(estimates)
