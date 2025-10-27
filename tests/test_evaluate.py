from typing import Optional, Tuple, cast

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

from bbstat.evaluate import BootstrapResult, credibility_interval


@pytest.fixture(scope="module")
def estimates() -> NDArray[np.floating]:
    return np.linspace(0, 1, 101)


@pytest.fixture(scope="module")
def bootstrap_result(estimates) -> BootstrapResult:
    return BootstrapResult(estimates=estimates, coverage=0.95)


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


def test_plot_returns_axes(bootstrap_result: BootstrapResult) -> None:
    ax = bootstrap_result.plot()
    assert isinstance(ax, plt.Axes)


@pytest.mark.parametrize(
    "coverage, expected_title",
    [
        pytest.param(None, "Bayesian bootstrap  •  101 resamples, 95% CI"),
        pytest.param(0.99, "Bayesian bootstrap  •  101 resamples, 99% CI"),
    ],
)
def test_plot_respects_coverage_in_title(
    bootstrap_result: BootstrapResult,
    coverage: Optional[float],
    expected_title: str,
) -> None:
    ax = bootstrap_result.plot(coverage=coverage)
    actual_title = ax.get_title()
    assert isinstance(actual_title, str)
    assert actual_title == expected_title


def test_plot_adds_three_lines_and_one_fill(bootstrap_result: BootstrapResult) -> None:
    fig, ax = plt.subplots()
    _ = bootstrap_result.plot(ax=ax)
    assert len(ax.lines) == 3
    assert len([c for c in ax.collections if isinstance(c, mcoll.PolyCollection)]) == 1


@pytest.mark.parametrize(
    "label, expected_label",
    [
        pytest.param(None, "0.499"),
        pytest.param("my_stat", "my_stat=0.499"),
    ],
)
def test_plot_labels_match(
    bootstrap_result: BootstrapResult,
    label: Optional[str],
    expected_label: str,
) -> None:
    label = "my_stat"
    ax = bootstrap_result.plot(label=label)
    actual_label = ax.lines[0].get_label()
    assert isinstance(label, str)
    cast(str, actual_label).startswith(expected_label)
