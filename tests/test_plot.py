from typing import Optional, cast

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

from bbstat.evaluate import BootstrapResult
from bbstat.plot import plot


@pytest.fixture(scope="module")
def estimates() -> NDArray[np.floating]:
    return np.linspace(0, 1, 101)


@pytest.fixture(scope="module")
def bootstrap_result(estimates) -> BootstrapResult:
    return BootstrapResult(estimates=estimates, coverage=0.95)


def test_plot_returns_axes(bootstrap_result: BootstrapResult) -> None:
    ax = plot(bootstrap_result)
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
    ax = plot(bootstrap_result, coverage=coverage)
    actual_title = ax.get_title()
    assert isinstance(actual_title, str)
    assert actual_title == expected_title


def test_plot_adds_three_lines_and_one_fill(bootstrap_result: BootstrapResult) -> None:
    fig, ax = plt.subplots()
    _ = plot(bootstrap_result, ax=ax)
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
    ax = plot(bootstrap_result, label=label)
    actual_label = ax.lines[0].get_label()
    assert isinstance(label, str)
    cast(str, actual_label).startswith(expected_label)
