from typing import Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from bbstat.utils import (
    compute_credible_interval,
    get_precision_for_rounding,
)


@pytest.fixture(scope="module")
def estimates() -> NDArray[np.floating]:
    return np.linspace(0, 1, 101)


@pytest.mark.parametrize(
    "level, expected",
    [
        pytest.param(0.5, (0.25, 0.75)),
        pytest.param(0.85, (0.075, 0.925)),
    ],
)
def test_credible_interval(
    estimates: NDArray[np.floating],
    level: float,
    expected: Tuple[float, float],
) -> None:
    actual = compute_credible_interval(estimates=estimates, level=level)
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
def test_compute_credible_interval_fail_on_ndim(
    estimates: NDArray[np.floating],
) -> None:
    with pytest.raises(ValueError):
        _ = compute_credible_interval(
            estimates=estimates,
            level=0.87,
        )


@pytest.mark.parametrize(
    "level",
    [
        pytest.param(-1),
        pytest.param(0),
        pytest.param(1),
    ],
)
def test_compute_credible_interval_fail_on_level(
    estimates: NDArray[np.floating],
    level: float,
) -> None:
    with pytest.raises(ValueError):
        _ = compute_credible_interval(
            estimates=estimates,
            level=level,
        )


@pytest.mark.parametrize(
    "ci_width, expected",
    [
        pytest.param(0.0, 0),
        pytest.param(0.01, 3),
        pytest.param(0.0999, 3),
        pytest.param(0.1, 2),
        pytest.param(0.999, 2),
        pytest.param(1.0, 1),
        pytest.param(9.9, 1),
        pytest.param(10.0, 0),
        pytest.param(99.9, 0),
        pytest.param(100.0, -1),
    ],
)
def test_get_precision_for_rounding(ci_width: float, expected: int) -> None:
    actual = get_precision_for_rounding(ci_width)
    assert actual == expected


@pytest.mark.parametrize(
    "ci_width",
    [
        pytest.param(-1.0),
        pytest.param(np.nan),
    ],
)
def test_get_precision_for_rounding_fail(ci_width: float) -> None:
    with pytest.raises(ValueError):
        _ = get_precision_for_rounding(ci_width)
