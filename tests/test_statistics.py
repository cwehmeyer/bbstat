import numpy as np
import pytest
from numpy.typing import NDArray

from bbstat.statistics import (
    compute_weighted_mean,
    compute_weighted_median,
    compute_weighted_percentile,
    compute_weighted_quantile,
    compute_weighted_std,
    compute_weighted_variance,
    compute_weighted_correlation,
)


@pytest.fixture(scope="module")
def data() -> NDArray[np.floating]:
    return np.random.normal(loc=0.0, scale=1.0, size=101)


@pytest.fixture(scope="module")
def data_2(data: NDArray[np.floating]) -> NDArray[np.floating]:
    return data + np.random.uniform(-1e-2, 1e-2, data.shape)


@pytest.fixture(scope="module")
def weights(data: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.ones_like(data) / len(data)


def test_compute_weighted_mean(
    data: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> None:
    actual = compute_weighted_mean(data=data, weights=weights)
    expected = np.mean(data)
    np.testing.assert_allclose(actual, expected)


def test_compute_weighted_variance(
    data: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> None:
    actual = compute_weighted_variance(data=data, weights=weights)
    expected = np.var(data, ddof=1)
    np.testing.assert_allclose(actual, expected)


def test_compute_weighted_std(
    data: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> None:
    actual = compute_weighted_std(data=data, weights=weights)
    expected = np.std(data, ddof=1)
    np.testing.assert_allclose(actual, expected)


def test_compute_weighted_correlation(
    data: NDArray[np.floating],
    data_2: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> None:
    actual = compute_weighted_correlation(data_1=data, data_2=data_2, weights=weights)
    array_1 = (data - np.mean(data)) / np.std(data, ddof=1)
    array_2 = (data_2 - np.mean(data_2)) / np.std(data_2, ddof=1)
    expected = np.mean(array_1 * array_2)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "use_sorter",
    [
        pytest.param(True),
        pytest.param(False),
    ],
)
def test_compute_weighted_median(
    data: NDArray[np.floating],
    weights: NDArray[np.floating],
    use_sorter: bool,
) -> None:
    actual = compute_weighted_median(
        data=data,
        weights=weights,
        sorter=np.argsort(data) if use_sorter else None,
    )
    expected = np.median(data)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "quantile",
    [
        pytest.param(0.2),
        pytest.param(0.5),
        pytest.param(0.99),
    ],
)
@pytest.mark.parametrize(
    "use_sorter",
    [
        pytest.param(True),
        pytest.param(False),
    ],
)
def test_compute_weighted_quantile(
    data: NDArray[np.floating],
    weights: NDArray[np.floating],
    quantile: float,
    use_sorter: bool,
) -> None:
    actual = compute_weighted_quantile(
        data=data,
        weights=weights,
        quantile=quantile,
        sorter=np.argsort(data) if use_sorter else None,
    )
    expected = np.quantile(data, quantile)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "percentile",
    [
        pytest.param(0.2),
        pytest.param(0.5),
        pytest.param(0.99),
    ],
)
@pytest.mark.parametrize(
    "use_sorter",
    [
        pytest.param(True),
        pytest.param(False),
    ],
)
def test_compute_weighted_percentile(
    data: NDArray[np.floating],
    weights: NDArray[np.floating],
    percentile: float,
    use_sorter: bool,
) -> None:
    actual = compute_weighted_percentile(
        data=data,
        weights=weights,
        percentile=percentile,
        sorter=np.argsort(data) if use_sorter else None,
    )
    expected = np.percentile(data, percentile)
    np.testing.assert_allclose(actual, expected)
