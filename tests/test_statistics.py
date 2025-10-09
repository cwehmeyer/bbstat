from typing import Optional

import numpy as np
import pytest
from numpy.typing import NDArray

from bbstat.statistics import (
    compute_weighted_aggregate,
    compute_weighted_mean,
    compute_weighted_median,
    compute_weighted_pearson_dependency,
    compute_weighted_percentile,
    compute_weighted_quantile,
    compute_weighted_std,
    compute_weighted_sum,
    compute_weighted_variance,
)


@pytest.fixture(scope="module")
def data_random() -> NDArray[np.floating]:
    return np.random.normal(loc=0.0, scale=1.0, size=101)


@pytest.fixture(scope="module")
def data_constant() -> NDArray[np.floating]:
    return np.ones(shape=(101,))


@pytest.fixture(scope="module")
def data_dependent(data_random: NDArray[np.floating]) -> NDArray[np.floating]:
    return data_random + np.random.uniform(
        -1e-2,
        1e-2,
        data_random.shape,
    )


@pytest.fixture(scope="module")
def weights_constant(data_random: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.ones_like(data_random) / len(data_random)


@pytest.fixture(scope="module")
def weights_random(data_constant: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.random.default_rng().dirichlet(alpha=np.ones_like(data_constant))


@pytest.mark.parametrize(
    "factor, expected",
    [
        pytest.param(None, 1.0),
        pytest.param(1.0, 1.0),
        pytest.param(2.0, 2.0),
        pytest.param(101.0, 101.0),
    ],
)
def test_compute_weighted_aggregate(
    data_constant: NDArray[np.floating],
    weights_random: NDArray[np.floating],
    factor: Optional[int],
    expected: float,
) -> None:
    actual = compute_weighted_aggregate(
        data=data_constant,
        weights=weights_random,
        factor=factor,
    )
    np.testing.assert_allclose(actual, expected)


def test_compute_weighted_mean_0(
    data_random: NDArray[np.floating],
    weights_constant: NDArray[np.floating],
) -> None:
    actual = compute_weighted_mean(data=data_random, weights=weights_constant)
    expected = np.mean(data_random)
    np.testing.assert_allclose(actual, expected)


def test_compute_weighted_mean_1(
    data_constant: NDArray[np.floating],
    weights_random: NDArray[np.floating],
) -> None:
    actual = compute_weighted_mean(data=data_constant, weights=weights_random)
    expected = np.mean(data_constant)
    np.testing.assert_allclose(actual, expected)


def test_compute_weighted_sum_0(
    data_random: NDArray[np.floating],
    weights_constant: NDArray[np.floating],
) -> None:
    actual = compute_weighted_sum(data=data_random, weights=weights_constant)
    expected = np.sum(data_random)
    np.testing.assert_allclose(actual, expected)


def test_compute_weighted_sum_1(
    data_constant: NDArray[np.floating],
    weights_random: NDArray[np.floating],
) -> None:
    actual = compute_weighted_sum(data=data_constant, weights=weights_random)
    expected = np.sum(data_constant)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("ddof", [pytest.param(0), pytest.param(1)])
def test_compute_weighted_variance(
    data_random: NDArray[np.floating],
    weights_constant: NDArray[np.floating],
    ddof: int,
) -> None:
    actual = compute_weighted_variance(
        data=data_random,
        weights=weights_constant,
        ddof=ddof,
    )
    expected = np.var(data_random, ddof=ddof)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("ddof", [pytest.param(0), pytest.param(1)])
def test_compute_weighted_std(
    data_random: NDArray[np.floating],
    weights_constant: NDArray[np.floating],
    ddof: int,
) -> None:
    actual = compute_weighted_std(data=data_random, weights=weights_constant, ddof=ddof)
    expected = np.std(data_random, ddof=ddof)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("ddof", [pytest.param(0), pytest.param(1)])
def test_compute_weighted_pearson_dependency(
    data_random: NDArray[np.floating],
    data_dependent: NDArray[np.floating],
    weights_constant: NDArray[np.floating],
    ddof: int,
) -> None:
    actual = compute_weighted_pearson_dependency(
        data=(data_random, data_dependent),
        weights=weights_constant,
        ddof=ddof,
    )
    array_1 = (data_random - np.mean(data_random)) / np.std(data_random, ddof=ddof)
    array_2 = (data_dependent - np.mean(data_dependent)) / np.std(
        data_dependent,
        ddof=ddof,
    )
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
    data_random: NDArray[np.floating],
    weights_constant: NDArray[np.floating],
    use_sorter: bool,
) -> None:
    actual = compute_weighted_median(
        data=data_random,
        weights=weights_constant,
        sorter=np.argsort(data_random) if use_sorter else None,
    )
    expected = np.median(data_random)
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
    data_random: NDArray[np.floating],
    weights_constant: NDArray[np.floating],
    quantile: float,
    use_sorter: bool,
) -> None:
    actual = compute_weighted_quantile(
        data=data_random,
        weights=weights_constant,
        quantile=quantile,
        sorter=np.argsort(data_random) if use_sorter else None,
    )
    expected = np.quantile(data_random, quantile)
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
    data_random: NDArray[np.floating],
    weights_constant: NDArray[np.floating],
    percentile: float,
    use_sorter: bool,
) -> None:
    actual = compute_weighted_percentile(
        data=data_random,
        weights=weights_constant,
        percentile=percentile,
        sorter=np.argsort(data_random) if use_sorter else None,
    )
    expected = np.percentile(data_random, percentile)
    np.testing.assert_allclose(actual, expected)
