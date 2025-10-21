from typing import Optional

import numpy as np
import pytest

from bbstat.statistics import (
    IArray,
    FArray,
    compute_weighted_aggregate,
    compute_weighted_entropy,
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
def data_random() -> FArray:
    return np.random.normal(loc=0.0, scale=1.0, size=101)


@pytest.fixture(scope="module")
def data_constant(data_random) -> FArray:
    return np.ones(shape=data_random.shape)


@pytest.fixture(scope="module")
def data_random_code(data_random) -> IArray:
    return np.random.choice(2, size=len(data_random))


@pytest.fixture(scope="module")
def data_constant_code(data_random_code) -> IArray:
    return np.zeros(shape=data_random_code.shape, dtype=data_random_code.dtype)


@pytest.fixture(scope="module")
def data_dependent(data_random: FArray) -> FArray:
    return data_random + np.random.uniform(
        -1e-2,
        1e-2,
        data_random.shape,
    )


@pytest.fixture(scope="module")
def weights_constant(data_random: FArray) -> FArray:
    return np.ones_like(data_random) / len(data_random)


@pytest.fixture(scope="module")
def weights_random(data_constant: FArray) -> FArray:
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
    data_constant: FArray,
    weights_random: FArray,
    factor: Optional[int],
    expected: float,
) -> None:
    actual = compute_weighted_aggregate(
        data=data_constant,
        weights=weights_random,
        factor=factor,
    )
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "data, weights",
    [
        pytest.param(np.array(1), np.array([0.5, 0.5])),
        pytest.param(np.array([[1]]), np.array([0.5, 0.5])),
        pytest.param(np.array([0.5, 0.5]), np.array(1)),
        pytest.param(np.array([0.5, 0.5]), np.array([[1]])),
        pytest.param(np.array([0.5, 0.5]), np.array([0.5])),
    ],
)
def test_compute_weighted_aggregate_fail(
    data: FArray,
    weights: FArray,
) -> None:
    with pytest.raises(ValueError):
        _ = compute_weighted_aggregate(
            data=data,
            weights=weights,
        )


def test_compute_weighted_mean_0(
    data_random: FArray,
    weights_constant: FArray,
) -> None:
    actual = compute_weighted_mean(data=data_random, weights=weights_constant)
    expected = np.mean(data_random)
    np.testing.assert_allclose(actual, expected)


def test_compute_weighted_mean_1(
    data_constant: FArray,
    weights_random: FArray,
) -> None:
    actual = compute_weighted_mean(data=data_constant, weights=weights_random)
    expected = np.mean(data_constant)
    np.testing.assert_allclose(actual, expected)


def test_compute_weighted_sum_0(
    data_random: FArray,
    weights_constant: FArray,
) -> None:
    actual = compute_weighted_sum(data=data_random, weights=weights_constant)
    expected = np.sum(data_random)
    np.testing.assert_allclose(actual, expected)


def test_compute_weighted_sum_1(
    data_constant: FArray,
    weights_random: FArray,
) -> None:
    actual = compute_weighted_sum(data=data_constant, weights=weights_random)
    expected = np.sum(data_constant)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("ddof", [pytest.param(0), pytest.param(1)])
def test_compute_weighted_variance(
    data_random: FArray,
    weights_constant: FArray,
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
    data_random: FArray,
    weights_constant: FArray,
    ddof: int,
) -> None:
    actual = compute_weighted_std(data=data_random, weights=weights_constant, ddof=ddof)
    expected = np.std(data_random, ddof=ddof)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("ddof", [pytest.param(0), pytest.param(1)])
def test_compute_weighted_pearson_dependency(
    data_random: FArray,
    data_dependent: FArray,
    weights_constant: FArray,
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
    data_random: FArray,
    weights_constant: FArray,
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
    data_random: FArray,
    weights_constant: FArray,
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
    "quantile",
    [
        pytest.param(-1),
        pytest.param(0),
    ],
)
def test_compute_weighted_quantile_underflow(
    data_random: FArray,
    weights_constant: FArray,
    quantile: float,
) -> None:
    actual = compute_weighted_quantile(
        data=data_random,
        weights=weights_constant,
        quantile=quantile,
        sorter=None,
    )
    expected = np.min(data_random)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "quantile",
    [
        pytest.param(1),
        pytest.param(2),
    ],
)
def test_compute_weighted_quantile_overflow(
    data_random: FArray,
    weights_constant: FArray,
    quantile: float,
) -> None:
    actual = compute_weighted_quantile(
        data=data_random,
        weights=weights_constant,
        quantile=quantile,
        sorter=None,
    )
    expected = np.max(data_random)
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
    data_random: FArray,
    weights_constant: FArray,
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


def test_compute_weighted_entropy_0(
    data_random_code: IArray, weights_constant: FArray,
) -> None:
    actual = compute_weighted_entropy(data=data_random_code, weights=weights_constant)
    distribution = np.bincount(data_random_code) / len(data_random_code)
    distribution = distribution[distribution > 0]
    expected = -np.sum(distribution * np.log(distribution))
    np.testing.assert_allclose(actual, expected)
    expected


def test_compute_weighted_entropy_1(
    data_constant_code: IArray, weights_random: FArray,
) -> None:
    actual = compute_weighted_entropy(data=data_constant_code, weights=weights_random)
    np.testing.assert_allclose(actual, 0.0, atol=1e-15)
