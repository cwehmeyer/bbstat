from typing import Any, Dict, Optional

import numpy as np
import pytest
from numpy.typing import NDArray

from bbstat.bootstrap import bootstrap
from bbstat.evaluate import BootstrapResult
from bbstat.statistics import StatisticFunctionData_T, compute_weighted_aggregate


@pytest.fixture(scope="module")
def data_constant() -> NDArray[np.floating]:
    return np.ones(shape=(101,))


@pytest.fixture(scope="module")
def data_random() -> NDArray[np.floating]:
    return np.random.default_rng(1).normal(size=1000)


@pytest.mark.parametrize(
    "n_boot",
    [
        pytest.param(10),
        pytest.param(100),
    ],
)
@pytest.mark.parametrize(
    "coverage",
    [
        pytest.param(0.1),
        pytest.param(0.9),
    ],
)
@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(None),
        pytest.param(1),
    ],
)
@pytest.mark.parametrize(
    "blocksize",
    [
        pytest.param(None),
        pytest.param(1),
        pytest.param(3),
    ],
)
def test_bootstrap_constant(
    data_constant: NDArray[np.floating],
    n_boot: int,
    coverage: float,
    seed: Optional[int],
    blocksize: Optional[int],
) -> None:
    bootstrap_result = bootstrap(
        data=data_constant,
        statistic_fn=compute_weighted_aggregate,
        n_boot=n_boot,
        coverage=coverage,
        seed=seed,
        blocksize=blocksize,
    )
    assert bootstrap_result.n_boot == n_boot
    assert len(bootstrap_result.estimates) == n_boot
    assert bootstrap_result.coverage == coverage
    assert bootstrap_result.ci[0] <= bootstrap_result.ci[1]
    np.testing.assert_allclose(bootstrap_result.mean, 1.0)
    np.testing.assert_allclose(bootstrap_result.ci, 1.0)
    np.testing.assert_allclose(bootstrap_result.estimates, 1.0)


def test_bootstrap_random(data_random: NDArray[np.floating]) -> None:
    bootstrap_result = bootstrap(
        data=data_random,
        statistic_fn=compute_weighted_aggregate,
        seed=1,
    )
    assert bootstrap_result.ci[0] < bootstrap_result.ci[1]
    np.testing.assert_allclose(bootstrap_result.mean, 0.0, atol=0.07)
    np.testing.assert_allclose(bootstrap_result.ci, (-0.05, 0.05), atol=0.07)


@pytest.mark.parametrize(
    "name, kwargs",
    [
        pytest.param("mean", {}),
        pytest.param("median", {}),
        pytest.param("percentile", {"percentile": 50}),
        pytest.param("quantile", {"quantile": 0.5}),
        pytest.param("std", {}),
        pytest.param("sum", {}),
        pytest.param("variance", {}),
    ],
)
def test_bootstrap_random_single_array(
    data_random: NDArray[np.floating],
    name: str,
    kwargs: Dict[str, Any],
) -> None:
    bootstrap_result = bootstrap(
        data=data_random,
        statistic_fn=name,
        seed=1,
        **kwargs,
    )
    assert isinstance(bootstrap_result, BootstrapResult)


@pytest.mark.parametrize(
    "name, kwargs",
    [
        pytest.param("eta_square_dependency", {}),
        pytest.param("spearman_dependency", {}),
        pytest.param("pearson_dependency", {}),
    ],
)
def test_bootstrap_random_two_arrays(
    data_random: NDArray[np.floating],
    name: str,
    kwargs: Dict[str, Any],
) -> None:
    bootstrap_result = bootstrap(
        data=(np.random.choice(3, size=len(data_random)), data_random),
        statistic_fn=name,
        seed=1,
        **kwargs,
    )
    assert isinstance(bootstrap_result, BootstrapResult)


def test_bootstrap_random_with_factor(data_random: NDArray[np.floating]) -> None:
    bootstrap_result = bootstrap(
        data=data_random,
        statistic_fn=compute_weighted_aggregate,
        seed=1,
        factor=len(data_random),
    )
    assert bootstrap_result.ci[0] < bootstrap_result.ci[1]
    np.testing.assert_allclose(bootstrap_result.mean, 0.0, atol=70.0)
    np.testing.assert_allclose(bootstrap_result.ci, (-50.0, 50.0), atol=70.0)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(np.array(1)),
        pytest.param(np.array([[1]])),
        pytest.param([np.array([1]), np.array([[1]])]),
        pytest.param([np.array([1]), np.array([1, 1])]),
    ],
)
def test_bootstrap_fail_on_data(data: StatisticFunctionData_T) -> None:
    with pytest.raises(ValueError):
        _ = bootstrap(
            data=data,
            statistic_fn=compute_weighted_aggregate,
        )


def test_bootstrap_fail_on_statistic_name(data_random: NDArray[np.floating]) -> None:
    with pytest.raises(ValueError):
        _ = bootstrap(
            data=data_random,
            statistic_fn="undefined function name",
        )