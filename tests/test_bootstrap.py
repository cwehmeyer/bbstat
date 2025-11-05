from typing import Any, Dict, Optional

import numpy as np
import pytest

from bbstat.bootstrap import bootstrap
from bbstat.evaluate import BootstrapDistribution
from bbstat.statistics import FArray, compute_weighted_aggregate


@pytest.fixture(scope="module")
def data_constant() -> FArray:
    return np.ones(shape=(101,))


@pytest.fixture(scope="module")
def data_random() -> FArray:
    return np.random.default_rng(1).normal(size=1000)


@pytest.mark.parametrize(
    "n_boot",
    [
        pytest.param(10),
        pytest.param(100),
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
    data_constant: FArray,
    n_boot: int,
    seed: Optional[int],
    blocksize: Optional[int],
) -> None:
    bootstrap_distribution = bootstrap(
        data=data_constant,
        statistic_fn=compute_weighted_aggregate,
        n_boot=n_boot,
        seed=seed,
        blocksize=blocksize,
    )
    assert isinstance(bootstrap_distribution, BootstrapDistribution)
    assert len(bootstrap_distribution) == n_boot
    summary = bootstrap_distribution.summarize()
    np.testing.assert_allclose(summary.mean, 1.0)
    np.testing.assert_allclose(summary.ci_low, 1.0)
    np.testing.assert_allclose(summary.ci_high, 1.0)
    np.testing.assert_allclose(bootstrap_distribution.estimates, 1.0)


def test_bootstrap_random(data_random: FArray) -> None:
    bootstrap_distribution = bootstrap(
        data=data_random,
        statistic_fn=compute_weighted_aggregate,
        seed=1,
    )
    assert len(bootstrap_distribution) == 1000  # default
    summary = bootstrap_distribution.summarize()
    assert summary.ci_low < summary.ci_high
    np.testing.assert_allclose(summary.mean, 0.0, atol=0.07)
    np.testing.assert_allclose(summary.ci_low, -0.05, atol=0.07)
    np.testing.assert_allclose(summary.ci_high, 0.05, atol=0.07)


@pytest.mark.parametrize(
    "n_jobs",
    [
        pytest.param(2),
        pytest.param(-1),
    ],
)
def test_bootstrap_random_serial_matches_parallel(
    data_random: FArray, n_jobs: int
) -> None:
    distribution_serial = bootstrap(
        data=data_random,
        statistic_fn=compute_weighted_aggregate,
        n_boot=20,
        seed=1,
    )
    distribution_n_jobs = bootstrap(
        data=data_random,
        statistic_fn=compute_weighted_aggregate,
        n_boot=20,
        seed=1,
        n_jobs=n_jobs,
    )
    assert len(distribution_serial) == len(distribution_n_jobs)
    summary_serial = distribution_serial.summarize(level=0.95, precision="auto")
    summary_n_jobs = distribution_n_jobs.summarize(level=0.95, precision="auto")
    np.testing.assert_allclose(summary_n_jobs.mean, summary_serial.mean, atol=0.02)
    np.testing.assert_allclose(summary_n_jobs.ci_low, summary_serial.ci_low, atol=0.02)
    np.testing.assert_allclose(
        summary_n_jobs.ci_high, summary_serial.ci_high, atol=0.02
    )


@pytest.mark.parametrize(
    "n_jobs",
    [
        pytest.param(2),
        pytest.param(-1),
    ],
)
def test_bootstrap_random_parallel_reproducible(
    data_random: FArray, n_jobs: int
) -> None:
    distribution0 = bootstrap(
        data=data_random,
        statistic_fn=compute_weighted_aggregate,
        n_boot=20,
        seed=1,
        n_jobs=n_jobs,
    )
    distribution1 = bootstrap(
        data=data_random,
        statistic_fn=compute_weighted_aggregate,
        n_boot=20,
        seed=1,
        n_jobs=n_jobs,
    )
    np.testing.assert_allclose(
        np.sort(distribution0.estimates),
        np.sort(distribution1.estimates),
    )


@pytest.mark.parametrize(
    "name, fn_kwargs",
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
    data_random: FArray,
    name: str,
    fn_kwargs: Dict[str, Any],
) -> None:
    bootstrap_distribution = bootstrap(
        data=data_random,
        statistic_fn=name,
        seed=1,
        fn_kwargs=fn_kwargs,
    )
    assert isinstance(bootstrap_distribution, BootstrapDistribution)


@pytest.mark.parametrize(
    "name, fn_kwargs",
    [
        pytest.param("eta_square_dependency", {}),
        pytest.param("spearman_dependency", {}),
        pytest.param("pearson_dependency", {}),
    ],
)
def test_bootstrap_random_two_arrays(
    data_random: FArray,
    name: str,
    fn_kwargs: Dict[str, Any],
) -> None:
    bootstrap_distribution = bootstrap(
        data=(np.random.choice(3, size=len(data_random)), data_random),
        statistic_fn=name,
        seed=1,
        fn_kwargs=fn_kwargs,
    )
    assert isinstance(bootstrap_distribution, BootstrapDistribution)


def test_bootstrap_random_with_factor(data_random: FArray) -> None:
    bootstrap_distribution = bootstrap(
        data=data_random,
        statistic_fn=compute_weighted_aggregate,
        seed=1,
        fn_kwargs={"factor": len(data_random)},
    )
    summary = bootstrap_distribution.summarize()
    assert summary.ci_low < summary.ci_high
    np.testing.assert_allclose(summary.mean, 0.0, atol=70.0)
    np.testing.assert_allclose(summary.ci_low, -50.0, atol=70.0)
    np.testing.assert_allclose(summary.ci_high, 50.0, atol=70.0)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(np.array(1)),
        pytest.param(np.array([[1]])),
        pytest.param([np.array([1]), np.array([[1]])]),
        pytest.param([np.array([1]), np.array([1, 1])]),
    ],
)
def test_bootstrap_fail_on_data(data: Any) -> None:
    with pytest.raises(ValueError):
        _ = bootstrap(
            data=data,
            statistic_fn=compute_weighted_aggregate,
        )


def test_bootstrap_fail_on_statistic_name(data_random: FArray) -> None:
    with pytest.raises(ValueError):
        _ = bootstrap(
            data=data_random,
            statistic_fn="undefined function name",
        )
