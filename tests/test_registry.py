import pytest

from bbstat.registry import StatisticFunction, get_statistic_fn, get_statistic_fn_names
from bbstat.statistics import (
    compute_weighted_aggregate,
    compute_weighted_eta_square_dependency,
    compute_weighted_mean,
    compute_weighted_median,
    compute_weighted_pearson_dependency,
    compute_weighted_percentile,
    compute_weighted_quantile,
    compute_weighted_spearman_dependency,
    compute_weighted_std,
    compute_weighted_sum,
    compute_weighted_variance,
)


@pytest.mark.parametrize(
    "name, statistic_fn",
    [
        pytest.param("aggregate", compute_weighted_aggregate),
        pytest.param("eta_square_dependency", compute_weighted_eta_square_dependency),
        pytest.param("mean", compute_weighted_mean),
        pytest.param("median", compute_weighted_median),
        pytest.param("pearson_dependency", compute_weighted_pearson_dependency),
        pytest.param("percentile", compute_weighted_percentile),
        pytest.param("quantile", compute_weighted_quantile),
        pytest.param("spearman_dependency", compute_weighted_spearman_dependency),
        pytest.param("std", compute_weighted_std),
        pytest.param("sum", compute_weighted_sum),
        pytest.param("variance", compute_weighted_variance),
    ],
)
def test_get_statistic_fn(name: str, statistic_fn: StatisticFunction) -> None:
    dispatched_fn = get_statistic_fn(name)
    assert dispatched_fn == statistic_fn


def test_get_statistic_fn_fail() -> None:
    with pytest.raises(ValueError):
        _ = get_statistic_fn("unknown")

def test_get_statistic_fn_names() -> None:
    statistic_fn_names = get_statistic_fn_names()
    assert len(statistic_fn_names) > 0
    assert len(statistic_fn_names) == len(set(statistic_fn_names))
    for name in statistic_fn_names:
        _ = get_statistic_fn(name)
