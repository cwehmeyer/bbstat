import math
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

FArray: TypeAlias = NDArray[np.floating]
IArray: TypeAlias = NDArray[np.integer]
FFArray: TypeAlias = Tuple[FArray, FArray]
IFArray: TypeAlias = Tuple[IArray, FArray]
StatisticFunctionDataInput: TypeAlias = Union[
    FArray,
    FFArray,
    IFArray,
    List[FArray],
    List[Union[IArray, FArray]],
]

T = TypeVar("T", bound=StatisticFunctionDataInput, contravariant=True)


class StatisticFunction(Protocol[T]):
    def __call__(self, data: T, weights: FArray, **kwargs: Dict[str, Any]) -> float: ...


class RawStatisticFunction(Protocol):
    def __call__(self, data: Any, weights: FArray, **kwargs: Dict[str, Any]) -> float: ...


_REGISTRY: dict[str, RawStatisticFunction] = {}


def register_stat(name: str):
    def decorator(fn: RawStatisticFunction) -> RawStatisticFunction:
        _REGISTRY[name] = fn
        return fn

    return decorator


@register_stat("aggregate")
def compute_weighted_aggregate(
    data: FArray,
    weights: FArray,
    factor: Optional[float] = None,
) -> float:
    """Compute a weighted aggregate of data.

    This is dot-product betweeen weights and data.

    Parameters
    ----------
    data: numpy.array(shape=(n,))
        Data points to resample and aggregate.
    weights: numpy.array(shape=(n_data,))
        Weights for resampling, either via block or loop.
    factor: Optional[int], default is None
        Rescaling for the aggregation, used to compute means or sums.

    Returns
    -------
    float
        Reweighted and aggregated (and optionally rescaled) data.
    """
    if data.ndim != 1:
        raise ValueError(f"Invalid parameter {data.ndim=:}: must be 1.")
    if weights.ndim != 1:
        raise ValueError(f"Invalid parameter {weights.ndim=:}: must be 1.")
    if weights.shape != data.shape:
        raise ValueError(
            f"Incompatible parameters shapes {weights.shape=:} ≠ {data.shape=:}: "
            "must be equal."
        )
    aggregate = np.dot(weights, data)
    if factor is not None:
        aggregate *= factor
    return aggregate.item()


@register_stat("mean")
def compute_weighted_mean(
    data: FArray,
    weights: FArray,
) -> float:
    return compute_weighted_aggregate(data=data, weights=weights, factor=None)


@register_stat("sum")
def compute_weighted_sum(
    data: FArray,
    weights: FArray,
) -> float:
    return compute_weighted_aggregate(data=data, weights=weights, factor=len(data))


@register_stat("variance")
def compute_weighted_variance(
    data: FArray,
    weights: FArray,
    weighted_mean: Optional[float] = None,
    ddof: int = 0,
) -> float:
    if weighted_mean is None:
        weighted_mean = compute_weighted_mean(data=data, weights=weights)
    return compute_weighted_aggregate(
        data=np.power(data - weighted_mean, 2.0),
        weights=weights,
        factor=len(data) / (len(data) - ddof),
    )


@register_stat("std")
def compute_weighted_std(
    data: FArray,
    weights: FArray,
    weighted_mean: Optional[float] = None,
    ddof: int = 0,
) -> float:
    weighted_variance = compute_weighted_variance(
        data=data,
        weights=weights,
        weighted_mean=weighted_mean,
        ddof=ddof,
    )
    return math.sqrt(weighted_variance)


@register_stat("quantile")
def compute_weighted_quantile(
    data: FArray,
    weights: FArray,
    quantile: float,
    sorter: Optional[IArray] = None,
) -> float:
    if sorter is None:
        sorter = np.argsort(data)
    cumulative_weights = np.cumsum(weights[sorter])
    cw_lo, cw_hi = cumulative_weights[[0, -1]]
    cumulative_weights -= cw_lo
    cumulative_weights /= cw_hi - cw_lo
    data = data[sorter]
    if quantile <= cumulative_weights[0]:
        return data[0]
    if quantile >= cumulative_weights[-1]:
        return data[-1]
    idx = np.searchsorted(cumulative_weights, quantile)
    w_lo, w_hi = cumulative_weights[[idx - 1, idx]]
    s_lo, s_hi = data[[idx - 1, idx]]
    w = (quantile - w_lo) / (w_hi - w_lo)
    return (s_lo + w * (s_hi - s_lo)).item()


@register_stat("percentile")
def compute_weighted_percentile(
    data: FArray,
    weights: FArray,
    percentile: float,
    sorter: Optional[NDArray[np.integer]] = None,
) -> float:
    return compute_weighted_quantile(
        data=data,
        weights=weights,
        quantile=0.01 * percentile,
        sorter=sorter,
    )


@register_stat("median")
def compute_weighted_median(
    data: FArray,
    weights: FArray,
    sorter: Optional[NDArray[np.integer]] = None,
) -> float:
    return compute_weighted_quantile(
        data=data,
        weights=weights,
        quantile=0.5,
        sorter=sorter,
    )


@register_stat("pearson_dependency")
def compute_weighted_pearson_dependency(
    data: FFArray,
    weights: FArray,
    ddof: int = 0,
) -> float:
    data_1, data_2 = data
    weighted_mean_1 = compute_weighted_mean(data=data_1, weights=weights)
    weighted_mean_2 = compute_weighted_mean(data=data_2, weights=weights)
    weighted_std_1 = compute_weighted_std(
        data=data_1,
        weights=weights,
        weighted_mean=weighted_mean_1,
        ddof=ddof,
    )
    weighted_std_2 = compute_weighted_std(
        data=data_2,
        weights=weights,
        weighted_mean=weighted_mean_2,
        ddof=ddof,
    )
    array_1 = (data_1 - weighted_mean_1) / weighted_std_1
    array_2 = (data_2 - weighted_mean_2) / weighted_std_2
    return compute_weighted_mean(data=array_1 * array_2, weights=weights)


@register_stat("spearman_dependency")
def compute_weighted_spearman_dependency(
    data: FFArray,
    weights: FArray,
    ddof: int = 0,
) -> float:
    data_1, data_2 = data
    ranks_1 = rankdata(data_1)
    ranks_2 = rankdata(data_2)
    return compute_weighted_pearson_dependency(
        data=(ranks_1, ranks_2),
        weights=weights,
        ddof=ddof,
    )


@register_stat("eta_square_dependency")
def compute_weighted_eta_square_dependency(
    data: IFArray,
    weights: FArray,
    ddof: int = 0,
) -> float:
    data_cat, data_num = data
    mean_sample = compute_weighted_mean(data=data_num, weights=weights)
    group_variance_sum = 0.0
    for value in np.unique(data_cat):
        subset = data_cat == value
        group_weight = weights[subset].sum()
        mean_group = compute_weighted_aggregate(
            data=data_num[subset],
            weights=weights[subset],
            factor=1.0 / group_weight,
        )
        group_variance_sum += group_weight * (mean_group - mean_sample) ** 2
    return group_variance_sum / compute_weighted_variance(
        data=data_num,
        weights=weights,
        ddof=0,
    )
