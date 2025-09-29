import math

import numpy as np
from numpy.typing import NDArray
from typing import Optional


def compute_weighted_mean(
    data: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> float:
    return np.sum(data * weights).item()


def compute_weighted_variance(
    data: NDArray[np.floating],
    weights: NDArray[np.floating],
    weighted_mean: Optional[float] = None,
) -> float:
    if weighted_mean is None:
        weighted_mean = compute_weighted_mean(data=data, weights=weights)
    numerator = compute_weighted_mean(
        data=(data - weighted_mean) ** 2,
        weights=weights,
    )
    denominator = 1.0 - np.sum(weights**2)  # Assumes normalised weights!
    return (numerator / denominator).item()


def compute_weighted_std(
    data: NDArray[np.floating],
    weights: NDArray[np.floating],
    weighted_mean: Optional[float] = None,
) -> float:
    weighted_variance = compute_weighted_variance(
        data=data,
        weights=weights,
        weighted_mean=weighted_mean,
    )
    return math.sqrt(weighted_variance)


def compute_weighted_quantile(
    data: NDArray[np.floating],
    weights: NDArray[np.floating],
    quantile: float,
    sorter: Optional[NDArray[np.integer]] = None,
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


def compute_weighted_percentile(
    data: NDArray[np.floating],
    weights: NDArray[np.floating],
    percentile: float,
    sorter: Optional[NDArray[np.integer]] = None,
) -> float:
    return compute_weighted_quantile(
        data=data,
        weights=weights,
        quantile=0.01 * percentile,
        sorter=sorter,
    )


def compute_weighted_median(
    data: NDArray[np.floating],
    weights: NDArray[np.floating],
    sorter: Optional[NDArray[np.integer]] = None,
) -> float:
    return compute_weighted_quantile(
        data=data,
        weights=weights,
        quantile=0.5,
        sorter=sorter,
    )


def compute_weighted_correlation(
    data_1: NDArray[np.floating],
    data_2: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> float:
    weighted_mean_1 = compute_weighted_mean(data=data_1, weights=weights)
    weighted_mean_2 = compute_weighted_mean(data=data_2, weights=weights)
    weighted_std_1 = compute_weighted_std(
        data=data_1,
        weights=weights,
        weighted_mean=weighted_mean_1,
    )
    weighted_std_2 = compute_weighted_std(
        data=data_2,
        weights=weights,
        weighted_mean=weighted_mean_2,
    )
    array_1 = (data_1 - weighted_mean_1) / weighted_std_1
    array_2 = (data_2 - weighted_mean_2) / weighted_std_2
    return compute_weighted_mean(data=array_1 * array_2, weights=weights)
