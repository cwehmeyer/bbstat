from typing import Any, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray

from . import statistics
from .evaluate import BootstrapResult
from .resample import resample

REGISTERED_FUNCTIONS: Dict[str, statistics.StatisticFunction_T] = {
    "eta_square_dependency": statistics.compute_weighted_eta_square_dependency,
    "mean": statistics.compute_weighted_mean,
    "median": statistics.compute_weighted_median,
    "pearson_dependency": statistics.compute_weighted_pearson_dependency,
    "percentile": statistics.compute_weighted_percentile,
    "quantile": statistics.compute_weighted_quantile,
    "spearman_dependency": statistics.compute_weighted_spearman_dependency,
    "std": statistics.compute_weighted_std,
    "sum": statistics.compute_weighted_sum,
    "variance": statistics.compute_weighted_variance,
}


def bootstrap(
    data: statistics.StatisticFunctionData_T,
    statistic_fn: Union[
        str,
        statistics.StatisticFunction_T,
    ],
    n_boot: int = 1000,
    coverage: float = 0.87,
    seed: Optional[int] = None,
    blocksize: Optional[int] = None,
    **kwargs: Dict[str, Any],
) -> NDArray[np.floating]:
    if isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError(f"Invalid parameter {data.ndim=:}: must be 1.")
        n_data: int = len(data)
    elif isinstance(data, (tuple, list)):
        n_data = len(data[0])
        for i, array in enumerate(data):
            if array.ndim != 1:
                raise ValueError(f"Invalid parameter {data[i].ndim=:}: must be 1.")
            if n_data != len(array):
                raise ValueError(
                    f"Invalid parameter {data[i].shape[0]=:}: must be {n_data=:}."
                )
    if isinstance(statistic_fn, str):
        if statistic_fn not in REGISTERED_FUNCTIONS.keys():
            raise ValueError(
                f"Unknown {statistic_fn=:}: "
                f"choose from {list(REGISTERED_FUNCTIONS.keys())}."
            )
        statistic_fn = REGISTERED_FUNCTIONS[statistic_fn]
    estimates = np.array(
        [
            statistic_fn(data=data, weights=weights, **kwargs)
            for weights in resample(
                n_boot=n_boot,
                n_data=n_data,
                seed=seed,
                blocksize=blocksize,
            )
        ]
    )
    return BootstrapResult(estimates=estimates, coverage=coverage)
