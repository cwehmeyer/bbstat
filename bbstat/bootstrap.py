from typing import Any, Dict, Optional, Union

import numpy as np

from . import statistics
from .evaluate import BootstrapResult
from .resample import resample


def bootstrap(
    data: statistics.StatisticFunctionDataInput,
    statistic_fn: Union[str, statistics.StatisticFunction],
    n_boot: int = 1000,
    coverage: float = 0.87,
    seed: Optional[int] = None,
    blocksize: Optional[int] = None,
    fn_kwargs: Optional[Dict[str, Any]] = None,
) -> BootstrapResult:
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
        statistic_fn = statistics.registry.get(statistic_fn)
    estimates = np.array(
        [
            statistic_fn(data=data, weights=weights, **(fn_kwargs or {}))
            for weights in resample(
                n_boot=n_boot,
                n_data=n_data,
                seed=seed,
                blocksize=blocksize,
            )
        ]
    )
    return BootstrapResult(estimates=estimates, coverage=coverage)
