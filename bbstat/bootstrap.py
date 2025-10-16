"""Bayesian bootstrap resampling for statistical estimation and uncertainty quantification.

This module provides the `bootstrap` function, which applies the Bayesian bootstrap
resampling method to estimate a statistic (such as the mean or median) along with its
credibility interval. It supports flexible input data formats, user-defined or
registered statistic functions, and additional customization via keyword arguments.

The function is designed for use in probabilistic data analysis workflows, where
quantifying uncertainty through resampling is critical. It is particularly well-suited
for small to moderate datasets and non-parametric inference.

Main Features:
    - Resampling via the Bayesian bootstrap method.
    - Support for scalar or multivariate data inputs.
    - Use of string-based or function-based statistic definitions.
    - Configurable number of resamples and credibility interval coverage.
    - Optional blockwise resampling for structured data.
    - Random seed control for reproducibility.

Example:
    ```python
    import numpy as np
    from bbstat.bootstrap import bootstrap
    data = np.random.randn(100)
    result = bootstrap(data, statistic_fn="mean")
    print(result)
    ```

See the function-level docstring of `bootstrap` for full details.
"""

from typing import Any, Dict, Optional, Union, Callable

import numpy as np

from .registry import get_statistic_fn
from .evaluate import BootstrapResult
from .resample import resample

__all__ = ["bootstrap"]


def bootstrap(
    data: Any,
    statistic_fn: Union[str, Callable],
    n_boot: int = 1000,
    coverage: float = 0.87,
    seed: Optional[int] = None,
    blocksize: Optional[int] = None,
    fn_kwargs: Optional[Dict[str, Any]] = None,
) -> BootstrapResult:
    """
    Performs Bayesian bootstrap resampling to estimate a statistic and its credibility interval.

    This function performs Bayesian bootstrap resampling by generating `n_boot` resamples from
    the provided `data` and applying the specified statistic function (`statistic_fn`). It then
    computes the mean and credibility interval for the estimated statistic across all resamples.

    Args:
        data (Any): The data to be resampled. It can be a 1D array, a tuple,
            or a list of arrays where each element represents a different group of data to resample.
        statistic_fn (Union[str, StatisticFunction]): The statistic function to be applied on each
            bootstrap resample. It can either be the name of a registered statistic function or the
            function itself.
        n_boot (int, optional): The number of bootstrap resamples to generate. Default is 1000.
        coverage (float, optional): The coverage level for the credibility interval (between 0 and 1).
            Default is 0.87.
        seed (int, optional): A seed for the random number generator to ensure reproducibility.
            Default is `None`, which means no fixed seed.
        blocksize (int, optional): The block size for resampling. If provided, resampling weights
            are generated in blocks of this size. Defaults to `None`, meaning all resampling weights
            are generated at once.
        fn_kwargs (Dict[str, Any], optional): Additional keyword arguments to be passed to
            the `statistic_fn` for each resample. Default is `None`.

    Returns:
        BootstrapResult: An object containing the mean of the resampled statistics, the credibility
            interval, and other details of the bootstrap procedure.

    Raises:
        ValueError: If any data array is not 1D or if the dimensions of the input arrays do not match.

    Example:
        ```python
        data = np.random.randn(100)
        statistic_fn = "mean"
        result = bootstrap(data, statistic_fn)
        print(result.mean)
        print(result.ci)
        ```

    Notes:
        - The `data` argument can be a single 1D array, or a tuple or list of 1D arrays where each array
          represents a feature of the data.
        - The `statistic_fn` can either be the name of a registered function (as a string) or the function
          itself. If a string is provided, it must match the name of a function in the `statistics.registry`.
        - The function uses the `resample` function to generate bootstrap resamples and apply the statistic
          function to each resample.
        - The default `coverage` level of 0.87 corresponds to a 87% credibility interval, but this can be
          adjusted as needed.
    """
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
        statistic_fn = get_statistic_fn(statistic_fn)
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
