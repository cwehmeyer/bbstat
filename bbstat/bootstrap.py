"""Bayesian bootstrap resampling for statistical estimation and uncertainty quantification.

This module provides the `bootstrap` function, which applies the Bayesian bootstrap
resampling method to estimate a statistic (such as the mean or median) along with its
credible interval. It supports flexible input data formats, user-defined or
registered statistic functions, and additional customization via keyword arguments.

The function is designed for use in probabilistic data analysis workflows, where
quantifying uncertainty through resampling is critical. It is particularly well-suited
for small to moderate datasets and non-parametric inference.

Main Features:
    - Resampling via the Bayesian bootstrap method.
    - Support for scalar or multivariate data inputs.
    - Use of string-based or function-based statistic definitions.
    - Configurable number of resamples and credible interval level.
    - Optional blockwise resampling for structured data.
    - Random seed control for reproducibility.

Example:
    ```python
    import numpy as np
    from bbstat.bootstrap import bootstrap
    data = np.random.randn(100)
    distribution = bootstrap(data, statistic_fn="mean")
    print(distribution)
    print(distribution.summarize())
    ```

See the function-level docstring of `bootstrap` for full details.
"""

from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from .evaluate import BootstrapDistribution
from .registry import get_statistic_fn
from .resample import resample

__all__ = ["bootstrap"]


def bootstrap(
    data: Any,
    statistic_fn: Union[str, Callable],
    n_boot: int = 1000,
    seed: Optional[int] = None,
    blocksize: Optional[int] = None,
    fn_kwargs: Optional[Dict[str, Any]] = None,
) -> BootstrapDistribution:
    """
    Performs Bayesian bootstrap resampling to estimate a statistic.

    This function performs Bayesian bootstrap resampling by generating `n_boot` resamples from
    the provided `data` and applying the specified statistic function (`statistic_fn`).

    Args:
        data (Any): The data to be resampled. It can be a 1D array, a tuple,
            or a list of arrays where each element represents a different group of data to resample.
        statistic_fn (Union[str, StatisticFunction]): The statistic function to be applied on each
            bootstrap resample. It can either be the name of a registered statistic function or the
            function itself.
        n_boot (int, optional): The number of bootstrap resamples to generate. Default is 1000.
        seed (int, optional): A seed for the random number generator to ensure reproducibility.
            Default is `None`, which means no fixed seed.
        blocksize (int, optional): The block size for resampling. If provided, resampling weights
            are generated in blocks of this size. Defaults to `None`, meaning all resampling weights
            are generated at once.
        fn_kwargs (Dict[str, Any], optional): Additional keyword arguments to be passed to
            the `statistic_fn` for each resample. Default is `None`.

    Returns:
        BootstrapDistribution: An object containing the array with the resampled statistics.

    Raises:
        ValueError: If any data array is not 1D or if the dimensions of the input arrays do not match.

    Example:
        ```python
        data = np.random.randn(100)
        statistic_fn = "mean"
        result = bootstrap(data, statistic_fn)
        print(result)
        print(result.summarize())
        ```

    Notes:
        - The `data` argument can be a single 1D array, or a tuple or list of 1D arrays where each array
          represents a feature of the data.
        - The `statistic_fn` can either be the name of a registered function (as a string) or the function
          itself. If a string is provided, it must match the name of a function in the `statistics.registry`.
        - The function uses the `resample` function to generate bootstrap resamples and apply the statistic
          function to each resample.
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
    return BootstrapDistribution(estimates=estimates)
