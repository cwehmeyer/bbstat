"""
resample.py
===========

Bootstrap resampling utilities using Dirichlet-distributed weights.

This module provides functionality for generating bootstrap resamples via the Bayesian
bootstrap method, where resamples are weighted using samples from a Dirichlet distribution.
It is intended for internal use within higher-level resampling and estimation workflows.

The primary function, `resample`, yields weighted resamples suitable for estimating
statistics under uncertainty without making parametric assumptions.

Features:
---------
- Dirichlet-based resampling for Bayesian bootstrap.
- Support for blockwise resample generation to control memory usage.
- Optional random seed for reproducibility.
- Generator interface for efficient streaming of resample weights.

Typical usage:
--------------
>>> from bbstat.resample import resample
>>> for weights in resample(n_boot=1000, n_data=50):
>>>     # Apply weights to compute statistic
>>>     ...

Notes:
------
- The function is designed to scale to large numbers of resamples.
- It is most useful as a low-level utility within a bootstrap framework.

See the `resample` function docstring for complete usage details.
"""

from typing import Generator, Optional

import numpy as np

from .statistics import FArray

__all__ = ["resample"]


def resample(
    n_boot: int,
    n_data: int,
    seed: Optional[int] = None,
    blocksize: Optional[int] = None,
) -> Generator[FArray, None, None]:
    """
    Generates bootstrap resamples with Dirichlet-distributed weights.

    This function performs resampling by generating weights from a Dirichlet distribution.
    The number of resamples is controlled by the `n_boot` argument, while the size of
    each block of resamples can be adjusted using the `blocksize` argument. The `seed`
    argument allows for reproducible results.

    Args:
        n_boot (int): The total number of bootstrap resamples to generate.
        n_data (int): The number of data points to resample (used for the dimension of the
            Dirichlet distribution).
        seed (int, optional): A random seed for reproducibility (default is `None` for
            random seeding).
        blocksize (int, optional): The number of resamples to generate in each block.
            If `None`, the entire number of resamples is generated in one block.
            Defaults to `None`.

    Yields:
        Generator[FArray, None, None]: A generator that yields each resample
            (a 1D array of floats) as it is generated. Each resample contains Dirichlet-
            distributed weights for the given `n_data`.

    Example:
        >>> for r in resample(n_boot=10, n_data=5):
        >>>     print(r)

    Notes:
        - If `blocksize` is specified, the resampling will be performed in smaller blocks,
          which can be useful for parallelizing or limiting memory usage.
        - The function uses NumPy's `default_rng` to generate random numbers, which provides
          a more flexible and efficient interface compared to `np.random.seed`.

    Raises:
        ValueError: If `n_boot` is less than 1 or `n_data` is less than 1.
    """
    if n_boot < 1:
        raise ValueError(f"Invalid parameter {n_boot=:}: must be positive.")
    if n_data < 1:
        raise ValueError(f"Invalid parameter {n_data=:}: must be positive.")
    rng = np.random.default_rng(seed)
    alpha = np.ones(n_data)
    if blocksize is None:
        blocksize = n_boot
    else:
        blocksize = min(blocksize, n_boot)
    remainder = n_boot
    while remainder > 0:
        size = min(blocksize, remainder)
        weights = rng.dirichlet(alpha=alpha, size=size)
        for w in weights:
            yield w
        remainder -= size
