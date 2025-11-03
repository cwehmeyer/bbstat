"""Utilities bootstrap-related tasks.

This module provides functions to aid interpretation and summarizing the output of
Bayesian bootstrap resampling procedures. It includes tools to compute credible
intervals for statistical estimates and gauging the appropriate precision for
rounding mean and crebilility interval values from the width of the latter.

Main Features:
    - `compute_credible_interval`: Computes a credible interval from a set of estimates.
    - `get_precision_for_rounding`: Gauges the precision for rounding from the
      width of the credible interval.

Notes:
    - The credible interval is calculated using quantiles of the empirical distribution
      of bootstrap estimates.
    - This module is designed to be used alongside the `evaluate` module to provide complete
      statistical summaries of resampled data.
"""

import math
from typing import Tuple

import numpy as np

from .statistics import FArray

__all__ = [
    "compute_credible_interval",
    "get_precision_for_rounding",
]


def compute_credible_interval(
    estimates: FArray,
    level: float = 0.87,
) -> Tuple[float, float]:
    """
    Compute the credible interval for a set of estimates.

    This function calculates the credible interval of the given `estimates` array,
    which is a range of values that contains a specified proportion of the data,
    determined by the `level` parameter.

    The credible interval is calculated by determining the quantiles at
    `(1 - level) / 2` and `1 - (1 - level) / 2` of the sorted `estimates` data.

    Args:
        estimates (FArray): A 1D array of floating-point numbers representing
            the estimates from which the credible interval will be calculated.
        level (float, optional): The proportion of data to be included in the credible
            interval. Must be between 0 and 1 (exclusive). Default is 0.87.

    Returns:
        Tuple[float, float]: A tuple containing the lower and upper bounds of the credible
            interval, with the lower bound corresponding to the `(1 - level) / 2` quantile,
            and the upper bound corresponding to the `1 - (1 - level) / 2` quantile.

    Raises:
        ValueError: If `estimates` is not a 1D array or if `level` is not between 0 and 1
            (exclusive).

    Example:
        ```python
        import numpy as np
        estimates = np.array([1.1, 2.3, 3.5, 2.9, 4.0])
        compute_credible_interval(estimates, 0.6)  # => (2.06, 3.6)
        ```
    """
    if estimates.ndim != 1:
        raise ValueError(f"Invalid parameter {estimates.ndim=:}: must be 1D array.")
    if level <= 0 or level >= 1:
        raise ValueError(f"Invalid parameter {level=:}: must be within (0, 1).")
    edge = (1.0 - level) / 2.0
    return tuple(np.quantile(estimates, [edge, 1.0 - edge]).tolist())


def get_precision_for_rounding(ci_width: float) -> int:
    """
    Returns number of digits for rounding.

    This method computes the precision (number of digits) for rounding mean and
    credible interval values for better readability. If the credible interval
    has width zero, we round to zero digits. Otherwise, we take one minus the floored
    order of magnitude of the width.

    Args:
        ci_width (float): The width of the credible interval.

    Returns:
        int: The number of digits for rounding.

    Raises:
        ValueError: If `ci_width` is negative or NaN.
    """
    if np.isnan(ci_width) or ci_width < 0.0:
        raise ValueError(f"Invalid parameter {ci_width=:}: must be non-negative.")
    if ci_width == 0:
        return 0
    return int(1 - math.floor(math.log10(ci_width)))
