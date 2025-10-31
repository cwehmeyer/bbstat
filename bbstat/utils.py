"""Utilities bootstrap-related tasks.

This module provides functions to aid interpretation and summarizing the output of
Bayesian bootstrap resampling procedures. It includes tools to compute credibility
intervals for statistical estimates and gauging the appropriate precision for
rounding mean and crebilility interval values from the width of the latter.

Main Features:
    - `compute_credibility_interval`: Computes a credibility interval from a set of estimates.
    - `get_precision_from_credibility_interval`: Gauges the precision for rounding from the
      width of the credibility interval.

Notes:
    - The credibility interval is calculated using quantiles of the empirical distribution
      of bootstrap estimates.
    - This module is designed to be used alongside the `evaluate` module to provide complete
      statistical summaries of resampled data.
"""

import math
from typing import Tuple

import numpy as np

from .statistics import FArray

__all__ = [
    "compute_credibility_interval",
    "get_precision_from_credibility_interval",
]


def compute_credibility_interval(
    estimates: FArray,
    coverage: float = 0.87,
) -> Tuple[float, float]:
    """
    Compute the credibility interval for a set of estimates.

    This function calculates the credibility interval of the given `estimates` array,
    which is a range of values that contains a specified proportion of the data,
    determined by the `coverage` parameter.

    The credibility interval is calculated by determining the quantiles at
    `(1 - coverage) / 2` and `1 - (1 - coverage) / 2` of the sorted `estimates` data.

    Args:
        estimates (FArray): A 1D array of floating-point numbers representing
            the estimates from which the credibility interval will be calculated.
        coverage (float, optional): The proportion of data to be included in the credibility
            interval. Must be between 0 and 1 (exclusive). Default is 0.87.

    Returns:
        Tuple[float, float]: A tuple containing the lower and upper bounds of the credibility
            interval, with the lower bound corresponding to the `(1 - coverage) / 2` quantile,
            and the upper bound corresponding to the `1 - (1 - coverage) / 2` quantile.

    Raises:
        ValueError: If `estimates` is not a 1D array or if `coverage` is not between 0 and 1
            (exclusive).

    Example:
        ```python
        import numpy as np
        estimates = np.array([1.1, 2.3, 3.5, 2.9, 4.0])
        compute_credibility_interval(estimates, 0.6)  # => (2.06, 3.6)
        ```
    """
    if estimates.ndim != 1:
        raise ValueError(f"Invalid parameter {estimates.ndim=:}: must be 1D array.")
    if coverage <= 0 or coverage >= 1:
        raise ValueError(f"Invalid parameter {coverage=:}: must be within (0, 1).")
    edge = (1.0 - coverage) / 2.0
    return tuple(np.quantile(estimates, [edge, 1.0 - edge]).tolist())


def get_precision_from_credibility_interval(
    credibility_interval: Tuple[float, float],
) -> int:
    """
    Returns number of digits for rounding.

    This method computes the precision (number of digits) for rounding mean and
    credibility interval values for better readability. If the credibility interval
    has width zero, we round to zero digits. Otherwise, we take one minus the floored
    order of magnitude of the width.

    Args:
        credibility_interval (Tuple[float, float]): The credibility interval given
            by its lower and upper bounds.

    Returns:
        int: The number of digits for rounding.
    """
    lo, hi = credibility_interval
    width = abs(hi - lo)  # use abs in case the values are swapped
    if width == 0:
        return 0
    return int(1 - math.floor(math.log10(width)))
