"""Evaluation utilities for summarizing bootstrap resampling results.

This module provides a data structure for interpreting and summarizing the output of
Bayesian bootstrap resampling procedures.

Main Features:
    - `BootstrapResult`: A data class that holds bootstrap estimates, computes the mean,
      and automatically evaluates the credible interval.

Example:
    ```python
    from bbstat.evaluate import BootstrapResult
    result = BootstrapResult(estimates=np.array([5.0, 2.3, 2.9]), level=0.95)
    print(result)  # => BootstrapResult(mean=3.4, ci=(2.3, 4.9), level=0.95, n_boot=3)
    ```

Notes:
    - This module is designed to be used alongside the `bootstrap` and `resample` modules
      to provide complete statistical summaries of resampled data.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from .statistics import FArray
from .utils import compute_credible_interval, get_precision_from_credible_interval

__all__ = ["BootstrapResult"]


@dataclass
class BootstrapResult:
    """
    A class representing the result of a bootstrap resampling procedure.

    This class stores the mean, credible interval, and other statistics resulting
    from a Bayesian bootstrap analysis, and provides methods to display the results and
    calculate related statistics such as the credible interval.

    Attributes:
        mean (float): The mean of the bootstrap estimates.
        ci (Tuple[float, float]): The lower and upper bounds of the credible interval.
        level (float): The desired level for the credible interval (between 0 and 1).
        n_boot (int): The number of bootstrap resamples (i.e., the number of estimates).
        estimates (FArray): The array of bootstrap resample estimates.

    Methods:
        __post_init__: Initializes the `mean`, `ci`, and `n_boot` attributes.
        __str__: Returns a string representation of the object.
        credible_interval: Calculates the credible interval for the bootstrap estimates.

    Raises:
        ValueError: If `estimates` is not a 1D array or if `level` is not between 0 and 1
            (exclusive).
    """

    mean: float = field(init=False)
    ci: Tuple[float, float] = field(init=False)
    level: float
    n_boot: int = field(init=False)
    estimates: FArray

    def __post_init__(self):
        """
        Post-initialization method to initialize the mean, credible interval,
        and the number of bootstrap resamples from the provided estimates and
        level paremeters.

        This method is automatically called after the object is initialized.
        It calculates:
            - The mean of the bootstrap estimates.
            - The credible interval using the provided level.
            - The number of bootstrap resamples.

        Raises:
            ValueError: If `estimates` is not a 1D array or if `level` is not
                between 0 and 1 (exclusive).
        """
        self.mean = np.mean(self.estimates).item()
        self.ci = compute_credible_interval(
            estimates=self.estimates,
            level=self.level,
        )
        self.n_boot = len(self.estimates)

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the bootstrap result.

        This method formats the mean, credible interval, level, and the
        number of bootstrap resamples for display.

        Returns:
            str: A formatted string representing the bootstrap result.
        """
        ndigits = get_precision_from_credible_interval(self.ci)
        mean = round(number=self.mean, ndigits=ndigits)
        lo = round(number=self.ci[0], ndigits=ndigits)
        hi = round(number=self.ci[1], ndigits=ndigits)
        return f"BootstrapResult(mean={mean}, ci={(lo, hi)}, level={self.level}, n_boot={self.n_boot})"

    def credible_interval(self, level: float) -> Tuple[float, float]:
        """
        Calculate the credible interval for the bootstrap estimates.

        This method is a wrapper for the `credible_interval` function. It takes
        a `level` value (between 0 and 1) and returns the lower and upper bounds
        of the credible interval.

        Args:
            level (float): The desired level for the credible interval
                (must be between 0 and 1).

        Returns:
            Tuple[float, float]: The lower and upper bounds of the credible
                interval based on the given level.

        Raises:
            ValueError: If the `level` is not between 0 and 1.
        """
        return compute_credible_interval(estimates=self.estimates, level=level)
