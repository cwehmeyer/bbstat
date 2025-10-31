"""Evaluation utilities for summarizing bootstrap resampling results.

This module provides a data structure for interpreting and summarizing the output of
Bayesian bootstrap resampling procedures.

Main Features:
    - `BootstrapResult`: A data class that holds bootstrap estimates, computes the mean,
      and automatically evaluates the credibility interval.

Example:
    ```python
    from bbstat.evaluate import BootstrapResult
    result = BootstrapResult(estimates=np.array([5.0, 2.3, 2.9]), coverage=0.95)
    print(result)  # => BootstrapResult(mean=3.4, ci=(2.3, 4.9), coverage=0.95, n_boot=3)
    ```

Notes:
    - This module is designed to be used alongside the `bootstrap` and `resample` modules
      to provide complete statistical summaries of resampled data.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from .statistics import FArray
from .utils import compute_credibility_interval, get_precision_from_credibility_interval

__all__ = ["BootstrapResult"]


@dataclass
class BootstrapResult:
    """
    A class representing the result of a bootstrap resampling procedure.

    This class stores the mean, credibility interval, and other statistics resulting
    from a Bayesian bootstrap analysis, and provides methods to display the results and
    calculate related statistics such as the credibility interval.

    Attributes:
        mean (float): The mean of the bootstrap estimates.
        ci (Tuple[float, float]): The lower and upper bounds of the credibility interval.
        coverage (float): The desired coverage for the credibility interval (between 0 and 1).
        n_boot (int): The number of bootstrap resamples (i.e., the number of estimates).
        estimates (FArray): The array of bootstrap resample estimates.

    Methods:
        __post_init__: Initializes the `mean`, `ci`, and `n_boot` attributes.
        __str__: Returns a string representation of the object.
        credibility_interval: Calculates the credibility interval for the bootstrap estimates.

    Raises:
        ValueError: If `estimates` is not a 1D array or if `coverage` is not between 0 and 1
            (exclusive).
    """

    mean: float = field(init=False)
    ci: Tuple[float, float] = field(init=False)
    coverage: float
    n_boot: int = field(init=False)
    estimates: FArray

    def __post_init__(self):
        """
        Post-initialization method to initialize the mean, credibility interval,
        and the number of bootstrap resamples from the provided estimates and
        coverage paremeters.

        This method is automatically called after the object is initialized.
        It calculates:
            - The mean of the bootstrap estimates.
            - The credibility interval using the provided coverage.
            - The number of bootstrap resamples.

        Raises:
            ValueError: If `estimates` is not a 1D array or if `coverage` is not
                between 0 and 1 (exclusive).
        """
        self.mean = np.mean(self.estimates).item()
        self.ci = compute_credibility_interval(
            estimates=self.estimates,
            coverage=self.coverage,
        )
        self.n_boot = len(self.estimates)

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the bootstrap result.

        This method formats the mean, credibility interval, coverage, and the
        number of bootstrap resamples for display.

        Returns:
            str: A formatted string representing the bootstrap result.
        """
        ndigits = get_precision_from_credibility_interval(self.ci)
        mean = round(number=self.mean, ndigits=ndigits)
        lo = round(number=self.ci[0], ndigits=ndigits)
        hi = round(number=self.ci[1], ndigits=ndigits)
        return f"BootstrapResult(mean={mean}, ci={(lo, hi)}, coverage={self.coverage}, n_boot={self.n_boot})"

    def credibility_interval(self, coverage: float) -> Tuple[float, float]:
        """
        Calculate the credibility interval for the bootstrap estimates.

        This method is a wrapper for the `credibility_interval` function. It takes
        a `coverage` value (between 0 and 1) and returns the lower and upper bounds
        of the credibility interval.

        Args:
            coverage (float): The desired coverage for the credibility interval
                (must be between 0 and 1).

        Returns:
            Tuple[float, float]: The lower and upper bounds of the credibility
                interval based on the given coverage.

        Raises:
            ValueError: If the `coverage` is not between 0 and 1.
        """
        return compute_credibility_interval(estimates=self.estimates, coverage=coverage)
