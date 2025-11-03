"""Evaluation utilities for summarizing bootstrap resampling results.

This module provides a data structure for interpreting and summarizing the output of
Bayesian bootstrap resampling procedures.

Main Features:
    - `BootstrapDistribution`: A frozen data class representing the resulting distribution
      of a bootstrap resampling procedure.
    - `BootstrapSummary`: A frozen data class that holds the summary (mean, credible interval,
      and level) of a Bayesian bootstrap procedure's result.

Example:
    ```python
    import numpy as np
    from bbstat.evaluate import BootstrapDistribution
    distribution = BootstrapDistribution(estimates=np.array([5.0, 2.3, 2.9]))
    print(distribution)  # => BootstrapDistribution(mean=3.4, size=3)
    summary = distribution.summarize(level=0.95)
    print(summary)  # => BootstrapSummary(mean=3.4, ci_low=2.33, ci_high=4.895, level=0.95)
    ```

Notes:
    - This module is designed to be used alongside the `bootstrap` and `resample` modules
      to provide complete statistical summaries of resampled data.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .statistics import FArray
from .utils import compute_credible_interval, get_precision_for_rounding

__all__ = [
    "BootstrapDistribution",
    "BootstrapSummary",
]


@dataclass(frozen=True)
class BootstrapSummary:
    """
    A class representing the summary of a Bayesian bootstrap resampling procedure.

    This class stores the mean, the credible interval, and level.

    Attributes:
        mean (float): The mean of the bootstrap estimates.
        ci_low (float): The lower bound of the credible interval.
        ci_high (float): The upper bound of the credible interval.
        level (float): The desired level for the credible interval (between 0 and 1).
        ci_width (float): The width of the credible interval (property).

    Methods:
        __post_init__: Validates the `mean`, `ci_low`, `ci_high`, and `level` attributes.
        round: Returns a new version of the summary with rounded values.
        from_estimates: Creates a summary object from estimates.

    Raises:
        ValueError: If `mean`, `ci_low`, `ci_high`, or `level` are NaN.
        ValueError: If the bounds are swapped, `ci_low > ci_high`.
        ValueError: If `level` is not between 0 and 1 (exclusive).
    """

    mean: float
    ci_low: float
    ci_high: float
    level: float

    def __post_init__(self) -> None:
        """
        Post-initialization method to validate the `mean`, `ci_low`,
        `ci_high`, and `level` attributes.

        Raises:
            ValueError: If `mean`, `ci_low`, `ci_high`, or `level` are NaN.
            ValueError: If the bounds are swapped, `ci_low > ci_high`.
            ValueError: If `level` is not between 0 and 1 (exclusive).
        """
        if np.isnan(self.mean):
            raise ValueError("Invalid parameter mean: must not be NaN.")
        if np.isnan(self.ci_low):
            raise ValueError("Invalid parameter ci_low: must not be NaN.")
        if np.isnan(self.ci_high):
            raise ValueError("Invalid parameter ci_high: must not be NaN.")
        if np.isnan(self.level):
            raise ValueError("Invalid parameter level: must not be NaN.")
        if self.ci_low > self.ci_high:
            raise ValueError(
                f"Invalid parameters {self.ci_low=:} and {self.ci_high=:}: "
                "higher end is smaller than lower end."
            )
        if self.level <= 0 or self.level >= 1:
            raise ValueError(
                f"Invalid parameter {self.level=:}: must be within (0, 1)."
            )
        if self.mean < self.ci_low or self.mean > self.ci_high:
            raise ValueError(
                f"Invalid parameter {self.mean=:}: is outside the credible interval "
                f"{self.ci_low=:}, {self.ci_high}."
            )

    @property
    def ci_width(self) -> float:
        """Returns the width of the credible interval."""
        return self.ci_high - self.ci_low

    def round(self, precision: Optional[int] = None) -> "BootstrapSummary":
        """
        Returns a new version of the summary with rounded values.

        When `precision` is given, the mean and credible interval bounds are rounded
        to this number of digits. If `precision=None` (default), the precision is
        computed form the width of the credible interval.

        Args:
            precision (int, optional): The desired precision for rounding.

        Returns:
            BootstrapSummary: The summary of a Bayesian bootstrap procedure's result.
        """
        if precision is None:
            precision = get_precision_for_rounding(self.ci_width)
        return self.__class__(
            mean=round(self.mean, precision),
            ci_low=round(self.ci_low, precision),
            ci_high=round(self.ci_high, precision),
            level=self.level,
        )

    @classmethod
    def from_estimates(
        cls,
        estimates: FArray,
        *,
        level: float = 0.87,
    ) -> "BootstrapSummary":
        """
        Creates a summary object from estimates.

        This method computes the `mean` and credible interval bounds `ci_low` and
        `ci_high`, and creates a `BootstrapSummary` object.

        Args:
            estimates (FArray): The estimated values from a Bayesian bootstrap procedure.
            level (float): The desired level for the credible interval (between 0 and 1),
                default is 0.87.

        Returns:
            BootstrapSummary: The summary of a Bayesian bootstrap procedure's result.

        Raises:
            ValueError: If estimates is empty, not a 1D array, or contains NaN values.
            ValueError: If level is not between 0 and 1 (exclusive).
        """
        if estimates.ndim != 1:
            raise ValueError(f"Invalid parameter {estimates.ndim=}: must be 1.")
        if len(estimates) < 1:
            raise ValueError("Invalid parameter estimates: must not be empty.")
        if np.isnan(estimates).any():
            raise ValueError(
                "Invalid parameter estimates: must not contain NaN values."
            )
        mean = np.mean(estimates).item()
        ci_low, ci_high = compute_credible_interval(estimates=estimates, level=level)
        return cls(mean=mean, ci_low=ci_low, ci_high=ci_high, level=level)


@dataclass(frozen=True)
class BootstrapDistribution:
    """
    A class representing the resulting distribution of a bootstrap resampling procedure.

    This class stores the distribution resulting from a Bayesian bootstrap analysis,
    and provides a method to summarize the result.

    Attributes:
        estimates (FArray): The array of bootstrap resample estimates.

    Methods:
        __post_init__: Validates and locks the `estimates` attribute.
        __len__: Returns the length of the `estimates` array.
        __str__: Returns a string representation of the object.
        summarize: Returns a `BootstrapSummary` object.

    Raises:
        ValueError: If `estimates` is not a 1D array or contains NaN values.
    """

    estimates: FArray

    def __post_init__(self):
        """
        Post-initialization method to validate and lock the estimates array.

        Raises:
            ValueError: If `estimates` is not a 1D array or contains NaN values.
        """
        if self.estimates.ndim != 1:
            raise ValueError(f"Invalid parameter {self.estimates.ndim=}: must be 1.")
        if len(self.estimates) < 1:
            raise ValueError("Invalid parameter estimates: must not be empty.")
        if np.isnan(self.estimates).any():
            raise ValueError(
                "Invalid parameter estimates: must not contain NaN values."
            )
        estimates_copy = np.array(self.estimates, copy=True)
        estimates_copy.setflags(write=False)
        object.__setattr__(self, "estimates", estimates_copy)

    def __len__(self) -> int:
        """Returns the length of the estimates array."""
        return len(self.estimates)

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the bootstrap distribution.

        This method formats the mean and size of the bootstrap distribution for display.

        Returns:
            str: A formatted string representing the bootstrap distribution.
        """
        mean = self.summarize().mean
        size = len(self)
        return f"BootstrapDistribution({mean=:}, {size=:})"

    def summarize(self, level: float = 0.87) -> BootstrapSummary:
        """
        Returns a `BootstrapSummary` object.

        This method is a wrapper for `BootstrapSummary.from_estimates`.

        Args:
            level (float): The desired level for the credible interval
                (must be between 0 and 1).

        Returns:
            BootstrapSummary: the summary object.

        Raises:
            ValueError: If the `level` is not between 0 and 1.
        """
        return BootstrapSummary.from_estimates(self.estimates, level=level)
