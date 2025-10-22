"""Evaluation utilities for summarizing bootstrap resampling results.

This module provides functions and data structures for interpreting and summarizing
the output of Bayesian bootstrap resampling procedures. It includes tools to compute
credibility intervals for statistical estimates and to encapsulate the results of a
bootstrap analysis in a convenient data class.

Main Features:
    - `credibility_interval`: Computes a credibility interval from a set of estimates.
    - `BootstrapResult`: A data class that holds bootstrap estimates, computes the mean,
      and automatically evaluates the credibility interval.

Example:
    ```python
    from bbstat.evaluate import BootstrapResult
    result = BootstrapResult(estimates=np.array([5.0, 2.3, 2.9]), coverage=0.95)
    print(result)  # => BootstrapResult(mean=3.4, ci=(2.3, 4.9), coverage=0.95, n_boot=3)
    ```

Notes:
    - The credibility interval is calculated using quantiles of the empirical distribution
      of bootstrap estimates.
    - This module is designed to be used alongside the `bootstrap` and `resample` modules
      to provide complete statistical summaries of resampled data.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .statistics import FArray

__all__ = [
    "BootstrapResult",
    "credibility_interval",
]


def credibility_interval(
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
        credibility_interval(estimates, 0.6)  # => (2.06, 3.6)
        ```
    """
    if estimates.ndim != 1:
        raise ValueError(f"Invalid parameter {estimates.ndim=:}: must be 1D array.")
    if coverage <= 0 or coverage >= 1:
        raise ValueError(f"Invalid parameter {coverage=:}: must be within (0, 1).")
    edge = (1.0 - coverage) / 2.0
    return tuple(np.quantile(estimates, [edge, 1.0 - edge]).tolist())


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
        self.ci = credibility_interval(
            estimates=self.estimates,
            coverage=self.coverage,
        )
        self.n_boot = len(self.estimates)

    @staticmethod
    def ndigits(ci: Tuple[float, float]) -> int:
        """
        Returns number of digits for rounding.

        This method computes the number of digits for rounding mean and credibility
        interval values for better readability. If the credibility interval has width
        zero, we round to zero digits. Otherwise, we take one minus the floored order
        of magnitude of the width.

        Args:
            ci (Tuple[float, float]): The lower and upper bounds of the credibility
                interval.

        Returns:
            int: The number of digits for readable rounding.
        """
        lo, hi = ci
        width = hi - lo
        if width == 0:
            return 0
        return int(1 - math.floor(math.log10(abs(width))))

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the bootstrap result.

        This method formats the mean, credibility interval, coverage, and the
        number of bootstrap resamples for display.

        Returns:
            str: A formatted string representing the bootstrap result.
        """
        ndigits = self.ndigits(self.ci)
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
        return credibility_interval(estimates=self.estimates, coverage=coverage)

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        coverage: Optional[float] = None,
        n_grid: int = 200,
        label: Optional[str] = None,
    ) -> plt.Axes:
        """
        Plot the kernel density estimate (KDE) of bootstrap estimates with
        credibility interval shading and a vertical line at the mean.

        If an axis is provided, the plot is drawn on it; otherwise, a new figure and axis are created.
        Displays a shaded credibility interval and labels the plot with a formatted mean
        and interval. If no axis is provided, the figure further is annotated with a title and ylabel,
        ylim[0] positioned at zero, the legend is set, and a tight layout applied.

        Args:
            ax (plt.Axes, optional): Matplotlib axis to draw the plot on. If None, a new axis is created.
            coverage (float, optional): Credibility interval coverage (e.g., 0.95 for 95% CI).
                If None, uses the default stored in `self.ci`. Default is None.
            n_grid (int): Number of grid points to use for evaluating the KDE, default is 200.
            label (str, optional): Optional label for the line. If provided, the label is
                extended to include the mean and credibility interval.

        Returns:
            plt.Axes: The axis object containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.5))
        else:
            fig = None

        if coverage is None:
            ci = self.ci
            coverage = self.coverage
        else:
            ci = self.credibility_interval(coverage=coverage)
        lo, hi = ci

        ndigits = self.ndigits(ci=ci)
        param_str = (
            f"{round(self.mean, ndigits)} ({round(lo, ndigits)}, {round(hi, ndigits)})"
        )

        if label is not None:
            param_str = f"{label}={param_str}"

        p = gaussian_kde(self.estimates)

        x_grid = np.linspace(self.estimates.min(), self.estimates.max(), n_grid)
        within_ci = np.logical_and(x_grid >= lo, x_grid <= hi)
        y_grid = p(x_grid)
        y_mean = p([self.mean]).item()

        (line,) = ax.plot(x_grid, y_grid, label=param_str)
        color = line.get_color()

        ax.fill_between(
            x_grid[within_ci],
            0,
            y_grid[within_ci],
            facecolor=color,
            alpha=0.5,
        )
        ax.plot([self.mean] * 2, [0, y_mean], "--", color=color)
        ax.plot([self.mean], [y_mean], "o", color=color)

        if fig is not None:
            ax.set_title(
                f"Bayesian bootstrap  •  {self.n_boot} resamples, {coverage * 100:.0f}% CI"
            )
            ax.set_ylim(0, ax.get_ylim()[1])
            ax.set_ylabel("Distribution of estimates")
            ax.legend()
            fig.tight_layout()
        return ax
