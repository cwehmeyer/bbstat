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
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

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

        ndigits = get_precision_from_credibility_interval(ci)
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
