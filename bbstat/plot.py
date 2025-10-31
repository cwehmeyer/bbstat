"""Plotting utility for bootstrap resampling results.

This module provides a function for visually interpreting and summarizing
the output of Bayesian bootstrap resampling procedures.

Main Features:
    - `plot`: Visualizes the result of a bootstrap resampling procedure.

Notes:
    - The credibility interval is calculated using quantiles of the empirical distribution
      of bootstrap estimates.
    - This module is designed to be used alongside the `evaluate` module to provide complete
      statistical summaries of resampled data.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .evaluate import BootstrapResult
from .utils import get_precision_from_credibility_interval


__all__ = ["plot"]


def plot(
    bootstrap_result: BootstrapResult,
    *,
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
        bootstrap_result (BootstrapResult): The result of a bootstrap resampling procedure.
        ax (plt.Axes, optional): Matplotlib axis to draw the plot on. If None, a new axis is created.
        coverage (float, optional): Credibility interval coverage (e.g., 0.95 for 95% CI).
            If None, uses the default stored in `bootstrap_result.ci`. Default is None.
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
        ci = bootstrap_result.ci
        coverage = bootstrap_result.coverage
    else:
        ci = bootstrap_result.credibility_interval(coverage=coverage)
    lo, hi = ci

    ndigits = get_precision_from_credibility_interval(ci)
    param_str = f"{round(bootstrap_result.mean, ndigits)} ({round(lo, ndigits)}, {round(hi, ndigits)})"

    if label is not None:
        param_str = f"{label}={param_str}"

    p = gaussian_kde(bootstrap_result.estimates)

    x_grid = np.linspace(
        bootstrap_result.estimates.min(), bootstrap_result.estimates.max(), n_grid
    )
    within_ci = np.logical_and(x_grid >= lo, x_grid <= hi)
    y_grid = p(x_grid)
    y_mean = p([bootstrap_result.mean]).item()

    (line,) = ax.plot(x_grid, y_grid, label=param_str)
    color = line.get_color()

    ax.fill_between(
        x_grid[within_ci],
        0,
        y_grid[within_ci],
        facecolor=color,
        alpha=0.5,
    )
    ax.plot([bootstrap_result.mean] * 2, [0, y_mean], "--", color=color)
    ax.plot([bootstrap_result.mean], [y_mean], "o", color=color)

    if fig is not None:
        ax.set_title(
            f"Bayesian bootstrap  •  {bootstrap_result.n_boot} resamples, {coverage * 100:.0f}% CI"
        )
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_ylabel("Distribution of estimates")
        ax.legend()
        fig.tight_layout()
    return ax
