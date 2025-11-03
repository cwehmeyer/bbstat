"""Plotting utility for bootstrap resampling results.

This module provides a function for visually interpreting and summarizing
the output of Bayesian bootstrap resampling procedures.

Main Features:
    - `plot`: Visualizes the result of a bootstrap resampling procedure.

Notes:
    - The credible interval is calculated using quantiles of the empirical distribution
      of bootstrap estimates.
    - This module is designed to be used alongside the `evaluate` module to provide complete
      statistical summaries of resampled data.
"""

from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .evaluate import BootstrapDistribution


__all__ = ["plot"]


def plot(
    bootstrap_distribution: BootstrapDistribution,
    level: float,
    *,
    ax: Optional[plt.Axes] = None,
    n_grid: int = 200,
    label: Optional[str] = None,
    precision: Optional[Union[int, Literal["auto"]]] = None,
) -> plt.Axes:
    """
    Plot the kernel density estimate (KDE) of bootstrap estimates with
    credible interval shading and a vertical line at the mean.

    If an axis is provided, the plot is drawn on it; otherwise, a new figure and axis are created.
    Displays a shaded credible interval and labels the plot with a formatted mean
    and credible interval. If no axis is provided, the figure further is annotated with a title and ylabel,
    ylim[0] positioned at zero, the legend is set, and a tight layout applied.

    Args:
        bootstrap_distribution (BootstrapDistribution): The result of a bootstrap resampling procedure.
        level (float): Credible interval level (e.g., 0.95 for 95% CI).
        ax (plt.Axes, optional): Matplotlib axis to draw the plot on. If None, a new axis is created.
        n_grid (int): Number of grid points to use for evaluating the KDE, default is 200.
        label (str, optional): Optional label for the line. If provided, the label is
            extended to include the mean and credible interval.
        precision (int or "auto" or None, optional): Optional precision for rounding the summary
            values (mean and credible interval). If None (default), no rounding is done; if "auto",
            the precision is computed from the width of the credible interval; if integer, we round to
            this many digits.

    Returns:
        plt.Axes: The axis object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = None

    summary = bootstrap_distribution.summarize(level)

    if precision is not None:
        if precision == "auto":
            summary = summary.round()
        else:
            summary = summary.round(precision)

    param_str = f"{summary.mean} ({summary.ci_low}, {summary.ci_high})"

    if label is not None:
        param_str = f"{label}={param_str}"

    p = gaussian_kde(bootstrap_distribution.estimates)

    x_grid = np.linspace(
        bootstrap_distribution.estimates.min(), bootstrap_distribution.estimates.max(), n_grid
    )
    within_ci = np.logical_and(x_grid >= summary.ci_low, x_grid <= summary.ci_high)
    y_grid = p(x_grid)
    y_mean = p([summary.mean]).item()

    (line,) = ax.plot(x_grid, y_grid, label=param_str)
    color = line.get_color()

    ax.fill_between(
        x_grid[within_ci],
        0,
        y_grid[within_ci],
        facecolor=color,
        alpha=0.5,
    )
    ax.plot([summary.mean] * 2, [0, y_mean], "--", color=color)
    ax.plot([summary.mean], [y_mean], "o", color=color)

    if fig is not None:
        ax.set_title(
            f"Bayesian bootstrap  •  {len(bootstrap_distribution)} resamples, {level * 100:.0f}% CI"
        )
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_ylabel("Distribution of estimates")
        ax.legend()
        fig.tight_layout()
    return ax
