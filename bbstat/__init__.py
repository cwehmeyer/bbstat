"""
bbstat: Bayesian Bootstrap Utilities

This package provides tools for performing and evaluating the Bayesian bootstrap,
a resampling method based on the Bayesian interpretation of uncertainty.

Main Features:
--------------
- `bootstrap`: Run the Bayesian bootstrap on compatible data structures.
- `BootstrapResult`: Analyze bootstrap outcomes with mean estimates and credibility intervals.
- `credibility_interval`: Compute one-dimensional credibility intervals from samples.
- `resample`: Generate weighted samples using the Dirichlet distribution.

Supported Statistic Functions:
------------------------------
Custom statistic functions must accept the signature:
    (data: ..., weights: NDarray[np.floating], **kwargs) -> float

Compatible examples in bbstat.statistics include:
- `compute_weighted_mean`: Weighted mean
- `compute_weighted_sum`: Weighted sum
- `compute_weighted_quantile`: Weighted quantile estimate
- `compute_weighted_pearson_dependency`: Weighted Pearson correlation
- `compute_weighted_spearman_dependency`: Weighted Spearman correlation
- `compute_weighted_eta_square_dependency`: Weighted eta-squared for categorical group differences

You can register your own functions using `statistics.registry.add("your_name")`.

Modules:
--------
- `bootstrap`: Core logic for Bayesian bootstrap
- `evaluate`: Tools for summarizing bootstrap results
- `resample`: Weighted resampling function
- `statistics`: Registry and built-in statistic functions
"""

from bbstat.evaluate import BootstrapResult, credibility_interval
from bbstat.resample import resample

from . import statistics
from .bootstrap import bootstrap

__all__ = [
    "bootstrap",
    "BootstrapResult",
    "credibility_interval",
    "resample",
    "statistics",
]
