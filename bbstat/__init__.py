"""bbstat: Bayesian Bootstrap Utilities

This package provides tools for performing and evaluating the Bayesian bootstrap,
a resampling method based on the Bayesian interpretation of uncertainty.

Main Features:
    - `bootstrap`: Run the Bayesian bootstrap on compatible data structures.
    - `BootstrapResult`: Analyze bootstrap outcomes with mean estimates and credibility intervals.
    - `resample`: Generate weighted samples using the Dirichlet distribution.
    - `statistics`: Collection of built-in weighted statistics.

Supported Statistic Functions:
    Custom statistic functions must accept the signature:

    `(data: ..., weights: numpy.typing.NDarray[numpy.floating], **kwargs) -> float`

    Compatible examples in bbstat.statistics include:

    - `compute_weighted_entropy`: Weighted entropy
    - `compute_weighted_eta_square_dependency`: Weighted eta-squared for categorical group differences
    - `compute_weighted_log_odds`: Weighted log-odds of a selected state
    - `compute_weighted_mean`: Weighted mean estimate
    - `compute_weighted_median`: Weighted median estimate
    - `compute_weighted_mutual_information`: Weighted mutual information
    - `compute_weighted_pearson_dependency`: Weighted Pearson correlation
    - `compute_weighted_percentile`: Weighted percentile estimate
    - `compute_weighted_probability`: Weighted probability of a selected state
    - `compute_weighted_quantile`: Weighted quantile estimate
    - `compute_weighted_self_information`: Weighted self-information of a selected state
    - `compute_weighted_spearman_dependency`: Weighted Spearman correlation
    - `compute_weighted_std`: Weighted standard deviation estimate
    - `compute_weighted_sum`: Weighted sum estimate
    - `compute_weighted_variance`: Weighted variance estimate

Modules:
    - `bootstrap`: Core logic for Bayesian bootstrap
    - `evaluate`: Tools for summarizing bootstrap results
    - `plot`: Tool for visualizing bootstrap results
    - `registry`: Registry for built-in statistic functions
    - `resample`: Weighted resampling function
    - `statistics`: Built-in statistic functions
    - `utils`: Utility functions
"""

from bbstat.evaluate import BootstrapResult
from bbstat.resample import resample

from . import statistics, utils
from ._version import version as __version__  # type: ignore[import-untyped]
from .bootstrap import bootstrap
from .plot import plot

__all__ = [
    "__version__",
    "bootstrap",
    "BootstrapResult",
    "plot",
    "resample",
    "statistics",
    "utils",
]
