# bbstat

[![PyPI version](https://img.shields.io/pypi/v/bbstat.svg)](https://pypi.org/project/bbstat/)
[![Python Versions](https://img.shields.io/pypi/pyversions/bbstat.svg)](https://pypi.org/project/bbstat/)
[![CodeQL](https://github.com/cwehmeyer/bbstat/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/cwehmeyer/bbstat/actions/workflows/github-code-scanning/codeql)
[![CI](https://github.com/cwehmeyer/bbstat/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/cwehmeyer/bbstat/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/cwehmeyer/bbstat/branch/main/graph/badge.svg?token=V3QV2DFJ9W)](https://codecov.io/gh/cwehmeyer/bbstat)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://cwehmeyer.github.io/bbstat/)

A lightweight library for Bayesian bootstrapping and statistical evaluation designed for learning, experimentation, and exploring Bayesian nonparametric ideas.

The Bayesian bootstrap (Rubin, 1981) is a simple nonparametric Bayesian method for estimating uncertainty in statistics without assuming a likelihood model. It replaces resampling with random Dirichlet-distributed weights on the observed data, producing a posterior-like distribution for any statistic (mean, quantile, regression, etc.). Results reflect uncertainty in the weights (not in unobserved data) and are asymptotically similar to the classical bootstrap. Assumes i.i.d. data; results may be overconfident if the sample is small or unrepresentative.

This package implements the core logic of Bayesian bootstrapping in Python, along with a few weighted statistic functions, as a way to learn and experiment with Bayesian nonparametric ideas. It's meant as an educational and exploratory project rather than a production-ready library, but may be useful for understanding or demonstrating how Bayesian bootstrap inference works in practice.

## Why use this package?

- Learn and experiment with Bayesian bootstrap inference in Python
- Quickly compute posterior-like uncertainty intervals for arbitrary statistics
- Extend easily with your own weighted statistic functions

## Installation

- From PyPI:

```bash
pip install bbstat
```

- From GitHub source code:

```bash
git clone https://github.com/cwehmeyer/bbstat.git
cd bbstat
pip install .
```

## Quickstart

```python
import numpy as np
from bbstat import bootstrap

# Data preparation: simulated income for a small population (e.g., a survey of 25 people)
income = np.array([
    24_000, 26_000, 28_000, 30_000, 32_000,
    35_000, 36_000, 38_000, 40_000, 41_000,
    45_000, 48_000, 50_000, 52_000, 54_000,
    58_000, 60_000, 62_000, 65_000, 68_000,
    70_000, 75_000, 80_000, 90_000, 100_000,
], dtype=np.float64)

# Direct estimate of mean income
print(np.mean(income))  # => 52280.0

# Bootstrapped distribution of the mean income.
distribution = bootstrap(data=income, statistic_fn="mean", seed=1)
print(distribution)  # => BootstrapDistribution(mean=52263.8..., size=1000)

# Summarize the bootstrapped distribution of the mean income.
summary = distribution.summarize(level=0.87)
print(summary)  # => BootstrapSummary(mean=52263.8..., ci_low=46566.8..., ci_high=58453.6..., level=0.87)
print(summary.round())  # => BootstrapSummary(mean=52000.0, ci_low=47000.0, ci_high=58000.0, level=0.87)
```

## API Overview

### `bootstrap(data, statistic_fn, n_boot=1000, ...)`

Performs Bayesian bootstrapping on `data` using the given statistic.

- `data`: 1D NumPy array, or tuple/list thereof
- `statistic_fn`: string or callable (e.g., `"mean"`, `"median"`, or custom function)
- `n_boot`: number of bootstrap samples
- `seed`: random seed (optional)
- `blocksize`: number of resamples to allocate in one block
- `fn_kwargs`: optional dictionary with parameters for `statistic_fn`

**Parameters**

- `data`: 1D NumPy array, or tuple/list of arrays
- `statistic_fn`: string or callable (e.g. `"mean"`, `"median"`, or custom function)
- `n_boot`: number of bootstrap samples
- `seed`: random seed (optional)
- `blocksize`: number of resamples processed per block
- `fn_kwargs`: optional dict of extra parameters for `statistic_fn`

**Returns**

A `BootstrapDistribution` object with:

- `.estimates`: array of bootstrapped statistic values
- `.summarize(level, precision)`: returns a `BootstrapSummary` with `mean`, `ci_low`, `ci_high`,
  and `level`; rounded if `precision` is integer-valued or `"auto"`

### Weighted statistic functions included

The module `bbstat.statistics` includes several univariate and bivariate weighted statistics, such as:

- `"mean"` – `compute_weighted_mean(data, weights)`
- `"median"` – `compute_weighted_median(data, weights)`
- `"quantile"` / `"percentile"`
- `"variance"` / `"std"` / `"sum"`
- `"entropy"` / `"log_odds"` / `"probability"` / `"self_information"`
- `"pearson_dependence"` / `"spearman_dependence"`
- `"eta_square_dependency"` / `"mutual_information"`

You can also supply your own functions following this pattern:

```python
def custom_statistic(data, weights, **kwargs) -> float:
    ...
```

where:

- `data`: 1D NumPy array or tuple/list of 1D arrays
- `weights`: 1D NumPy array of non-negative values summing to 1
- `**kwargs`: optional keyword arguments passed by `fn_kwargs`

If you want to use your own custom functions, please adhere to this pattern.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
