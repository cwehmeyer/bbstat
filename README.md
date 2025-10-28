# bbstat

[![CI](https://github.com/cwehmeyer/bbstat/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/cwehmeyer/bbstat/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/cwehmeyer/bbstat/branch/main/graph/badge.svg?token=V3QV2DFJ9W)](https://codecov.io/gh/cwehmeyer/bbstat)

A lightweight library for Bayesian bootstrapping and statistical evaluation.

## Installation

### From PyPI:

```bash
pip install bbstat
```

### From GitHub source code:

```bash
git clone https://github.com/cwehmeyer/bbstat.git
cd bbstat
pip install .
```

## Quickstart

```python
import numpy as np
import bbstat

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

# Bootstrapped estimate of mean income with 87% credibility interval
result = bootstrap(data=income, statistic_fn="median", coverage=0.87, seed=1)
print(result)  # => BootstrapResult(mean=50000.0, ci=(40000.0, 59000.0), coverage=0.87, n_boot=1000)
```

## API Overview

### `bootstrap(data, statistic_fn, coverage=0.87, n_boot=1000, ...)`

Performs Bayesian bootstrapping on input `data` using the given statistic.

- `data`: 1D NumPy array, or tuple/list thereof
- `statistic_fn`: string or callable (e.g., `"mean"`, `"median"`, or custom function)
- `coverage`: credibility interval (default 0.87)
- `n_boot`: number of bootstrap samples
- `seed`: random seed (optional)
- `blocksize`: number of resamples to allocate in one block
- `fn_kwargs`: optional dictionary with parameters for `statistic_fn`

Returns a `BootstrapResult` with:
- `.mean`: estimated statistic value
- `.ci`: tuple representing lower and upper bounds of the credibility interval
- `.coverage`: credibility level used
- `.n_boot`: number of bootstraps performed
- `.estimates`: array of statistic values computed across the bootstrapped posterior samples

### Weighted statistic functions included

The module bbstat.statistics provides a number univariate and bivariate weighted statistics:
- `"entropy"`: `bbstat.statistics.compute_weighted_entropy(data, weights)`
- `"eta_square_dependency"`: `bbstat.statistics.compute_weighted_eta_square_dependency(data, weights)`
- `"log_odds"`: `bbstat.statistics.compute_weighted_log_odds(data, weights, state: int)`
- `"mean"`: `bbstat.statistics.compute_weighted_mean(data, weights)`
- `"median"`: `bbstat.statistics.compute_weighted_median(data, weights)`
- `"mutual_information"`: `bbstat.statistics.compute_weighted_mutual_information(data, weights)`
- `"pearson_dependence"`: `bbstat.statistics.compute_weighted_pearson_dependence(data, weights, ddof: int = 0)`
- `"percentile"`: `bbstat.statistics.compute_weighted_percentile(data, weights, percentile: float)`
- `"probability"`: `bbstat.statistics.compute_weighted_probability(data, weights, state: int)`
- `"quantile"`: `bbstat.statistics.compute_weighted_quantile(data, weights, quantile: float)`
- `"self_information"`: `bbstat.statistics.compute_weighted_self_information(data, weights, state: int)`
- `"spearman_depedence"`: `bbstat.statistics.compute_weighted_spearman_depedence(data, weights, ddof: int = 0)`
- `"std"`: `bbstat.statistics.compute_weighted_std(data, weights, ddof: int = 0)`
- `"sum"`: `bbstat.statistics.compute_weighted_sum(data, weights)`
- `"variance"`: `bbstat.statistics.compute_weighted_variance(data, weights, ddof: int = 0)`

If you want to use your own custom functions, please adhere to this pattern

```python
def custom_statistics(data, weights, *, **kwargs) -> float
```

where `data` is
- a 1D numpy array of length `n_data` or
- a tuple/list of 1D numpy arrays, each of length `n_data`

and `weights` is a 1D numpy array of length `n_data`, with non-negative elements that sum up to one. The function may also take additional parameters which can be supplied via `**kwargs`.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
