# bbstat

Welcome to **bbstat**, a lightweight library for Bayesian bootstrapping and statistical evaluation.

## Features

- Bayesian bootstrap resampling
- Compute weighted statistics
- Evaluate uncertainty via credibility intervals
- Easy-to-use and extensible

## Installation

Installation from GitHub source code:

```bash
git clone https://github.com/cwehmeyer/bbstat.git
cd bbstat
pip install .
```

### Optional Extras

This package includes optional dependencies for development, testing, and documentation. To install them:

- For development:

```bash
pip install '.[dev]'
```

- For testing:

```bash
pip install '.[test]'
```

- For documentation:

```bash
pip install '.[docs]'
```

## Getting started

```python
import numpy as np
import bbstat

# Data preparation: simulated income for a small population (e.g., a survey of 25 people)
income = np.array([
    24_000, 26_000, 28_000, 30_000, 32_000,
    35_000, 36_000, 38_000, 40_000, 41_000,
    45_000, 48_000, 50_000, 52_000, 54_000,
    58_000, 60_000, 62_000, 65_000, 68_000,
    70_000, 75_000, 80_000, 90_000, 100_000
], dtype=np.float64)

# Direct estimate of mean income
print(np.mean(income))  # => 52280.0

# Bootstrapped estimate of mean income with 95% credibility interval
result = bootstrap(data=income, statistic_fn="median", coverage=0.87, seed=1)
print(result)  # => BootstrapResult(mean=50000.0, ci=(40000.0, 59000.0), coverage=0.87, n_boot=1000)
```
