# bbstat

Welcome to **bbstat**, a lightweight library for Bayesian bootstrapping and statistical evaluation.

## Features

- Bayesian bootstrap resampling
- Compute weighted statistics
- Evaluate uncertainty via credible intervals
- Easy-to-use and extensible

## Installation

Installation from PyPi:

```bash
pip install bbstat
```

Installation from GitHub source code:

```bash
git clone https://github.com/cwehmeyer/bbstat.git
cd bbstat
pip install .
```

### Optional Extras

This package includes optional dependencies for development, testing, and documentation. To install them from GitHub source:

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
