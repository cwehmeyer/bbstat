# Getting Started

This guide shows how to use `bbstat` to perform Bayesian bootstrapping with both built-in and custom statistics.

We'll start with a quick example on univariate data, move on to a bivariate case, and then show how to write your own weighted statistic. The goal is to help you see how Bayesian bootstrapping works in practice.

## Installation

You can install bbstat from PyPI:

```bash
pip install bbstat
```

Then import what you need:

```python
import numpy as np
from bbstat import bootstrap
```

## Bootstrapping a simple statistic

Let's start with something familiar: estimating the mean of a small dataset. We'll use the Bayesian bootstrap to quantify uncertainty in that mean.

```python
# Sample data: daily coffee consumption (in cups) from a small survey
coffee = np.array([2.0, 3.0, 1.5, 2.5, 3.0, 2.0, 4.0])

# Run the Bayesian bootstrap with 2000 Dirichlet-weighted replicates
distribution = bootstrap(data=coffee, statistic_fn="mean", n_boot=2000, seed=1)

# Summarize the distribution as a posterior mean and 95% credible interval
summary = distribution.summarize(level=0.95)
print(summary)
# BootstrapSummary(mean=2.583..., ci_low=2.057..., ci_high=3.159..., level=0.95)
```

If you'd like cleaner, human-readable output, the `BootstrapSummary.round()` method can automatically round values to a sensible precision based on the width of the credible interval:

```python
print(summary.round())
# BootstrapSummary(mean=2.6, ci_low=2.1, ci_high=3.2, level=0.95)
```

Additionally, you can specify the precision in `BootstrapDistribution.summarize`:

```python
summary = distribution.summarize(level=0.95, precision="auto")
print(summary)
# BootstrapSummary(mean=2.6, ci_low=2.1, ci_high=3.2, level=0.95)
```

Here the mean estimate is about 2.6 cups per day, with a 95% credible interval of roughly [2.1, 3.2]. The uncertainty reflects variation in the weights each sample could have in the population, not in resampled data points.

## Bootstrapping a quantile

You can use any of the built-in weighted statistics the same way. For example, let's estimate the 90th percentile of the same dataset:

```python
distribution = bootstrap(
    data=coffee,
    statistic_fn="quantile",
    fn_kwargs={"quantile": 0.9},
    seed=1,
)

summary = distribution.summarize(level=0.95)
print(summary)
# BootstrapSummary(mean=3.28, ci_low=2.85, ci_high=3.81, level=0.95)
```

The bootstrapped 0.9 quantile is around 3.3 cups, meaning that about 90% of coffee drinkers in this sample consume 3.3 or fewer cups per day.

## Bivariate example: dependence between variables

For bivariate data, bbstat includes functions such as `"pearson_dependency"` (weighted correlation) and `"mutual_information"` (a nonlinear dependence measure).

Let's look at the relationship between study time and exam score:

```python
# Simulated data: study hours vs exam scores
study_hours = np.array([2, 3, 4, 5, 6, 8, 9])
exam_scores = np.array([60, 65, 70, 72, 78, 85, 90])

data = (study_hours, exam_scores)

# Weighted Pearson correlation via Bayesian bootstrapping
distribution = bootstrap(data=data, statistic_fn="pearson_dependency", n_boot=2000, seed=1)
summary = distribution.summarize(level=0.95, precision="auto")
print(summary)
# BootstrapSummary(mean=0.9969, ci_low=0.9911, ci_high=0.9992, level=0.95)
```

This shows a strong positive correlation, and the credible interval indicates high confidence that the true correlation is above 0.99. You could switch to "mutual_information" to estimate a nonlinear dependency instead.

## Writing your own weighted statistic

Defining a custom statistic is simple. All functions used with bootstrap() must follow this signature:

```python
def custom_statistic(data, weights, **kwargs) -> float:
    ...
```

Here's an example that implements a **weighted geometric mean**, which is not (yet) included among the built-ins but demonstrates how to use the weights properly:

For a set of positive numbers \(x_1, x_2, \dots, x_n > 0\) with associated weights
\(w_1, w_2, \dots, w_n\) such that \(w_i \ge 0\) and \(\sum_{i=1}^n w_i = 1\),
the **weighted geometric mean** is defined as:

\[
\text{GM}_w = \prod_{i=1}^{n} x_i^{w_i} = \exp\Bigg( \sum_{i=1}^{n} w_i \ln x_i \Bigg)
\]

In the Bayesian bootstrap, the weights \(w_i\) are drawn from a Dirichlet distribution:

\[
(w_1, \dots, w_n) \sim \text{Dirichlet}(\alpha_1=1, \dots, \alpha_n=1)
\]

Each bootstrap replicate computes:

\[
\text{GM}_\text{replicate} = \exp\Bigg( \sum_{i=1}^{n} w_i \ln x_i \Bigg)
\]

Repeating this for many replicates produces a posterior-like distribution of the geometric mean.

```python
def weighted_geometric_mean(data, weights):
    """Compute the weighted geometric mean."""
    data = np.asarray(data)
    weights = np.asarray(weights)
    # Avoid log(0): require positive data
    if np.any(data <= 0):
        raise ValueError("Geometric mean requires positive data.")
    log_mean = np.sum(weights * np.log(data))
    return np.exp(log_mean)


data = np.array([1.2, 1.5, 2.0, 2.8, 3.1])
distribution = bootstrap(data=data, statistic_fn=weighted_geometric_mean, n_boot=1500, seed=1)
summary = distribution.summarize(precision="auto")
print(summary)
# BootstrapSummary(mean=2.01, ci_low=1.58, ci_high=2.48, level=0.87)
```

The same pattern applies if your statistic takes multiple arrays (e.g., (x, y)). The function receives the data and weights, computes its result, and returns a single float.

## Common questions and pitfalls
- **Why are the credible intervals sometimes narrow?**
  Bayesian bootstrapping assumes that the observed data already represent the full population support. Uncertainty is only about how much weight each observation should get, not about unseen data. If the sample is small or has heavy tails, results can appear overconfident.
- **Can I get negative weights or resampled data?**
  No. Weights are drawn from a uniform Dirichlet distribution, so they're always non-negative and sum to one. This approach replaces the random resampling in the classical bootstrap, rather than supplementing it.
- **What if my statistic ignores the weights?**
  Then it is not a Bayesian bootstrap anymore and you are just re-evaluating the same statistic repeatedly. Always make sure your custom statistic uses the provided weights.
- **What if my data contain zeros or negative values?**
  That's fine for most statistics, but not all (the geometric mean above is a case in point). Handle such cases carefully or filter the data before applying those statistics.
- **Can I use bbstat for regression or multivariate models?**
  Yes, as long as your statistic can be written as a weighted function of the data. For example, a weighted regression slope or a loss function summary. The Bayesian bootstrap does not assume any specific model form.
- **How does rounding work in summarize()?**
  When you call `BootstrapSummary.round()`, the method automatically picks a decimal precision suitable for the width of the credible interval so that the displayed digits reflect the level of uncertainty. You can also set a fixed precision manually if you prefer.
