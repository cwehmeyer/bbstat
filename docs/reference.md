# API Reference

This section documents the public API of the `bbstat` package.

---

## `bbstat` Package

::: bbstat
    options:
      show_source: false
      members: false

---

## `bootstrap` Module

::: bbstat.bootstrap
    options:
      show_source: true
      members:
        - bootstrap

---

## `evaluate` Module

::: bbstat.evaluate
    options:
      show_source: true
      members:
        - BootstrapResult
        - credibility_interval

---

## `registry` Module

::: bbstat.registry
    options:
      show_source: true
      members:
        - StatisticFunction
        - get_statistic_fn

---

## `resample` Module

::: bbstat.resample
    options:
      show_source: true
      members:
        - resample

---

## `statistics` Module

::: bbstat.statistics
    options:
      show_source: true
      members:
        - FArray
        - IArray
        - FFArray
        - IFArray
        - compute_weighted_aggregate
        - compute_weighted_entropy
        - compute_weighted_eta_square_dependency
        - compute_weighted_mean
        - compute_weighted_median
        - compute_weighted_pearson_dependency
        - compute_weighted_percentile
        - compute_weighted_probability
        - compute_weighted_quantile
        - compute_weighted_spearman_dependency
        - compute_weighted_std
        - compute_weighted_sum
        - compute_weighted_variance
