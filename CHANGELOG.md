# Changelog

All notable changes to this project are documented here.

<!-- ## [Unreleased] -->

## [0.1.1] - 2025-11-03

### Added
- `bbstat.evaluate.BootstrapSummary`: summarises a bootstraped distribution of estimates.
- `bbstat.evaluate.BootstrapDistribution`: container for a bootstraped distribution of estimates as replacement for `bbstat.evaluate.BootstrapResult`.
- `bbstat.plot.plot` uses a new optional parameter `precision` to control rounding.

### Changed
- Moved `bbstat.evaluate.BootstrapResult.ndigits` to `bbstat.utils.get_precision_for_rounding` and changed parameter from interval bounds to width.
- Moved `bbstat.evaluate.credibility_interval` to `bbstat.utils.compute_credibility_interval`.
- Moved `bbstat.evaluate.BootstrapResult.plot` to `bbstat.plot.plot`.
- Renamed all instances "credibility" -> "credible" and "coverage" -> "level".
- `bbstat.bootstrap.bootstrap` returns `bbstat.evaluate.BootstrapDistribution`.
- `bbstat.plot.plot` expects `boostrap_distribution: bbstat.evaluate.BootstrapDistribution` as as first and `level: float` as (no longer optional) second parameter.

### Removed
- Removed obsolete `bbstat.evaluate.BootstrapResult`

## [0.1.0] - 2025-10-27
Core logic and selected statistic functions.

### Added
- Bayesian bootstrapping function `bbstat.bootstrap`.
- Dirichlet weights generator `bbstat.resample`.
- Bootstrap results container `bbstat.BootstrapResult` and credibility interval calculator `bbstat.credibility_interval`.
- Module with initial set of weighted univariate and bivariate statistic functions `bbstat.statistics` (entropy, eta_square_dependency, log_odds, mean, median, mutual_information, pearson_dependency, percentile, probability, quantile, self_information, spearman_dependency, std, sum, variance).
- Registry to look up included statistics by name.
- Documentation, tests, and packaging.
