# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Changed
- Moved `bbstat.evaluate.BootstrapResult.ndigits` to `bbstat.utils.get_precision_from_credibility_interval`.
- Moved `bbstat.evaluate.credibility_interval` to `bbstat.utils.compute_credibility_interval`.

## [0.1.0] - 2025-10-27
Core logic and selected statistic functions.

### Added
- Bayesian bootstrapping function `bbstat.bootstrap`.
- Dirichlet weights generator `bbstat.resample`.
- Bootstrap results container `bbstat.BootstrapResult` and credibility interval calculator `bbstat.credibility_interval`.
- Module with initial set of weighted univariate and bivariate statistic functions `bbstat.statistics` (entropy, eta_square_dependency, log_odds, mean, median, mutual_information, pearson_dependency, percentile, probability, quantile, self_information, spearman_dependency, std, sum, variance).
- Registry to look up included statistics by name.
- Documentation, tests, and packaging.
