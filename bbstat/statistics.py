"""Type definitions and weighted statistical functions for use in bootstrap resampling and analysis.

This module defines types for statistical functions that operate on weighted data,
particularly in the context of Bayesian bootstrap procedures. It provides a collection
of pre-defined weighted statistics (e.g., mean, variance, quantile).

Main Features:
    - Type aliases for data and weights.
    - A library of built-in weighted statistical functions (e.g., mean, std, quantile, etc.)

Type Aliases:
    - `FArray`: Alias for `numpy.typing.NDArray[numpy.floating]`, used for floating-point data and weights.
    - `IArray`: Alias for `numpy.typing.NDArray[numpy.integer]`, used for index arrays.
    - `FFArray`, `IFArray`, `IIArray`: Tuples of data arrays used in bivariate computations.

Built-in Functions:
    - `"compute_weighted_aggregate"`: Weighted dot product, optionally scaled by a factor (internal use only).
    - `"compute_weighted_entropy"`: Weighted entropy.
    - `"compute_weighted_eta_square_dependency"`: Effect size for categorical-continuous variable relationships.
    - `"compute_weighted_mean"`: Weighted arithmetic mean.
    - `"compute_weighted_median"`: Weighted median.
    - `"compute_weighted_mutual_information"`: Weighted mutual information.
    - `"compute_weighted_pearson_dependency"`: Weighted Pearson correlation for two variables.
    - `"compute_weighted_probability"`: Weighted probability of a state.
    - `"compute_weighted_quantile"` / `"compute_weighted_percentile"`: Weighted quantile estimation.
    - `"compute_weighted_spearman_dependency"`: Weighted Spearman correlation.
    - `"compute_weighted_std"`: Weighted standard deviation.
    - `"compute_weighted_sum"`: Weighted sum.
    - `"compute_weighted_variance"`: Weighted variance with optional degrees of freedom correction.

Notes:
    - All functions assume normalized weights (i.e., sum to 1).
    - Functions raise `ValueError` for invalid shapes, mismatched dimensions, or inappropriate input types.
    - This module is intended for use with `bootstrap`, which applies these functions across bootstrap resamples.
"""

import math
from typing import Optional, Tuple, TypeAlias, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

__all__ = [
    "FArray",
    "FFArray",
    "IArray",
    "IFArray",
    "IIArray",
    "compute_weighted_aggregate",
    "compute_weighted_entropy",
    "compute_weighted_eta_square_dependency",
    "compute_weighted_log_odds",
    "compute_weighted_mean",
    "compute_weighted_median",
    "compute_weighted_mutual_information",
    "compute_weighted_pearson_dependency",
    "compute_weighted_percentile",
    "compute_weighted_probability",
    "compute_weighted_quantile",
    "compute_weighted_self_information",
    "compute_weighted_spearman_dependency",
    "compute_weighted_std",
    "compute_weighted_sum",
    "compute_weighted_variance",
]

FArray: TypeAlias = NDArray[np.floating]
IArray: TypeAlias = NDArray[np.integer]
FFArray: TypeAlias = Tuple[FArray, FArray]
IFArray: TypeAlias = Tuple[IArray, FArray]
IIArray: TypeAlias = Tuple[IArray, IArray]


def compute_weighted_aggregate(
    data: FArray,
    weights: FArray,
    *,
    factor: Optional[float] = None,
) -> float:
    """
    Computes a weighted aggregate of the input data.

    This function calculates the dot product of the input `data` and `weights`.
    The function assumes that both `data` and `weights` are 1D arrays of the same
    length and that the weights sum to 1. If a `factor` is provided, the dot product
    is multiplied with it.

    Args:
        data (FArray): A 1D array of numeric values representing the
            data to be aggregated.
        weights (FArray): A 1D array of numeric values representing
            the weights for the data.
        factor (float, optional): A scalar factor to multiply with the computed
            aggregate (default is None).

    Returns:
        float: The computed weighted aggregate, potentially scaled by the `factor`.

    Raises:
        ValueError: If `data` or `weights` are not 1D arrays.
        ValueError: If the shapes of `data` and `weights` do not match.

    Example:
        ```python
        data = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.2, 0.5, 0.3])
        print(compute_weighted_aggregate(data, weights))  # => 2.1
        print(compute_weighted_aggregate(data, weights, factor=1.5))  # => 3.15
        ```

    Notes:
        The weighted aggregate is computed using the dot product between `data` and `weights`.
        The optional `factor` scales the result of this dot product. If no factor is given,
        the aggregation computes the weighted arithmetic mean of the data; if instead the factor
        equals the length of the data array, the aggregation computes the weighted sum.

    """
    validate_array(data=data, weights=weights)
    aggregate = np.dot(weights, data)
    if factor is not None:
        aggregate *= factor
    return aggregate.item()


def compute_weighted_entropy(
    data: IArray,
    weights: FArray,
) -> float:
    """
    Computes a weighted entropy of 1D code data.

    This function calculates the weighted entropy by first computing
    the weighted distribution, dropping the zero elements (contribute
    zero to the following sum), and computing the negative dot product
    between the distribution and log-distribution.

    Args:
        data (IArray): A 1D array of numeric values representing
            the sample data in code format.
        weights (FArray): A 1D array of numeric weights corresponding
            to the data.

    Returns:
        float: The weighted entropy value.

    Raises:
        ValueError: If `data` and `weights` have different shapes or are not 1D.

    Example:
        ```python
        data = np.array([1, 0, 0])
        weights = np.array([0.4, 0.2, 0.4])
        print(compute_weighted_entropy(data, weights))  # => 0.673...
        ```
    """
    validate_array(data=data, weights=weights)
    active_set = get_active_set(data=data)
    distribution = weighted_discrete_distribution(data=active_set, weights=weights)
    distribution = distribution[distribution > 0.0]
    return -np.dot(distribution, np.log(distribution)).item()


def compute_weighted_eta_square_dependency(
    data: IFArray,
    weights: FArray,
) -> float:
    """
    Computes the weighted eta-squared (η²) statistic to assess dependency between
    a categorical and a numerical variable.

    Eta-squared measures the proportion of total variance in the numerical variable
    that is explained by the categorical grouping. It is commonly used in ANOVA-like
    analyses and effect size estimation. The value is bounded between 0 and 1.

    Args:
        data (IFArray): A tuple `(data_cat, data_num)` where:
            - `data_cat` is a 1D array of integer-encoded categorical values.
            - `data_num` is a 1D array of corresponding numeric values.
        weights (FArray): A 1D array of non-negative weights,
            same length as `data_cat`.

    Returns:
        float: Weighted eta-squared value in the range [0, 1], where higher values
        indicate stronger association between the categorical and numeric variable.

    Raises:
        ValueError: If input arrays are not 1D or do not have matching shapes.

    Example:
        ```python
        data_cat = np.array([0, 0, 1, 1])
        data_num = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.array([0.25, 0.25, 0.25, 0.25)
        print(compute_weighted_eta_square_dependency((data_cat, data_num), weights))  # => 0.8
        ```

    Notes:
        - Internally, η² is computed as the ratio of weighted between-group variance
          to the total weighted variance.
        - The statistic is sensitive to group sizes and imbalance in weights.
        - When all group means equal the global mean, η² is 0.
        - When groups are perfectly separated by the numeric variable, η² is 1.
    """
    data_cat, data_num = data
    mean_sample = compute_weighted_mean(data=data_num, weights=weights)
    group_variance_sum = 0.0
    for value in np.unique(data_cat):
        subset = data_cat == value
        group_weight = weights[subset].sum().item()
        mean_group = compute_weighted_aggregate(
            data=data_num[subset],
            weights=weights[subset],
            factor=1.0 / group_weight,
        )
        group_variance_sum += group_weight * (mean_group - mean_sample) ** 2
    return group_variance_sum / compute_weighted_variance(
        data=data_num,
        weights=weights,
        ddof=0,
    )


def compute_weighted_log_odds(
    data: IArray,
    weights: FArray,
    state: int,
) -> float:
    """
    Computes a weighted log-odds of a state within 1D code data.

    This function calculates the weighted probability of a `state`, `p(state)` by summing
    the `weights` which coincide with `data == state`. The log-odds then
    computes as logarithm of the odds `p(state) / (1 - p(state))`.

    Args:
        data (IArray): A 1D array of numeric values representing
            the sample data in code format.
        weights (FArray): A 1D array of numeric weights corresponding
            to the data.
        state (int): The state for which we estimate the log-odds.

    Returns:
        float: The weighted log-odds value.

    Raises:
        ValueError: If `data` and `weights` have different shapes or are not 1D,
            or if `state` is not in `data`.

    Example:
        ```python
        data = np.array([1, 0, 0])
        weights = np.array([0.4, 0.2, 0.4])
        print(compute_weighted_log_odds(data, weights, state=0))  # => 0.405...
        ```
    """
    probability = compute_weighted_probability(data=data, weights=weights, state=state)
    return math.log(probability / (1.0 - probability))


def compute_weighted_mean(
    data: FArray,
    weights: FArray,
) -> float:
    """
    Computes a weighted mean of the input data.

    This function calculates the weighted arithmetic mean of the input `data`
    and `weights` via the `compute_weighted_aggregate` function.

    Args:
        data (FArray): A 1D array of numeric values representing
            the data to be averaged.
        weights (FArray): A 1D array of numeric values representing
            the weights for the data.

    Returns:
        float: The computed weighted mean.

    Raises:
        ValueError: If `data` or `weights` are not 1D arrays.
        ValueError: If the shapes of `data` and `weights` do not match.

    Example:
        ```python
        data = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.2, 0.5, 0.3])
        print(compute_weighted_mean(data, weights))  # => 2.1
        ```
    """
    return compute_weighted_aggregate(data=data, weights=weights, factor=None)


def compute_weighted_median(
    data: FArray,
    weights: FArray,
    *,
    sorter: Optional[NDArray[np.integer]] = None,
) -> float:
    """
    Computes a weighted median of 1D data using linear interpolation.

    This function calculates the weighted meadian of the given `data` array
    based on the provided `weights` via `compute_weighted_quantile` with parameter
    `quantile=0.5`.

    Args:
        data (FArray): A 1D array of numeric values representing
            the sample data.
        weights (FArray): A 1D array of numeric weights corresponding
            to the data.
        sorter (Optional[NDArray[np.integer]]): Optional array of indices that
            sorts `data`.

    Returns:
        float: The interpolated weighted median value.

    Raises:
        ValueError: If `data` and `weights` have different shapes or are not 1D.

    Example:
        ```python
        data = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.4, 0.2, 0.4])
        print(compute_weighted_median(data, weights))  # => 2.25
        ```
    """
    return compute_weighted_quantile(
        data=data,
        weights=weights,
        quantile=0.5,
        sorter=sorter,
    )


def compute_weighted_mutual_information(
    data: IIArray,
    weights: FArray,
    *,
    normalize: bool = True,
) -> float:
    """
    Computes the weighted mutual information (dependency) between two 1D arrays.

    This function calculates the mutual information dependency between two integer variables
    using the direct mutual information formula on weighted joint distribution estimates.
    The inputs `data_1` and `data_2` are expected to be 1D arrays of the same length, provided
    as a tuple `data`. Each data point is assigned a weight from the `weights` array.

    Args:
        data (IIArray): A tuple of two 1D integer arrays `(data_1, data_2)`
            of equal length.
        weights (FArray): A 1D float array of weights, same length as
            each array in `data`.
        normalize (bool, optional): Whether or not to normalize the mutual information
            to range [0, 1]. Default is True

    Returns:
        float: The weighted mutual information.

    Raises:
        ValueError: If the input arrays are not 1D or have mismatched lengths.

    Example:
        ```python
        data_1 = np.array([0, 0, 1])
        data_2 = np.array([0, 1, 1])
        weights = np.array([0.2, 0.5, 0.3])
        print(compute_weighted_mutual_information((data_1, data_2), weights))  # => 0.146...
        ```

    Notes:
        - The normalized result is bounded by [0, 1], where 0 indicates perfect independence and
          1 indicates perfect dependence.
        - The unnormalized result is bounded between 0 (perfect independence) and
          `min(H(data_0), H(data_1))`, where `H(data_0)` and `H(data_1)` are the entropies of
          the two variables. If we reach the upper bound and `H(data_0) == H(data_1)`,
          we have perfect dependence.
    """
    validate_arrays(data=data, weights=weights)
    active_sets = (
        get_active_set(data[0]),
        get_active_set(data[1]),
    )
    joint_distribution = weighted_discrete_joint_distribution(
        data=active_sets,
        weights=weights,
    )
    distribution_0 = np.sum(joint_distribution, axis=1)
    distribution_1 = np.sum(joint_distribution, axis=0)
    divisor = np.outer(distribution_0, distribution_1).reshape(-1)
    joint_distribution = joint_distribution.reshape(-1)
    non_zero = joint_distribution > 0.0
    mutual_information = np.sum(
        joint_distribution[non_zero]
        * np.log(joint_distribution[non_zero] / divisor[non_zero])
    ).item()
    if mutual_information > 0 and normalize:
        mask = distribution_0 > 0.0
        entropy_0 = -np.dot(distribution_0[mask], np.log(distribution_0[mask])).item()
        mask = distribution_1 > 0.0
        entropy_1 = -np.dot(distribution_1[mask], np.log(distribution_1[mask])).item()
        mutual_information = 2.0 * mutual_information / (entropy_0 + entropy_1)
    return mutual_information


def compute_weighted_pearson_dependency(
    data: FFArray,
    weights: FArray,
    *,
    ddof: int = 0,
) -> float:
    """
    Computes the weighted Pearson correlation coefficient (dependency) between two 1D arrays.

    This function calculates the linear dependency between two variables using a weighted
    version of Pearson's correlation coefficient. The inputs `data_1` and `data_2` are
    expected to be 1D arrays of the same length, provided as a tuple `data`. Each data point
    is assigned a weight from the `weights` array.

    The function normalizes both variables by subtracting their weighted means and dividing
    by their weighted standard deviations, then computes the weighted mean of the element-wise
    product of these normalized arrays.

    Args:
        data (FArray): A tuple of two 1D float arrays `(data_1, data_2)`
            of equal length.
        weights (FArray): A 1D float array of weights, same length as
            each array in `data`.
        ddof (int, optional): Delta degrees of freedom for standard deviation.
            Defaults to 0 (population formula). Use 1 for sample-based correction.

    Returns:
        float: The weighted Pearson correlation coefficient in the range [-1, 1].

    Raises:
        ValueError: If the input arrays are not 1D or have mismatched lengths.

    Example:
        ```python
        data_1 = np.array([1.0, 2.0, 3.0])
        data_2 = np.array([1.0, 2.0, 2.9])
        weights = np.array([0.2, 0.5, 0.3])
        print(compute_weighted_pearson_dependency((data_1, data_2), weights))  # => 0.998...
        ```

    Notes:
        - The function relies on `compute_weighted_mean` and `compute_weighted_std`.
        - The correlation is computed using the formula:
            corr = weighted_mean(z1 * z2)
          where z1 and z2 are the standardized variables.
        - The result is bounded between -1 (perfect negative linear relationship)
          and 1 (perfect positive linear relationship), with 0 indicating no linear dependency.
    """
    data_1, data_2 = data
    weighted_mean_1 = compute_weighted_mean(data=data_1, weights=weights)
    weighted_mean_2 = compute_weighted_mean(data=data_2, weights=weights)
    weighted_std_1 = compute_weighted_std(
        data=data_1,
        weights=weights,
        weighted_mean=weighted_mean_1,
        ddof=ddof,
    )
    weighted_std_2 = compute_weighted_std(
        data=data_2,
        weights=weights,
        weighted_mean=weighted_mean_2,
        ddof=ddof,
    )
    array_1 = (data_1 - weighted_mean_1) / weighted_std_1
    array_2 = (data_2 - weighted_mean_2) / weighted_std_2
    return compute_weighted_mean(data=array_1 * array_2, weights=weights)


def compute_weighted_percentile(
    data: FArray,
    weights: FArray,
    *,
    percentile: float,
    sorter: Optional[NDArray[np.integer]] = None,
) -> float:
    """
    Computes a weighted percentile of 1D data using linear interpolation.

    This function calculates the weighted percentile of the given `data` array
    based on the provided `weights` via `compute_weighted_quantile` with parameter
    `quantile=0.01 * percentile`.

    Args:
        data (FArray): A 1D array of numeric values representing
            the sample data.
        weights (FArray): A 1D array of numeric weights corresponding
            to the data.
        percentile (float): The desired percentile in the interval [0, 100].
        sorter (Optional[NDArray[np.integer]]): Optional array of indices that
            sorts `data`.

    Returns:
        float: The interpolated weighted percentile value.

    Raises:
        ValueError: If `data` and `weights` have different shapes or are not 1D.

    Example:
        ```python
        data = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.2, 0.5, 0.3])
        print(compute_weighted_percentile(data, weights, percentile=70))  # => 2.2
        ```
    """
    return compute_weighted_quantile(
        data=data,
        weights=weights,
        quantile=0.01 * percentile,
        sorter=sorter,
    )


def compute_weighted_probability(
    data: IArray,
    weights: FArray,
    state: int,
) -> float:
    """
    Computes a weighted probability of a state within 1D code data.

    This function calculates the weighted probability of a `state` by summing
    the `weights` which coincide with `data == state`.

    Args:
        data (IArray): A 1D array of numeric values representing
            the sample data in code format.
        weights (FArray): A 1D array of numeric weights corresponding
            to the data.
        state (int): The state for which we estimate the probability.

    Returns:
        float: The weighted probability value.

    Raises:
        ValueError: If `data` and `weights` have different shapes or are not 1D,
            or if `state` is not in `data`.

    Example:
        ```python
        data = np.array([1, 0, 0])
        weights = np.array([0.4, 0.2, 0.4])
        print(compute_weighted_probability(data, weights, state=0))  # => 0.6
        ```
    """
    validate_array(data=data, weights=weights)
    if state not in data:
        raise ValueError(f"Incompatible parameter {state=:}: not included in data.")
    return np.sum(weights[data == state]).item()


def compute_weighted_quantile(
    data: FArray,
    weights: FArray,
    *,
    quantile: float,
    sorter: Optional[IArray] = None,
) -> float:
    """
    Computes a weighted quantile of 1D data using linear interpolation.

    This function calculates the weighted quantile of the given `data` array
    based on the provided `weights`. It uses a normalized cumulative weight
    distribution to determine the interpolated quantile value. The computation
    assumes both `data` and `weights` are 1D arrays of equal length.

    A precomputed `sorter` (array of indices that would sort `data`) can be
    optionally provided to avoid recomputing it internally.

    Args:
        data (FArray): A 1D array of numeric values representing
            the sample data.
        weights (FArray): A 1D array of numeric weights corresponding
            to the data.
        quantile (float): The desired quantile in the interval [0, 1].
        sorter (Optional[NDArray[np.integer]]): Optional array of indices that
            sorts `data`.

    Returns:
        float: The interpolated weighted quantile value.

    Raises:
        ValueError: If `data` and `weights` have different shapes or are not 1D.

    Example:
        ```python
        data = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.2, 0.5, 0.3])
        print(compute_weighted_quantile(data, weights, quantile=0.7))  # => 2.2
        ```

    Notes:
        - If `quantile` is less than or equal to the minimum cumulative weight,
          the smallest data point is returned.
        - If `quantile` is greater than or equal to the maximum cumulative weight,
          the largest data point is returned.
        - Linear interpolation is used between the two closest surrounding data points.
        - Providing a precomputed `sorter` can optimize performance in repeated calls.
    """
    validate_array(data=data, weights=weights)
    if sorter is None:
        sorter = np.argsort(data)
    elif sorter.shape != data.shape:
        raise ValueError(f"Invalid parameter {sorter.shape}: must match {data.shape}.")
    cumulative_weights = np.cumsum(weights[sorter])
    cw_lo, cw_hi = cumulative_weights[[0, -1]]
    cumulative_weights -= cw_lo
    cumulative_weights /= cw_hi - cw_lo
    data = data[sorter]
    if quantile <= cumulative_weights[0]:
        return data[0]
    if quantile >= cumulative_weights[-1]:
        return data[-1]
    idx = np.searchsorted(cumulative_weights, quantile)
    w_lo, w_hi = cumulative_weights[[idx - 1, idx]]
    s_lo, s_hi = data[[idx - 1, idx]]
    w = (quantile - w_lo) / (w_hi - w_lo)
    return (s_lo + w * (s_hi - s_lo)).item()


def compute_weighted_self_information(
    data: IArray,
    weights: FArray,
    state: int,
) -> float:
    """
    Computes a weighted self-information of a state within 1D code data.

    This function calculates the weighted probability of a `state` by summing
    the `weights` which coincide with `data == state`. The self-information then
    computes as negative logarithm of the weighted probability.

    Args:
        data (IArray): A 1D array of numeric values representing
            the sample data in code format.
        weights (FArray): A 1D array of numeric weights corresponding
            to the data.
        state (int): The state for which we estimate the self-information.

    Returns:
        float: The weighted self-information value.

    Raises:
        ValueError: If `data` and `weights` have different shapes or are not 1D,
            or if `state` is not in `data`.

    Example:
        ```python
        data = np.array([1, 0, 0])
        weights = np.array([0.4, 0.2, 0.4])
        print(compute_weighted_self_information(data, weights, state=0))  # => 0.510...
        ```
    """
    probability = compute_weighted_probability(data=data, weights=weights, state=state)
    return -math.log(probability)


def compute_weighted_spearman_dependency(
    data: FFArray,
    weights: FArray,
    *,
    ddof: int = 0,
) -> float:
    """
    Computes the weighted Spearman rank correlation coefficient between two 1D arrays.

    This function measures the monotonic relationship between two variables by computing
    the weighted Pearson correlation between their ranked values (i.e., their order statistics).
    It is particularly useful for assessing non-linear relationships.

    Args:
        data (FArray): A tuple of two 1D float arrays `(data_1, data_2)`
            of equal length.
        weights (FArray): A 1D float array of weights, same length as
            each array in `data`.
        ddof (int, optional): Delta degrees of freedom for standard deviation.
            Defaults to 0 (population formula). Use 1 for sample-based correction.

    Returns:
        float: The weighted Spearman rank correlation coefficient in the range [-1, 1].

    Raises:
        ValueError: If the input arrays are not 1D or have mismatched lengths.

    Example:
        ```python
        data_1 = np.array([1.0, 2.0, 3.0])
        data_2 = np.array([0.3, 0.2, 0.1])
        weights = np.array([0.2, 0.5, 0.3])
        print(compute_weighted_spearman_dependency((data_1, data_2), weights))  # => -0.9999...
        ```

    Notes:
        - Internally, ranks are computed using `scipy.stats.rankdata`, which handles ties
          by assigning average ranks.
        - The Spearman coefficient is equivalent to the Pearson correlation between
          rank-transformed data.
        - Output is bounded between -1 (perfect inverse monotonic relationship)
          and 1 (perfect direct monotonic relationship), with 0 indicating no
          monotonic correlation.
        - Weights are applied after ranking.
    """
    data_1, data_2 = data
    ranks_1 = cast(FArray, rankdata(data_1))
    ranks_2 = cast(FArray, rankdata(data_2))
    return compute_weighted_pearson_dependency(
        data=(ranks_1, ranks_2),
        weights=weights,
        ddof=ddof,
    )


def compute_weighted_std(
    data: FArray,
    weights: FArray,
    *,
    weighted_mean: Optional[float] = None,
    ddof: int = 0,
) -> float:
    """
    Computes a weighted standard deviation of the input data.

    This function calculates the weighted standard deviation of the
    input `data` and `weights` via the square root of the
    `compute_weighted_variance` function.

    Args:
        data (FArray): A 1D array of numeric values representing
            the data for which we want the standard deviation.
        weights (FArray): A 1D array of numeric values representing
            the weights for the data.
        weighted_mean (float, optional): The weighted mean of the data (default is
            None). If missing, this value is computed via `compute_weighted_mean`
            via `compute_weighted_variance`.
        ddof (int, optional): Delta degrees of freedom.
            Defaults to 0 (population formula). Use 1 for sample-based correction.

    Returns:
        float: The computed weighted standard deviation.

    Raises:
        ValueError: If `data` or `weights` are not 1D arrays.
        ValueError: If the shapes of `data` and `weights` do not match.

    Example:
        ```python
        data = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.2, 0.5, 0.3])
        print(compute_weighted_std(data, weights))  # => 0.7
        ```
    """
    weighted_variance = compute_weighted_variance(
        data=data,
        weights=weights,
        weighted_mean=weighted_mean,
        ddof=ddof,
    )
    return math.sqrt(weighted_variance)


def compute_weighted_sum(
    data: FArray,
    weights: FArray,
) -> float:
    """
    Computes a weighted sum of the input data.

    This function calculates the weighted sum of the input `data`
    and `weights` via the `compute_weighted_aggregate` function with
    `factor=len(data)`.

    Args:
        data (FArray): A 1D array of numeric values representing
            the data to be summed.
        weights (FArray): A 1D array of numeric values representing
            the weights for the data.

    Returns:
        float: The computed weighted sum.

    Raises:
        ValueError: If `data` or `weights` are not 1D arrays.
        ValueError: If the shapes of `data` and `weights` do not match.

    Example:
        ```python
        data = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.2, 0.5, 0.3])
        print(compute_weighted_sum(data, weights))  # => 6.3
        ```
    """
    return compute_weighted_aggregate(data=data, weights=weights, factor=len(data))


def compute_weighted_variance(
    data: FArray,
    weights: FArray,
    *,
    weighted_mean: Optional[float] = None,
    ddof: int = 0,
) -> float:
    """
    Computes a weighted variance of the input data.

    This function calculates the weighted variance of the input `data`
    and `weights` via the `compute_weighted_aggregate` function with
    `factor=len(data) / (len(data) - ddof)`, where `ddof` specifies the
    delta degrees of freedom.

    Args:
        data (FArray): A 1D array of numeric values representing
            the data for which we want the variance.
        weights (FArray): A 1D array of numeric values representing
            the weights for the data.
        weighted_mean (float, optional): The weighted mean of the data (default is
            None). If missing, this value is computed via `compute_weighted_mean`.
        ddof (int, optional): Delta degrees of freedom.
            Defaults to 0 (population formula). Use 1 for sample-based correction.

    Returns:
        float: The computed weighted variance.

    Raises:
        ValueError: If `data` or `weights` are not 1D arrays.
        ValueError: If the shapes of `data` and `weights` do not match.

    Example:
        ```python
        data = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.2, 0.5, 0.3])
        print(compute_weighted_variance(data, weights))  # => 0.49
        print(compute_weighted_variance(data, weights, ddof=1))  # => 0.735
        ```
    """
    if weighted_mean is None:
        weighted_mean = compute_weighted_mean(data=data, weights=weights)
    return compute_weighted_aggregate(
        data=np.power(data - weighted_mean, 2.0),
        weights=weights,
        factor=len(data) / (len(data) - ddof),
    )


def get_active_set(data: Union[FArray, IArray]) -> IArray:
    """
    Extracts the active set of states from a 1D array.

    The active set is a gap-free collection of codes 0,...,n-1 for
    the n unique values in the array.

    Args:
        data (FArray or IArray): A 1D array of numeric values.

    Returns:
        IArray: The active set of data.

    Example:
        ```python
        data = np.array([1.0, 4.0, 3.0])
        print(get_active_set(data))  # => np.array([0, 2, 1])
        ```
    """
    _, active_set = np.unique(data, return_inverse=True)
    return active_set


def validate_array(data: Union[FArray, IArray], weights: FArray) -> None:
    """
    Validates data and weights parameters.

    This function probes that both data and weights are 1D arrays of the same length.

    Args:
        data (FArray or IArray): A 1D array of numeric values.
        weights (FArray): A 1D array of numeric values representing
            the weights for the data.

    Raises:
        ValueError: If `data` or `weights` are not 1D arrays.
        ValueError: If the shapes of `data` and `weights` do not match.
    """
    if data.ndim != 1:
        raise ValueError(f"Invalid parameter {data.ndim=:}: must be 1.")
    if weights.ndim != 1:
        raise ValueError(f"Invalid parameter {weights.ndim=:}: must be 1.")
    if weights.shape != data.shape:
        raise ValueError(
            f"Incompatible parameters shapes {weights.shape=:} ≠ {data.shape=:}: "
            "must be equal."
        )


def validate_arrays(data: Union[FFArray, IFArray, IIArray], weights: FArray) -> None:
    """
    Validates data and weights parameters.

    This function probes that both data is a tuple of 1D arrays, weights is 1D array,
    and all arrays are of the same length.

    Args:
        data (FFArray or IFArray or IIArray): A tuple of two 1D array of numeric values.
        weights (FArray): A 1D array of numeric values representing
            the weights for the data.

    Raises:
        ValueError: If the two elements of `data` or `weights` are not 1D arrays.
        ValueError: If the shapes of `data` and `weights` do not match.
    """
    if len(data) != 2:
        raise ValueError(f"Invalid parameter {len(data)}: must contain two arrays.")
    if data[0].ndim != 1:
        raise ValueError(f"Invalid parameter {data[0].ndim=:}: must be 1.")
    if data[1].ndim != 1:
        raise ValueError(f"Invalid parameter {data[1].ndim=:}: must be 1.")
    if weights.ndim != 1:
        raise ValueError(f"Invalid parameter {weights.ndim=:}: must be 1.")
    if weights.shape != data[0].shape:
        raise ValueError(
            f"Incompatible parameters shapes {weights.shape=:} ≠ {data[0].shape=:}: "
            "must be equal."
        )
    if weights.shape != data[1].shape:
        raise ValueError(
            f"Incompatible parameters shapes {weights.shape=:} ≠ {data[1].shape=:}: "
            "must be equal."
        )


def weighted_discrete_distribution(data: IArray, weights: FArray) -> FArray:
    """
    Computes the weighted discrete distribution from a 1D array.

    Args:
        data (IArray): A 1D array of code values.
        weights (FArray): A 1D array of numeric values representing
            the weights for the data.

    Returns:
        FArray: The weighted discrete distribution of data.
    """
    return cast(FArray, np.bincount(data, weights=weights))


def weighted_discrete_joint_distribution(data: IIArray, weights: FArray) -> FArray:
    """
    Computes the weighted discrete joint distribution from two 1D arrays.

    Args:
        data (IIArray): Tuple of two 1D arrays of code values.
        weights (FArray): A 1D array of numeric values representing
            the weights for the data.

    Returns:
        FArray: The weighted discrete joint distribution of data.
    """
    data_0, data_1 = data
    minlength = np.max(data_1) + 1
    rows = []
    for state_0 in range(np.max(data_0) + 1):
        mask = data_0 == state_0
        rows.append(
            np.bincount(data_1[mask], weights=weights[mask], minlength=minlength)
        )
    distribution = np.array(rows)
    return cast(FArray, distribution)
