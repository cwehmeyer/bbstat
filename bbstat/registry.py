"""Statistic function registry and protocol definition.

This module defines a strict `Protocol` (`StatisticFunction`) for all supported
statistical aggregation functions used in the system. It also provides a typed
mapping of statistic function names to their concrete implementations and a
lookup function (`get_statistic_fn`) for retrieving them by name.

All registered functions are callable with specific combinations of arguments
(e.g. `data`, `weights`, and optional parameters like `ddof`, `factor`, or
`sorter`) depending on the computation type. Static typing ensures correct
usage of each registered function.
"""

from typing import (
    Dict,
    Optional,
    Protocol,
    Tuple,
    cast,
    overload,
)

from .statistics import (
    FArray,
    FFArray,
    IArray,
    IFArray,
    compute_weighted_aggregate,
    compute_weighted_entropy,
    compute_weighted_eta_square_dependency,
    compute_weighted_log_odds,
    compute_weighted_mean,
    compute_weighted_median,
    compute_weighted_pearson_dependency,
    compute_weighted_percentile,
    compute_weighted_probability,
    compute_weighted_quantile,
    compute_weighted_self_information,
    compute_weighted_spearman_dependency,
    compute_weighted_std,
    compute_weighted_sum,
    compute_weighted_variance,
)

__all__ = [
    "get_statistic_fn",
    "StatisticFunction",
]


class StatisticFunction(Protocol):
    """
    A protocol defining the interface for all statistical computation functions.

    Each implementing function must take `data` and `weights` arrays and may
    accept additional keyword-only arguments depending on the computation type.

    Overloads:

    - `aggregate`: accepts `data: FArray`, `weights: FArray`, and optional `factor: float`
    - `mean`, `sum`: accept only `data: FArray`, `weights: FArray`
    - `variance`, `std`: accept optional `weighted_mean: float` and `ddof: int`
    - `quantile`: requires `quantile: float` and optional `sorter: IArray`
    - `percentile`: requires `percentile: float` and optional `sorter: IArray`
    - `median`: accepts optional `sorter`
    - `pearson_dependency`, `spearman_dependency`: take tuple of two
      float arrays (`FFArray`) and `ddof`
    - `eta_square_dependency`: takes tuple of and integer and a float
      array (`IFArray`)
    - `entropy`: accepts `data: IFArray` and `weights: FArray`
    - `probability`, `self_information`, `log_odds`: accepts `data: IFArray`,
      `weights: FArray`, and `state: int`
    """

    # aggregate
    @overload
    def __call__(
        self,
        data: FArray,
        weights: FArray,
        *,
        factor: Optional[float],
    ) -> float: ...

    # mean, sum
    @overload
    def __call__(
        self,
        data: FArray,
        weights: FArray,
    ) -> float: ...

    # entropy
    @overload
    def __call__(
        self,
        data: IArray,
        weights: FArray,
    ) -> float: ...

    # probability, log_odds, self_information
    @overload
    def __call__(
        self,
        data: IArray,
        weights: FArray,
        state: int,
    ) -> float: ...

    # variance, std
    @overload
    def __call__(
        self,
        data: FArray,
        weights: FArray,
        *,
        weighted_mean: Optional[float],
        ddof: int,
    ) -> float: ...

    # quantile
    @overload
    def __call__(
        self,
        data: FArray,
        weights: FArray,
        *,
        quantile: float,
        sorter: Optional[IArray],
    ) -> float: ...

    # percentile
    @overload
    def __call__(
        self,
        data: FArray,
        weights: FArray,
        *,
        percentile: float,
        sorter: Optional[IArray],
    ) -> float: ...

    # median
    @overload
    def __call__(
        self,
        data: FArray,
        weights: FArray,
        *,
        sorter: Optional[IArray],
    ) -> float: ...

    # pearson_dependency, spearman_dependency
    @overload
    def __call__(
        self,
        data: FFArray,
        weights: FArray,
        *,
        ddof: int,
    ) -> float: ...

    # eta_squared_dependency
    @overload
    def __call__(
        self,
        data: IFArray,
        weights: FArray,
    ) -> float: ...


STATISTIC_FUNCTIONS: Dict[str, StatisticFunction] = {
    "aggregate": cast(
        StatisticFunction,
        compute_weighted_aggregate,
    ),
    "mean": cast(
        StatisticFunction,
        compute_weighted_mean,
    ),
    "sum": cast(
        StatisticFunction,
        compute_weighted_sum,
    ),
    "variance": cast(
        StatisticFunction,
        compute_weighted_variance,
    ),
    "std": cast(
        StatisticFunction,
        compute_weighted_std,
    ),
    "quantile": cast(
        StatisticFunction,
        compute_weighted_quantile,
    ),
    "percentile": cast(
        StatisticFunction,
        compute_weighted_percentile,
    ),
    "median": cast(
        StatisticFunction,
        compute_weighted_median,
    ),
    "entropy": cast(
        StatisticFunction,
        compute_weighted_entropy,
    ),
    "probability": cast(
        StatisticFunction,
        compute_weighted_probability,
    ),
    "self_information": cast(
        StatisticFunction,
        compute_weighted_self_information,
    ),
    "log_odds": cast(
        StatisticFunction,
        compute_weighted_log_odds,
    ),
    "pearson_dependency": cast(
        StatisticFunction,
        compute_weighted_pearson_dependency,
    ),
    "spearman_dependency": cast(
        StatisticFunction,
        compute_weighted_spearman_dependency,
    ),
    "eta_square_dependency": cast(
        StatisticFunction,
        compute_weighted_eta_square_dependency,
    ),
}


def get_statistic_fn(name: str) -> StatisticFunction:
    """
    Retrieve a registered statistic function by name.

    Parameters:
        name (str): The lowercase name of the statistic function to retrieve.
            Must be one of:
            - "aggregate"
            - "mean"
            - "sum"
            - "variance"
            - "std"
            - "quantile"
            - "percentile"
            - "median"
            - "entropy"
            - "probability"
            - "self_information"
            - "log_odds"
            - "pearson_dependency"
            - "spearman_dependency"
            - "eta_square_dependency"

    Returns:
        StatisticFunction: The corresponding function implementation.

    Raises:
        ValueError: If the name does not correspond to a registered function.
    """
    try:
        return STATISTIC_FUNCTIONS[name.lower()]
    except KeyError:
        raise ValueError(
            f"Invalid {name=:}: choose from {list(STATISTIC_FUNCTIONS.keys())}"
        )


def get_statistic_fn_names() -> Tuple[str, ...]:
    """
    Retrieve the names of registered statistic functions.

    Returns:
        Tuple[str, ...]: The names of the available statistic functions.
    """
    return tuple(STATISTIC_FUNCTIONS.keys())
