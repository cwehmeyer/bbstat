from typing import Generator, Optional

import numpy as np
from numpy.typing import NDArray


def resample(
    n_boot: int,
    n_data: int,
    seed: Optional[int] = None,
    blocksize: Optional[int] = None,
) -> Generator[NDArray[np.floating], None, None]:
    """
    Generates bootstrap resamples with Dirichlet-distributed weights.

    This function performs resampling by generating weights from a Dirichlet distribution.
    The number of resamples is controlled by the `n_boot` argument, while the size of
    each block of resamples can be adjusted using the `blocksize` argument. The `seed`
    argument allows for reproducible results.

    Args:
        n_boot (int): The total number of bootstrap resamples to generate.
        n_data (int): The number of data points to resample (used for the dimension of the
            Dirichlet distribution).
        seed (Optional[int]): A random seed for reproducibility (default is `None` for
            random seeding).
        blocksize (Optional[int]): The number of resamples to generate in each block.
            If `None`, the entire number of resamples is generated in one block.
            Defaults to `None`.

    Yields:
        Generator[NDArray[np.floating], None, None]: A generator that yields each resample
            (a 1D array of floats) as it is generated. Each resample contains Dirichlet-
            distributed weights for the given `n_data`.

    Example:
        >>> for r in resample(n_boot=10, n_data=5):
        >>>     print(r)

    Notes:
        - If `blocksize` is specified, the resampling will be performed in smaller blocks,
          which can be useful for parallelizing or limiting memory usage.
        - The function uses NumPy's `default_rng` to generate random numbers, which provides
          a more flexible and efficient interface compared to `np.random.seed`.

    Raises:
        ValueError: If `n_boot` is less than or equal to 0, or `n_data` is less than 1.
    """
    rng = np.random.default_rng(seed)
    alpha = np.ones(n_data)
    if blocksize is None:
        blocksize = n_boot
    else:
        blocksize = min(blocksize, n_boot)
    remainder = n_boot
    while remainder > 0:
        size = min(blocksize, remainder)
        weights = rng.dirichlet(alpha=alpha, size=size)
        for w in weights:
            yield w
        remainder -= size
