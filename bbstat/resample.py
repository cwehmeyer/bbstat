from typing import Generator, Optional

import numpy as np
from numpy.typing import NDArray


def resample(
    n_boot: int,
    n_data: int,
    seed: Optional[int] = None,
    blocksize: Optional[int] = None,
) -> Generator[NDArray[np.floating], None, None]:
    """Generate Dirichlet-distributed weights.

    Parameters
    ----------
    n_boot: int
        Number of resampling steps for the bootstrap.
    n_data: int
        Number of data points to resample.
    seed: Optional[int], default is None
        Optional seed for the random number generator.
    blocksize: Optional[int], default is None
        Optional size for the `numpy.random.Generator.dirichlet()` call
        (how many resampling steps are generated in one call).

    Returns
    -------
    Generator[numpy.array(shape=n_data), None, None]
        Dirichlet-distributed weights with `alpha=numpy.ones(n_data)`.
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
