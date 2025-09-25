from typing import Generator, Optional

import numpy as np
from numpy.typing import NDArray


class Resample:
    def __init__(self, alpha: NDArray, blocksize: Optional[int] = None):
        self.alpha = alpha
        self.blocksize = blocksize

    def __call__(self, n_samples: int) -> Generator[NDArray[np.floating], None, None]:
        while n_samples > 0:
            size = min(self.blocksize or n_samples, n_samples)
            weights_block = np.random.default_rng().dirichlet(self.alpha, size=size)
            for weights in weights_block:
                yield weights
            n_samples = n_samples - size

    @classmethod
    def from_data(
        cls,
        data: NDArray[np.floating],
        blocksize: Optional[int] = None,
    ) -> "Resample":
        alpha = np.ones(shape=(len(data),), dtype=data.dtype)
        return cls(alpha=alpha, blocksize=blocksize)
