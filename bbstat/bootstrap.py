from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .evaluate import BootstrapResult
from .resample import resample


def bootstrap(
    data: Union[
        NDArray[np.floating],
        List[NDArray[np.floating]],
        Tuple[NDArray[np.floating]],
    ],
    statistic_fn: Callable[
        [
            Union[
                NDArray[np.floating],
                List[NDArray[np.floating]],
                Tuple[NDArray[np.floating]],
            ],
            NDArray[np.floating],
            Dict[str, Any],
        ],
        NDArray[np.floating],
    ],
    n_boot: int = 1000,
    coverage: float = 0.87,
    seed: Optional[int] = None,
    blocksize: Optional[int] = None,
    **kwargs: Dict[str, Any],
) -> NDArray[np.floating]:
    if isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError(f"Invalid parameter {data.ndim=:}: must be 1.")
        n_data: int = len(data)
    elif isinstance(data, (tuple, list)):
        n_data = len(data[0])
        for i, array in enumerate(data):
            if array.ndim != 1:
                raise ValueError(f"Invalid parameter {data[i].ndim=:}: must be 1.")
            if n_data != len(array):
                raise ValueError(
                    f"Invalid parameter {data[i].shape[0]=:}: must be {n_data=:}."
                )
    estimates = np.array(
        [
            statistic_fn(data=data, weights=weights, **kwargs)
            for weights in resample(
                n_boot=n_boot,
                n_data=n_data,
                seed=seed,
                blocksize=blocksize,
            )
        ]
    )
    return BootstrapResult(estimates=estimates, coverage=coverage)
