from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class Estimate:
    key: str
    samples: NDArray[np.floating]
    n_samples: int = field(init=False)

    def __post_init__(self):
        self.n_samples = len(self.samples)

    def mean(self) -> float:
        return np.mean(self.samples).item()

    def ci(self, width: float = 0.87) -> Tuple[float, float]:
        out = (1.0 - width) / 2.0
        return tuple(np.quantile(self.samples, [out, 1.0 - out]).tolist())

    def __add__(self, other: "Estimate") -> "Estimate":
        if self.key != other.key:
            raise ValueError(f"Incompatible keys: {self.key}≠{other.key}.")
        return Estimate(
            key=self.key,
            samples=np.concatenate([self.samples, other.samples]),
        )
