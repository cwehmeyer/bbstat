from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def credibility_interval(
    estimates: NDArray[np.floating],
    coverage: float = 0.87,
) -> Tuple[float, float]:
    if estimates.ndim != 1:
        raise ValueError(f"Invalid parameter {estimates.ndim=:}: must be 1D array.")
    if coverage <= 0 or coverage >= 1:
        raise ValueError(f"Invalid parameter {coverage=:}: must be within (0, 1).")
    edge = (1.0 - coverage) / 2.0
    return tuple(np.quantile(estimates, [edge, 1.0 - edge]).tolist())


@dataclass
class BootstrapResult:
    mean: float = field(init=False)
    ci: Tuple[float, float] = field(init=False)
    coverage: float
    n_boot: int = field(init=False)
    estimates: Optional[NDArray[np.floating]]

    def __post_init__(self):
        self.mean = np.mean(self.estimates).item()
        self.ci = credibility_interval(
            estimates=self.estimates,
            coverage=self.coverage,
        )
        self.n_boot = len(self.estimates)

    def __str__(self) -> str:
        return f"BootstrapResult(mean={self.mean}, ci={self.ci}, coverage={self.coverage}, n_boot={self.n_boot})"
    
    def __repr__(self) -> str:
        return self.__str__()

    def credibility_interval(self, coverage: float) -> Tuple[float, float]:
        return credibility_interval(estimates=self.estimates, coverage=coverage)
