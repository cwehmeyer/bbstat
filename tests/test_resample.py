from typing import Optional

import numpy as np
import pytest

from bbstat.resample import Resample


@pytest.mark.parametrize(
    "data_size",
    [
        pytest.param(2),
        pytest.param(5),
    ],
)
@pytest.mark.parametrize(
    "n_samples",
    [
        pytest.param(1),
        pytest.param(3),
    ],
)
@pytest.mark.parametrize(
    "blocksize",
    [
        pytest.param(None),
        pytest.param(1),
        pytest.param(10),
    ],
)
def test_resample(data_size: int, n_samples: int, blocksize: Optional[int]) -> None:
    resample = Resample(alpha=np.ones(shape=(data_size,), dtype=np.float64))
    samples = list(resample(n_samples))
    assert len(samples) == n_samples
    for sample in samples:
        assert sample.shape == (data_size,)
        assert (0 <= sample).all()
        assert (sample <= 1).all()
        np.testing.assert_allclose(np.sum(sample), 1.0)
