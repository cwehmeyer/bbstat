from typing import Optional

import numpy as np
import pytest

from bbstat.resample import Resample, resample


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
def test_Resample(data_size: int, n_samples: int, blocksize: Optional[int]) -> None:
    resampler = Resample(alpha=np.ones(shape=(data_size,), dtype=np.float64))
    samples = list(resampler(n_samples))
    assert len(samples) == n_samples
    for sample in samples:
        assert sample.shape == (data_size,)
        assert (0 <= sample).all()
        assert (sample <= 1).all()
        np.testing.assert_allclose(np.sum(sample), 1.0)


@pytest.mark.parametrize(
    "n_boot",
    [
        pytest.param(1),
        pytest.param(5),
    ],
)
@pytest.mark.parametrize(
    "n_data",
    [
        pytest.param(2),
        pytest.param(3),
    ],
)
@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(None),
        pytest.param(1),
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
def test_resample(
    n_boot: int,
    n_data: int,
    seed: Optional[int],
    blocksize: Optional[int],
) -> None:
    samples = list(
        resample(
            n_boot=n_boot,
            n_data=n_data,
            seed=seed,
            blocksize=blocksize,
        )
    )
    assert len(samples) == n_boot
    for sample in samples:
        assert sample.shape == (n_data,)
        assert (0 <= sample).all()
        assert (sample <= 1).all()
        np.testing.assert_allclose(np.sum(sample), 1.0)
