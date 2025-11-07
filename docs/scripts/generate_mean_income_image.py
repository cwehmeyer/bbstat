from pathlib import Path

import numpy as np

from bbstat import bootstrap, plot

scripts_dir = Path(__file__).resolve().parent
images_dir = scripts_dir.parent / "images"

outfile = images_dir / "mean-income.png"

# Data preparation: simulated income for a small population (e.g., a survey of 25 people)
income = np.array(
    [
        24_000,
        26_000,
        28_000,
        30_000,
        32_000,
        35_000,
        36_000,
        38_000,
        40_000,
        41_000,
        45_000,
        48_000,
        50_000,
        52_000,
        54_000,
        58_000,
        60_000,
        62_000,
        65_000,
        68_000,
        70_000,
        75_000,
        80_000,
        90_000,
        100_000,
    ],
    dtype=np.float64,
)

distribution = bootstrap(data=income, statistic_fn="mean", seed=1)

ax = plot(distribution, 0.95, precision="auto", label="mean income")
ax.get_figure().savefig(outfile)  # type: ignore[union-attr]
