"""Timestamped Gaussian conditional means and marginal standard deviations."""

from dataclasses import dataclass

import numpy as np

from .data import readonly_array
from .results import SeriesResult


@dataclass(frozen=True, eq=False)
class PosteriorSeriesResult(SeriesResult):
    """An immutable result whose uncertainty is conditional on fitted parameters.

    ``std`` matches ``values`` and uses the same units. Invalid entries in both
    arrays are NaN; a valid entry cannot have a negative standard deviation.
    """

    std: np.ndarray | None = None

    def __post_init__(self):
        super().__post_init__()
        std = np.array(self.std, dtype=float, copy=True)
        if std.shape != self.values.shape:
            raise ValueError("std must match the shape of result values")
        if np.any(std[self.valid] < 0):
            raise ValueError("std must be nonnegative at valid entries")
        std[~self.valid] = np.nan
        object.__setattr__(self, "std", readonly_array(std))
