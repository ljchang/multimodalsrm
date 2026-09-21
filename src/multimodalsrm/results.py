"""Named timestamped point estimates; validity does not imply observed coverage."""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from .data import readonly_array, validate_times


@dataclass(frozen=True, eq=False)
class SeriesResult:
    values: np.ndarray
    times: np.ndarray
    valid: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        values = np.array(self.values, dtype=float, copy=True)
        times = validate_times(self.times, allow_empty=True)
        valid = np.asarray(self.valid)
        if values.ndim != 2 or len(values) != len(times):
            raise ValueError("result values must have shape time by component/feature")
        if valid.dtype != np.dtype(bool) or valid.shape not in (
            (len(times),),
            values.shape,
        ):
            raise ValueError("valid must be Boolean per timestamp or per entry")
        values[~valid] = np.nan
        object.__setattr__(self, "values", readonly_array(values))
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "valid", readonly_array(valid, bool))
        object.__setattr__(self, "metadata", deepcopy(self.metadata))
