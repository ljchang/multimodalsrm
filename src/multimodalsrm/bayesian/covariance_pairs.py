"""Reuse exact modality/separation pairs on fixed native observation clocks."""

from dataclasses import dataclass

import numpy as np

from ._backend import runtime, temporal_values


@dataclass(frozen=True)
class CovariancePairs:
    left_modalities: np.ndarray
    right_modalities: np.ndarray
    deltas: np.ndarray
    inverse: np.ndarray
    size: int

    @classmethod
    def prepare(cls, times, modalities):
        """Return a lookup only when at least half the pair evaluations repeat.

        Keys retain ordered modality IDs and exact float64 differences. There
        is no binning, lag quantization, stationarity approximation or merging
        of modalities. Building this fixed lookup costs O(n²) temporary storage;
        a low-repetition clock discards it and uses the direct kernel.
        """
        size = len(times)
        delta = times[:, None] - times[None, :]
        keys = np.rec.fromarrays(
            [
                np.broadcast_to(modalities[:, None], delta.shape).ravel(),
                np.broadcast_to(modalities[None, :], delta.shape).ravel(),
                delta.ravel(),
            ],
            names="left,right,delta",
        )
        unique, inverse = np.unique(keys, return_inverse=True)
        if len(unique) >= size * size / 2:
            return None
        return cls(unique.left, unique.right, unique.delta, inverse.astype(np.int32), size)

    def evaluate(self, widths, lags, length_scale):
        _, jnp, _, _ = runtime()
        left, right = self.left_modalities, self.right_modalities
        delta = jnp.asarray(self.deltas) - lags[left] + lags[right]
        values = temporal_values(delta, widths[left], widths[right], length_scale)
        return values[self.inverse].reshape(self.size, self.size)
