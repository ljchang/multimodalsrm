# Prepare your data

Both models accept a nested mapping:

```text
participant → run → modality → TimeSeries
```

A `TimeSeries` holds an array of shape `(observations, features)` and a timestamp vector of shape `(observations,)`. Each stream keeps its own sampling times. Use consistent time units across all streams; lag, response width, latent grid spacing, and GP timescale use those units.

## A small input example

```python
import numpy as np
from multimodalsrm import TimeSeries

brain_times = np.arange(0.0, 20.0, 2.0)
rating_times = np.arange(0.0, 20.0, 0.5)
brain = np.column_stack([np.sin(brain_times), np.cos(brain_times)])
rating = np.sin(rating_times)[:, None]

observed = np.ones_like(rating, dtype=bool)
observed[5:8] = False

data = {
    "participant-01": {
        "movie-01": {
            "brain": TimeSeries(brain, brain_times),
            "rating": TimeSeries(rating, rating_times, observed),
        }
    }
}
```

This demonstrates the input structure; use the [tutorial scripts](tutorials.md) for complete fitting examples with multiple participants. Missing streams may be omitted. A mask marks observed entries within an available stream.

## Give run names scientific meaning

Participants observing the same aligned recording should use the same run ID and time origin. That ID associates their observations with one shared latent response. Different recordings have different run IDs and separate latent responses.

An independent test recording must have a new run ID. Reusing a training run name does not make it an independent prediction problem.

## Separate training, donors, and targets

Training observations estimate mappings and response parameters. Donor observations provide information when inferring a response or predicting an excluded target. A target is the participant–modality stream being predicted.

For R-MSRM, pass explicit `targets` to `predict`. For GP-MSRM, pass them to `condition`. Named target payloads are excluded before model preparation. Any preprocessing you perform yourself must respect the same exclusion; the model cannot undo information leakage introduced earlier.

R-MSRM learns its preprocessing from training observations and reuses it. GP-MSRM operates in supplied observation units. If you use an external standardizer for GP data, fit it only on training observations and retain it with the fitted model.

## Inspect valid support

A response filter can require observations beyond the requested prediction time. Near recording boundaries, some outputs can therefore be invalid. Use returned validity masks when scoring or plotting; do not assume every requested time is supported.

See [how the models work](concepts.md) for the relationship between shared responses, individual mappings, and response filters.
