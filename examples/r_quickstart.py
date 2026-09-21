"""Small synthetic R-MSRM fit, new-run prediction and participant calibration.

Run after installing multimodalsrm. This demonstrates the API and reports fit
status; it is not a physiological response-recovery experiment.
"""

import numpy as np

from multimodalsrm import Gaussian, Identity, MultimodalSRM, Response, TimeSeries


def recording(run, participants):
    """Independent run phases, stable individual mappings and native clocks."""
    times = np.arange(0, 24.01, 0.5)
    phase = np.random.default_rng(run).uniform(0, 2 * np.pi, 2)
    latent = np.sin(2 * np.pi * times / 8 + phase[0])
    latent += 0.3 * np.cos(2 * np.pi * times / 3 + phase[1])
    width, lag = 0.3, 0.6
    # Analytic filtering of the two harmonics by an L2-normalized Gaussian.
    mass = np.sqrt(2) * np.pi**0.25 * np.sqrt(width)
    filtered = mass * (
        np.exp(-0.5 * (width * 2 * np.pi / 8) ** 2)
        * np.sin(2 * np.pi * (times - lag) / 8 + phase[0])
        + 0.3
        * np.exp(-0.5 * (width * 2 * np.pi / 3) ** 2)
        * np.cos(2 * np.pi * (times - lag) / 3 + phase[1])
    )
    data = {}
    for participant in participants:
        index = {"a": 1, "b": 2, "new": 3}[participant]
        rng = np.random.default_rng(100 * run + index)
        brain = filtered[:, None] * np.array([[0.7 + index / 10, -1.2]])
        brain += rng.normal(0, 0.02, brain.shape)
        rating = (0.8 + index / 10) * latent[::2, None]
        rating += rng.normal(0, 0.02, rating.shape)
        mask = np.ones_like(brain, dtype=bool)
        mask[10, 1] = False
        data[participant] = {
            f"run-{run}": {
                "brain": TimeSeries(brain, times, mask),
                "rating": TimeSeries(rating, times[::2]),
            }
        }
    return data


def main():
    participants = ["a", "b"]
    training = recording(1, participants)
    for participant, runs in recording(2, participants).items():
        training[participant].update(runs)
    model = MultimodalSRM(
        features=1,
        latent_dt=0.5,
        latent_pooling="shared",
        responses={
            "brain": Response(
                Gaussian(width=0.3),
                estimate=True,
                fixed={"width": 0.3},
                bounds={"lag": (-1.5, 1.5)},
                pooling="shared",
            ),
            "rating": Response(Identity(), estimate=False),
        },
        init="hybrid",
        n_init=2,
        max_iter=100,
        tol=1e-6,
        random_state=0,
    ).fit(training)
    independent = recording(3, participants)
    # All named targets are excluded before preparing the source observations.
    prediction = model.predict(independent, targets={"a": ["brain"]}, source="both")
    brain = prediction["a"]["run-3"]["brain"]
    latents = model.infer_latent(independent)
    assert brain.valid.any()
    assert np.isfinite(brain.values[brain.valid]).all()
    print("Prediction shape:", brain.values.shape, "valid times:", brain.valid.sum())
    print("Latent shape:", latents["a"]["run-3"].values.shape)
    print("Fitted brain parameters:", model.kernel("brain", subject="a").parameters)
    print("Training converged:", model.converged_)

    # Fitted donors in a separate recording anchor the newcomer's mappings.
    calibrated = model.calibrate(recording(4, participants + ["new"]))
    newcomer = calibrated.predict(
        recording(5, participants + ["new"]), targets={"new": ["brain"]}, source="within"
    )["new"]["run-5"]["brain"]
    assert "new" not in model.loadings_
    assert newcomer.valid.any()
    print("Newcomer prediction:", newcomer.values.shape)


if __name__ == "__main__":
    main()
