"""Small recordings with one known departure from the fitted model at a time.

Two modalities observe the same smooth latent without any response delay, on
the same one-second clock. Departures are opt-in: a participant-specific
signal that reaches modality ``b`` a known number of seconds after modality
``a``, and autocorrelated private noise. Features are standardized.
"""

import numpy as np

from multimodalsrm import TimeSeries


def _smooth(rng, samples, columns, width=2.0):
    """Unit-variance Gaussian-smoothed white noise, time by column."""
    reach = int(4 * width)
    kernel = np.exp(-0.5 * (np.arange(-reach, reach + 1) / width) ** 2)
    white = rng.normal(size=(samples + 2 * reach, columns))
    paths = np.column_stack([np.convolve(white[:, j], kernel, "valid") for j in range(columns)])
    return (paths - paths.mean(0)) / paths.std(0)


def _noise(rng, shape, rho):
    white = rng.normal(size=shape)
    if not rho:
        return white
    out = np.empty(shape)
    out[0] = white[0]
    for i in range(1, shape[0]):
        out[i] = rho * out[i - 1] + np.sqrt(1 - rho**2) * white[i]
    return out


def recording(
    *,
    participants=6,
    runs=1,
    samples=100,
    features=(5, 4),
    factors=2,
    private=0.0,
    delay=3,
    ar=0.0,
    seed=0,
):
    """Return ``{participant: {run: {"a": TimeSeries, "b": TimeSeries}}}``.

    ``private`` is the variance of the participant-specific signal relative to
    the shared signal, ``delay`` its arrival in ``b`` after ``a`` in seconds,
    and ``ar`` the lag-one autocorrelation of the private noise. Shared signal
    and noise have equal variance. Loadings are fixed across runs.
    """
    rng = np.random.default_rng(seed)
    times = np.arange(float(samples))
    latents = [_smooth(rng, samples, factors) for _ in range(runs)]
    data = {}
    for p in range(participants):
        loadings = [rng.normal(size=(count, factors)) / np.sqrt(factors) for count in features]
        mixing = [rng.normal(size=count) for count in features]
        values = {"a": [], "b": []}
        for latent in latents:
            own = _smooth(rng, samples + delay, 1)[:, 0]
            # ``a`` sees the participant's own signal ``delay`` samples before ``b``.
            for modality, w, mix, shift in zip("ab", loadings, mixing, (delay, 0)):
                x = latent @ w.T + _noise(rng, (samples, len(w)), ar)
                if private:
                    x += np.sqrt(private) * np.outer(own[shift : shift + samples], mix)
                values[modality].append(x)
        data[f"p{p}"] = {}
        for modality, parts in values.items():
            stacked = np.concatenate(parts)
            center, scale = stacked.mean(0), stacked.std(0)
            for r, x in enumerate(parts):
                data[f"p{p}"].setdefault(f"run{r}", {})[modality] = TimeSeries(
                    (x - center) / scale, times
                )
    return data
