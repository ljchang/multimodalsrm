"""Independent density, normalization and cell-integral oracles for Bateman SCR."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from multimodalsrm import BatemanSCR
from multimodalsrm.operators import observation_operator, response_operator_derivatives


@pytest.mark.parametrize(
    "rise,decay", [(0.7, 3), (2, 2), (2, 2 + 1e-9), (3, 0.7), (0.03, 4), (20, 30)]
)
def test_density_and_finite_energy_against_independent_convolution(rise, decay):
    kernel = BatemanSCR(rise, decay, 0.25)

    def density(t):
        return quad(
            lambda u: np.exp(-u / rise - (t - u) / decay) / (rise * decay),
            0,
            t,
            epsabs=1e-14,
            epsrel=1e-12,
        )[0]

    times = np.array([0, 0.001, 0.1, 1, 3, 10, 40, 90])
    expected = np.array([density(t) for t in times])
    assert_allclose(kernel._raw(times + kernel.lag), expected, atol=3e-14, rtol=2e-10)
    integral = quad(lambda t: density(t) ** 2, 0, 90, epsabs=1e-13)[0]
    assert_allclose(kernel._energy**2, integral, rtol=3e-11, atol=1e-13)
    assert_allclose(kernel(np.array([-0.1, 91]) + kernel.lag), 0)
    assert set(kernel.parameters) == {"rise", "decay", "lag"}


@pytest.mark.parametrize("rise,decay", [(0.7, 3), (2, 2), (0.03, 4), (3, 0.7)])
def test_cell_integrals_and_physical_derivatives(rise, decay):
    grid = np.arange(-100, 101, 2.0)
    times = np.array([-95, -5.123, 0.345, 10.231, 99.9])
    kernel = BatemanSCR(rise, decay, 0.17)
    H, derivatives, valid = response_operator_derivatives(grid, times, kernel, (-1, 91))
    z = np.sin(grid / 3) + np.cos(grid / 7)
    for i in np.flatnonzero(valid):
        expected = quad(
            lambda u: float(kernel(u) * np.interp(times[i] - u, grid, z)),
            *kernel.support,
            points=(times[i] - grid)[
                ((times[i] - grid) > kernel.support[0]) & ((times[i] - grid) < kernel.support[1])
            ],
            epsabs=1e-10,
            limit=300,
        )[0]
        assert_allclose((H @ z)[i], expected, atol=2e-10)
    for p, value in kernel.parameters.items():
        step = 1e-5 * max(abs(value), 0.1)
        plus = observation_operator(
            grid, times, kernel.with_parameters(**{p: value + step}), (-1, 91)
        )[0]
        minus = observation_operator(
            grid, times, kernel.with_parameters(**{p: value - step}), (-1, 91)
        )[0]
        assert_allclose(
            derivatives[p].toarray(), (plus - minus).toarray() / (2 * step), atol=2e-8, rtol=3e-6
        )


def test_invalid_constants_rejected():
    for kwargs in ({"rise": 0}, {"decay": -1}, {"lag": np.nan}):
        with pytest.raises(ValueError):
            BatemanSCR(**kwargs)


def test_r_fit_learns_bateman_without_finite_difference_fallback(monkeypatch):
    from multimodalsrm import Identity, MultimodalSRM, Response, TimeSeries
    from multimodalsrm.kernel_optimization import KernelObjective

    original = KernelObjective.value_gradient
    calls = []

    def checked(objective, kernels):
        calls.append(objective.gradient_method)
        return original(objective, kernels)

    monkeypatch.setattr(KernelObjective, "value_gradient", checked)
    times = np.arange(0.0, 110.0, 1.0)
    data = {
        s: {
            "run": {
                "ref": TimeSeries(np.sin(times / 3)[:, None], times),
                "scr": TimeSeries(
                    (np.sin((times - 2) / 3) + 0.1 * np.cos(times / 7))[:, None], times
                ),
            }
        }
        for s in ("a", "b")
    }
    model = MultimodalSRM(
        features=1,
        latent_dt=1.0,
        responses={
            "ref": Response(Identity(), estimate=False),
            "scr": Response(
                BatemanSCR(),
                pooling="shared",
                bounds={"rise": (0.4, 1), "decay": (2, 4), "lag": (-1, 1)},
            ),
        },
        init="hybrid",
        max_iter=3,
        kernel_max_iter=3,
        random_state=11,
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
    assert calls and set(calls) == {"complex_step"}
    assert all(
        np.isfinite(k.parameters["rise"])
        for mods in model.subject_kernels_.values()
        for k in mods.values()
        if type(k) is BatemanSCR
    )
