import numpy as np
import pytest
from scipy.integrate import quad
from scipy.sparse import isspmatrix_csr

from multimodalsrm.data import TimeSeries, normalize_data
from multimodalsrm.kernels import (
    BatemanSCR,
    DoubleGamma,
    Gamma,
    Gaussian,
    Identity,
    KernelPrior,
    Normal,
    Response,
    SampledKernel,
)
from multimodalsrm.operators import observation_operator
from multimodalsrm.results import SeriesResult


def test_data_copy_mask_and_shorthand():
    x = np.array([[1.0, np.nan], [2.0, 3.0]])
    ts = TimeSeries(x, [0, 1], [[True, False], [True, True]])
    x[0, 0] = 20
    assert ts.values[0, 0] == 1
    assert not ts.values.flags.writeable
    assert normalize_data({"s": {"m": ts}})["s"]["run-01"]["m"] is ts


@pytest.mark.parametrize(
    "values,times,mask",
    [
        ([], [], None),
        ([[1]], [np.inf], None),
        ([[1], [2]], [1, 1], None),
        ([[np.nan]], [0], None),
        ([[1]], [0], [[0]]),
        ([[1]], [0], [[False]]),
        ([1, 2], [0, 1], None),
    ],
)
def test_invalid_series(values, times, mask):
    with pytest.raises(ValueError):
        TimeSeries(values, times, mask)


def test_mixed_nesting_and_feature_count():
    t = TimeSeries([[1]], [0])
    for d in (
        {"s": {"m": t}, "b": {"r": {"m": t}}},
        {"s": {"m": t, "r": {"m": t}}},
        {"s": {}},
        {},
    ):
        with pytest.raises(ValueError):
            normalize_data(d)
    with pytest.raises(ValueError):
        normalize_data({"s": {"a": {"m": t}, "b": {"m": TimeSeries([[1, 2]], [0])}}})


@pytest.mark.parametrize("dt", [0.2, 0.1])
def test_signed_convolution_linear_reference(dt):
    grid = np.arange(0, 10 + dt / 2, dt)
    k = SampledKernel([-1, 0, 0.5, 1], [0, 1, -0.5, 0])
    times = np.array([2.03, 4.17, 8.08])
    op, valid = observation_operator(grid, times, k)
    ref = [
        quad(lambda u: float(k.evaluate(u)) * (2 * (t - u) + 3), -1, 1, points=[0, 0.5])[0]
        for t in times
    ]
    assert isspmatrix_csr(op)
    assert valid.all()
    np.testing.assert_allclose(op @ (2 * grid + 3), ref, atol=1e-9)


def test_identity_exact_interpolation_and_boundary():
    op, valid = observation_operator([0, 1, 2], [-1, 0, 0.25, 2, 3], Identity())
    np.testing.assert_array_equal(valid, [False, True, True, True, False])
    np.testing.assert_allclose(op @ np.array([1.0, 3.0, 5.0]), [0, 1, 1.5, 5, 0])


def test_impulse_timing_and_fixed_envelope():
    dt = 0.01
    grid = np.arange(0, 10 + dt / 2, dt)
    k = Gaussian(width=0.2, lag=0.5)
    times = np.array([4.0, 4.5, 5.0])
    op, valid = observation_operator(grid, times, k)
    impulse = np.zeros(len(grid))
    impulse[400] = 1 / dt
    np.testing.assert_allclose(op @ impulse, k.evaluate(times - 4), atol=0.002)
    _, v1 = observation_operator(grid, [1, 3, 7, 9], k, support=(-2, 3))
    _, v2 = observation_operator(grid, [1, 3, 7, 9], Gaussian(width=0.1), support=(-2, 3))
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(v1, [False, True, True, False])


@pytest.mark.parametrize(
    "kernel",
    [
        Gaussian(),
        Gamma(),
        DoubleGamma(),
        BatemanSCR(),
        SampledKernel([0, 1, 2], [0, 1, 0]),
    ],
)
def test_continuous_energy_and_metadata(kernel):
    assert quad(
        lambda t: float(kernel.evaluate(t)) ** 2,
        *kernel.support,
        epsabs=1e-7,
        limit=200,
    )[0] == pytest.approx(1, abs=1e-6)
    assert kernel.metadata["normalization"] == "continuous_l2"
    np.testing.assert_allclose(
        kernel.from_coordinates(kernel.to_coordinates()).evaluate([0, 1, 2]),
        kernel.evaluate([0, 1, 2]),
    )


def test_response_validation_fixed_and_transforms():
    r = Response(Gaussian(), fixed={"lag": 0.5}, prior=KernelPrior({"width": 1}, 2))
    assert r.initial_kernel().parameters["lag"] == 0.5
    assert r.free_parameters == ("width",)
    assert Response(Identity()).free_parameters == ()
    assert Response(SampledKernel([0, 1], [1, -1])).free_parameters == ()
    assert Response(Gamma(), estimate=False).free_parameters == ()
    env = r.support_envelope()
    assert env[0] <= r.initial_kernel().support[0] and env[1] >= r.initial_kernel().support[1]
    for fn in [
        lambda: Normal(0, 0),
        lambda: Gaussian(width=0),
        lambda: Gamma(shape=-1),
        lambda: DoubleGamma(undershoot_ratio=-1),
        lambda: BatemanSCR(rise=0),
        lambda: Response(Gaussian(), pooling="bad"),
        lambda: Response(Gaussian(), fixed={"foo": 1}),
        lambda: SampledKernel([0, 1], [0, 0]),
        lambda: Response(Gaussian(), bounds={"width": (0, 1)}),
    ]:
        with pytest.raises(ValueError):
            fn()


def test_result_copies_and_marks_invalid():
    v = np.array([[1.0], [2.0]])
    r = SeriesResult(v, [0, 1], [True, False], {"source": "test"})
    assert np.isnan(r.values[1, 0])
    assert not r.values.flags.writeable
    assert r.metadata["source"] == "test"


def test_narrow_kernel_is_integrated_on_coarse_latent_grid():
    k = Gaussian(width=0.001)
    op, valid = observation_operator(np.arange(5.0), [2.3], k)
    expected = quad(lambda u: float(k.evaluate(u)) * (2.3 - u), *k.support)[0]
    assert valid.all()
    np.testing.assert_allclose(op @ np.arange(5.0), [expected], rtol=1e-7)


def test_double_gamma_envelope_covers_valid_interior_with_invalid_corner():
    kernel = DoubleGamma(peak_shape=1, peak_scale=1, undershoot_shape=100, undershoot_scale=0.12)
    fixed = {k: v for k, v in kernel.parameters.items() if k != "peak_scale"}
    response = Response(kernel, fixed=fixed, bounds={"peak_scale": (1, 20)})
    candidate = kernel.with_parameters(peak_scale=11)
    assert response.support_envelope()[1] >= candidate.support[1]


@pytest.mark.parametrize("ratio", [0.999, 0.0001])
@pytest.mark.parametrize("estimate", [True, False])
def test_ratio_default_bounds_contain_valid_initialization(ratio, estimate):
    response = Response(DoubleGamma(undershoot_ratio=ratio), estimate=estimate)
    lo, hi = response.parameter_bounds()["undershoot_ratio"]
    assert 0 < lo <= ratio <= hi < 1
    assert response.initial_kernel().undershoot_ratio == ratio


def test_fixed_values_are_not_restricted_by_inactive_optimization_bounds():
    response = Response(
        DoubleGamma(),
        fixed={"undershoot_ratio": 0.999},
        bounds={"undershoot_ratio": (0.1, 0.5)},
    )
    assert response.initial_kernel().undershoot_ratio == 0.999
    assert "undershoot_ratio" not in response.free_parameters
