"""Advice about an unused NVIDIA card is accurate, silent when pinned, and one-shot."""

import warnings

from .test_bayesian_model import make_model
from .test_bayesian_problem import api

LINUX = dict(system="linux", machine="x86_64")


def test_advice_only_for_cpu_backend_on_linux_x86_64_with_a_visible_driver():
    api()
    from multimodalsrm.bayesian.environment import INSTALL_COMMAND, cuda_advice

    missing = cuda_advice(backend="cpu", driver=True, plugin=False, requested=None, **LINUX)
    assert INSTALL_COMMAND in missing and "JAX_PLATFORMS=cpu" in missing
    unused = cuda_advice(backend="cpu", driver=True, plugin=True, requested=None, **LINUX)
    assert "CUDA_VISIBLE_DEVICES" in unused and INSTALL_COMMAND not in unused
    # The card is in use, or the user pinned a platform on purpose.
    assert cuda_advice(backend="gpu", driver=True, plugin=True, requested=None, **LINUX) is None
    assert cuda_advice(backend="cpu", driver=True, plugin=False, requested="cpu", **LINUX) is None
    # No driver, or a platform without CUDA wheels: Apple silicon and Linux aarch64.
    assert cuda_advice(backend="cpu", driver=False, plugin=False, requested=None, **LINUX) is None
    for system, machine in [("darwin", "arm64"), ("linux", "aarch64"), ("win32", "AMD64")]:
        assert (
            cuda_advice(
                backend="cpu",
                driver=True,
                plugin=False,
                requested=None,
                system=system,
                machine=machine,
            )
            is None
        )


def test_requested_platform_reads_either_jax_variable():
    api()
    from multimodalsrm.bayesian.environment import requested_platform

    assert requested_platform({}) is None
    assert requested_platform({"JAX_PLATFORMS": ""}) is None
    assert requested_platform({"JAX_PLATFORMS": "cpu"}) == "cpu"
    assert requested_platform({"JAX_PLATFORM_NAME": " CPU "}) == "cpu"
    assert requested_platform({"JAX_PLATFORMS": "cuda", "JAX_PLATFORM_NAME": "cpu"}) == "cuda"


def test_plugin_detection_does_not_import_jax_and_tolerates_missing_parents():
    api()
    from multimodalsrm.bayesian.environment import cuda_plugin_installed

    assert cuda_plugin_installed(("no_such_package.xla_cuda13",)) is False
    assert cuda_plugin_installed(("os.path",)) is True
    assert cuda_plugin_installed(()) is False
    assert cuda_plugin_installed() in (True, False)


def test_fit_warns_once_per_process_and_never_when_the_platform_is_pinned(monkeypatch):
    from multimodalsrm.bayesian import environment

    model, data = make_model()
    model.set_params(search=api().SearchConfig(starts=1, maxiter=50))
    monkeypatch.setattr(environment, "SYSTEM", "linux")
    monkeypatch.setattr(environment, "MACHINE", "x86_64")
    monkeypatch.setattr(environment, "nvidia_driver_present", lambda: True)
    monkeypatch.setattr(environment, "cuda_plugin_installed", lambda: False)
    monkeypatch.setattr(environment, "requested_platform", lambda: None)
    monkeypatch.setattr(environment, "_advice_issued", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data)
    advice = [
        w for w in caught if issubclass(w.category, RuntimeWarning) and "NVIDIA" in str(w.message)
    ]
    import jax

    if jax.default_backend() == "cpu":
        assert len(advice) == 1, [str(w.message) for w in caught]
        assert environment.INSTALL_COMMAND in str(advice[0].message)
    else:
        assert advice == []
    # Second fit in the same process: silent.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data)
    assert not [w for w in caught if "NVIDIA" in str(w.message)]
    # A pinned platform is a deliberate choice, so a fresh process stays silent.
    monkeypatch.setattr(environment, "requested_platform", lambda: "cpu")
    monkeypatch.setattr(environment, "_advice_issued", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data)
    assert not [w for w in caught if "NVIDIA" in str(w.message)]
