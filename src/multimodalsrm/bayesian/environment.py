"""One-time advice about the numerical environment of a GP-MSRM fit.

The package never installs anything or changes process-global settings. When
an NVIDIA card is visible but JAX initialized its CPU backend, it says once
per process how to use the card; the companion thread-count warning lives in
:mod:`multimodalsrm.bayesian.factorization`.
"""

import importlib.util
import logging
import os
import platform
import shutil
import sys
import warnings
from pathlib import Path

from ._backend import runtime

logger = logging.getLogger(__name__)

SYSTEM = sys.platform
MACHINE = platform.machine()
CUDA_PLUGIN_MODULES = ("jax_plugins.xla_cuda13", "jax_plugins.xla_cuda12")
PLATFORM_VARIABLES = ("JAX_PLATFORMS", "JAX_PLATFORM_NAME")
DRIVER_PROC_FILE = Path("/proc/driver/nvidia/version")
INSTALL_COMMAND = "python -m pip install 'multimodalsrm[bayesian-cuda]'"


def nvidia_driver_present():
    """True when the NVIDIA kernel driver or its management tool is visible."""
    return DRIVER_PROC_FILE.exists() or shutil.which("nvidia-smi") is not None


def cuda_plugin_installed(modules=CUDA_PLUGIN_MODULES):
    """True when a JAX CUDA PJRT plugin is importable, without importing JAX."""
    for name in modules:
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except (ImportError, ValueError):
            continue
    return False


def requested_platform(environ=None):
    """The platform the user pinned through JAX's environment variables, or None."""
    environ = os.environ if environ is None else environ
    for name in PLATFORM_VARIABLES:
        value = environ.get(name, "").strip().lower()
        if value:
            return value
    return None


def cuda_advice(*, backend, driver, plugin, requested, system=None, machine=None):
    """Describe an unused NVIDIA card, or return None when nothing should be said.

    Silent when JAX runs on an accelerator, when the user pinned a platform
    through the environment (``JAX_PLATFORMS=cpu`` is a deliberate choice),
    when no driver is visible, and on every platform other than Linux x86_64,
    which is the only platform the CUDA wheels support. Apple MPS has no
    float64 JAX backend, so macOS never receives advice.
    """
    system = SYSTEM if system is None else system
    machine = MACHINE if machine is None else machine
    if backend != "cpu" or requested is not None:
        return None
    if not (system == "linux" and machine == "x86_64" and driver):
        return None
    if not plugin:
        return (
            "An NVIDIA driver is present but JAX initialized its cpu backend because no "
            "CUDA plugin is installed. GP-MSRM runs on CUDA in float64 without code "
            "changes, and its likelihood gradient ran 3x to 10x faster on the recorded "
            f"RTX 3090 and RTX PRO 6000 than on the CPU. Install it with {INSTALL_COMMAND} "
            "(Linux x86_64 only), or set JAX_PLATFORMS=cpu before starting Python to keep "
            "the CPU backend and silence this warning. See the getting-started "
            "documentation on NVIDIA GPUs."
        )
    return (
        "A JAX CUDA plugin and an NVIDIA driver are both present, but JAX initialized its "
        "cpu backend. Check CUDA_VISIBLE_DEVICES, the driver version against the installed "
        "CUDA runtime, and JAX's own startup messages, or set JAX_PLATFORMS=cpu before "
        "starting Python to keep the CPU backend and silence this warning."
    )


_advice_issued = False


def check_environment_once():
    """Warn once per process about an unused NVIDIA card; log the backend in use.

    Called at the start of a fit, after configuration validation. Initializes
    the JAX backend, which a fit does anyway; configure devices in the process
    environment before starting Python.
    """
    global _advice_issued
    if _advice_issued:
        return
    _advice_issued = True
    jax, _, _, _ = runtime()
    backend = jax.default_backend()
    message = cuda_advice(
        backend=backend,
        driver=nvidia_driver_present(),
        plugin=cuda_plugin_installed(),
        requested=requested_platform(),
    )
    if message is not None:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    logger.info(
        "GP-MSRM JAX backend %s with devices %s", backend, [str(d) for d in jax.local_devices()]
    )
