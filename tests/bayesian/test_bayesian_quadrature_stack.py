"""Order-32 Gaussian quadrature must not overflow an XLA worker-thread stack."""

import subprocess
import sys

from .test_bayesian_problem import api

CODE = """
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from multimodalsrm import Gaussian, Response
from multimodalsrm.bayesian.response_quadrature import ResponseQuadrature

quadrature = ResponseQuadrature(
    {"face": Response(Gaussian(1.0, 0.5), estimate=False)}, {}, 3.0, 32
)
times = np.arange(0.0, 60.0, 0.5)
modalities = np.zeros(len(times), dtype=int)
for _ in range(8):
    value = quadrature.covariance(None, times, modalities, times, modalities)
    assert np.isfinite(np.asarray(value)).all()
"""


def test_order_32_gaussian_covariance_survives_eager_evaluation():
    # 32 Gauss-Legendre points on the 8 Gaussian panels give 256 nodes; with the
    # former fixed batch of 32 that made 64 KiB float64 intermediates, which
    # jaxlib 0.11 stack-allocated until the process died with a bus error. The
    # crash kills the interpreter, so it has to be observed from outside.
    api()
    result = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True)
    assert result.returncode == 0, f"exit {result.returncode}\n{result.stderr[-2000:]}"
