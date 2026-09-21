"""Parallel requests must run as requested and retain chain/draw identity."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from .test_bayesian_problem import api


def fresh_process(script, devices):
    api()
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "JAX_ENABLE_X64": "true",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        "JAX_PLATFORMS": "cpu",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=root,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_parallel_request_rejects_insufficient_devices_before_problem_access():
    fresh_process(
        """
from multimodalsrm.bayesian import SamplerConfig
from multimodalsrm.bayesian.fitting import sample
try:
    sample(None, [], SamplerConfig(chains=4, chain_method='parallel'), 1)
except ValueError as exc:
    assert 'devices' in str(exc), str(exc)
else:
    raise AssertionError('parallel request silently fell back')
""",
        1,
    )


def test_real_parallel_sampling_retains_all_draws_and_execution_metadata():
    fresh_process(
        """
import numpy as np
from tests.bayesian.test_bayesian_problem import problem_fixture
from multimodalsrm.bayesian import SamplerConfig
from multimodalsrm.bayesian.fitting import sample
p, _, _ = problem_fixture(False)
records = [dict(start=0, parameters=p.initial.tolist(), objective=float(p.objective(p.initial)))]
draws, stats, loglik, diagnostic = sample(
    p, records, SamplerConfig(chains=4, warmup=8, draws=8, max_tree_depth=4,
                            chain_method='parallel'), 917)
assert draws.shape == (4, 8, len(p.names))
assert loglik.shape == (4, 8)
assert stats['num_steps'].shape == (4, 8)
assert np.isfinite(draws).all() and np.isfinite(loglik).all()
assert np.any(draws[0] != draws[1])
execution = diagnostic['execution']
assert execution['requested_chain_method'] == 'parallel'
assert execution['effective_chain_method'] == 'parallel'
assert execution['local_device_count'] == 4
assert execution['backend'] == 'cpu'
assert len(execution['devices']) == 4
assert execution['parallel_verified'] is True
assert diagnostic['passes'] is False
""",
        4,
    )


def test_effective_mode_mismatch_is_rejected():
    api()
    from multimodalsrm.bayesian import SamplerConfig
    from multimodalsrm.bayesian.execution import execution_info

    with pytest.raises(RuntimeError, match="fallback"):
        execution_info(SamplerConfig(chain_method="sequential"), effective_method="parallel")
