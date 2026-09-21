"""Instrumentation preserves the sampler trajectory and exposes failures."""

import numpy as np
import pytest

from .test_bayesian_parallel import fresh_process
from .test_bayesian_problem import api, problem_fixture


@pytest.mark.parametrize("method", ["sequential", "vectorized"])
def test_phase_checkpoints_preserve_sampler_draws_and_statistics(method):
    b = api()
    from multimodalsrm.bayesian.fitting import sample

    p, _, _ = problem_fixture(True)
    records = [
        dict(
            start=0,
            parameters=p.initial.tolist(),
            objective=float(p.objective(p.initial)),
        )
    ]
    config = b.SamplerConfig(chains=2, warmup=12, draws=12, max_tree_depth=5, chain_method=method)
    original = sample(p, records, config, 941)
    events = []
    measured = sample(p, records, config, 941, progress=events.append)
    np.testing.assert_array_equal(original[0], measured[0])
    np.testing.assert_array_equal(original[2], measured[2])
    for key in original[1]:
        np.testing.assert_array_equal(original[1][key], measured[1][key])
    finished = {e["phase"] for e in events if e["status"] == "completed"}
    assert {
        "initialization",
        "warmup",
        "sampling",
        "physical_draws",
        "log_likelihood",
        "diagnostics",
    } <= finished
    assert all(e["elapsed_seconds"] >= 0 for e in events if e["status"] == "completed")
    assert measured[3]["phase_seconds"]["warmup"] > 0
    assert measured[3]["passes"] is False


def test_parallel_phase_split_preserves_the_advanced_rng_keys():
    fresh_process(
        """
import numpy as np
from tests.bayesian.test_bayesian_problem import problem_fixture
from multimodalsrm.bayesian import SamplerConfig
from multimodalsrm.bayesian.fitting import sample
p, _, _ = problem_fixture(True)
r = [dict(start=0, parameters=p.initial.tolist(), objective=float(p.objective(p.initial)))]
c = SamplerConfig(chains=4, warmup=12, draws=12, max_tree_depth=5, chain_method='parallel')
a = sample(p,r,c,944)
events=[]
b = sample(p,r,c,944,progress=events.append)
np.testing.assert_array_equal(a[0],b[0])
np.testing.assert_array_equal(a[2],b[2])
for k in a[1]: np.testing.assert_array_equal(a[1][k],b[1][k])
assert any(e['phase']=='warmup' and e['status']=='completed' for e in events)
assert b[3]['execution']['parallel_verified']
""",
        4,
    )


def test_failed_phase_checkpoint_is_kept_before_error_propagates():
    api()
    from multimodalsrm.bayesian import timing

    events = []
    recorder = timing.PhaseTimings(events.append)
    with pytest.raises(RuntimeError, match="deliberate failure"):
        with recorder.phase("warmup"):
            raise RuntimeError("deliberate failure")
    assert [e["status"] for e in events] == ["started", "failed"]
    assert events[-1]["phase"] == "warmup"
    assert recorder.seconds["warmup"] >= 0


def test_fit_and_donor_condition_report_map_progress_without_reading_targets():
    b = api()
    p, adapter, data = problem_fixture(True)
    model = b.BayesianMultimodalSRM(
        responses=adapter.responses_,
        priors=p.priors,
        anchor=p.anchor,
        inference="map",
        search=b.SearchConfig(starts=1, maxiter=50),
    )
    events = []
    model.fit(data, progress=events.append)
    assert any(e["phase"] == "map" and e["status"] == "completed" for e in events)
    donors = {s: {"new": dict(runs["train"])} for s, runs in data.items()}
    donors["b"]["new"]["signal"] = object()
    events.clear()
    conditioned = model.condition(donors, targets={"b": ["signal"]}, progress=events.append)
    assert conditioned.phase_seconds_["map"] > 0
    assert any(e["phase"] == "map" and e["status"] == "completed" for e in events)
