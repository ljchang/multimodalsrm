"""Known-truth and active-coordinate contracts for the recovery screen."""

import numpy as np
import pytest


def test_interrupted_search_retains_completed_and_inflight_restarts(tmp_path):
    import json

    from multimodalsrm.bayesian.fitting import SearchConfig, search

    from .test_bayesian_problem import problem_fixture

    p, _, _ = problem_fixture(gaussian=True)
    path = tmp_path / "progress.json"
    events = []

    def progress(event):
        events.append(event)
        path.write_text(json.dumps(events))
        if event["phase"] == "started" and event["record"]["start"] == 1:
            raise RuntimeError("simulated worker interruption")

    with pytest.raises(RuntimeError, match="simulated worker interruption"):
        search(p, SearchConfig(starts=2, maxiter=2), 314, progress=progress)
    saved = json.loads(path.read_text())
    boundaries = [e for e in saved if e["phase"] != "iteration"]
    assert [e["phase"] for e in boundaries] == ["started", "finished", "started"]
    assert np.isfinite(boundaries[1]["record"]["objective"])
    assert boundaries[2]["record"]["start"] == 1
    assert len(boundaries[2]["record"]["initial_parameters"]) == len(p.names)
