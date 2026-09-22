"""Legacy response archives fail with actionable guidance before model rebuilding."""

from dataclasses import make_dataclass

import pytest

from multimodalsrm import BatemanSCR, Response
from multimodalsrm.bayesian import _archive
from multimodalsrm.bayesian.workflow import load_model


def test_bach_archive_reports_refit_requirement_before_rebuilding(tmp_path, monkeypatch):
    from multimodalsrm.bayesian import persistence

    # Recreate the historical data record only, without retaining its kernel
    # implementation. Its serialized identifier and fields are the original ones.
    legacy_type = make_dataclass(
        "BachSCR",
        [
            (name, type(value), value)
            for name, value in {
                "version": "2010",
                "t0": 3.0745,
                "sigma": 0.7013,
                "lambda1": 0.3176,
                "lambda2": 0.0708,
                "lag": 0.0,
            }.items()
        ],
        frozen=True,
    )
    with monkeypatch.context() as patch:
        patch.setitem(_archive.CLASSES, "BachSCR", legacy_type)
        _archive.write(tmp_path / "legacy", {"constructor": {"responses": {"scr": legacy_type()}}})

    def forbidden(*args, **kwargs):
        pytest.fail("a removed response must be rejected before model preparation")

    monkeypatch.setattr(persistence, "_prepare", forbidden)
    with pytest.raises(ValueError, match="BachSCR support has been removed") as caught:
        load_model(tmp_path / "legacy")
    message = str(caught.value)
    assert "0.1.0" in message
    assert "BatemanSCR" in message and "refit" in message
    assert "cannot be converted automatically" in message


def test_supported_response_archive_still_roundtrips(tmp_path):
    state = {"responses": {"scr": Response(BatemanSCR(), estimate=False, pooling="shared")}}
    _archive.write(tmp_path / "bateman", state)
    assert _archive.same(_archive.read(tmp_path / "bateman"), state)
