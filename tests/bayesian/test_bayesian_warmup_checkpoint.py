"""Durable warmup continuation must preserve the exact sampling trajectory."""

import copy
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from multimodalsrm.bayesian import SamplerConfig, SearchConfig

from .test_bayesian_bateman_posterior import bateman_fixture
from .test_bayesian_model import make_model


def fixture(kind="bateman", method="sequential"):
    model, data = bateman_fixture(3, order=16) if kind == "bateman" else make_model("posterior")
    model.search = SearchConfig(starts=1, maxiter=3, refine_maxiter=0)
    model.sampler = SamplerConfig(
        chains=2 if kind != "single" else 1,
        warmup=30 if kind == "gaussian" else 4,
        draws=4,
        max_tree_depth=2,
        mass_matrix="dense" if kind == "gaussian" else "diagonal",
        chain_method=method,
        orientation_refresh="haar" if kind == "bateman" else "none",
    )
    return model, data


@pytest.mark.parametrize(
    "kind,method",
    [
        ("bateman", "sequential"),
        ("bateman", "vectorized"),
        ("gaussian", "sequential"),
        ("single", "sequential"),
    ],
)
def test_checkpoint_and_resume_match_uninterrupted_sampling(kind, method, tmp_path, monkeypatch):
    from numpyro.infer import MCMC

    base, data = fixture(kind, method)
    original, stored, resumed = (copy.deepcopy(base) for _ in range(3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original.fit(data)
        stored.fit(data, warmup_checkpoint=tmp_path / "warmup")
        if kind == "gaussian":
            from multimodalsrm.bayesian import _archive

            state = _archive.read(tmp_path / "warmup")["state"]["fields"]
            metric = state["adapt_state"]["fields"]["inverse_mass_matrix"]
            assert not np.allclose(metric, np.eye(metric.shape[-1]))
            off_diagonal = metric * (1 - np.eye(metric.shape[-1]))
            assert np.any(np.abs(off_diagonal) > 1e-8)

        def no_warmup(*args, **kwargs):
            raise AssertionError("resume must not repeat warmup")

        monkeypatch.setattr(MCMC, "warmup", no_warmup)
        resumed.fit(data, warmup_checkpoint=tmp_path / "warmup", resume_warmup=True)
    for model in (stored, resumed):
        assert_array_equal(model.parameter_draws_, original.parameter_draws_)
        assert_array_equal(model.log_likelihood_draws_, original.log_likelihood_draws_)
        for name in original.sample_stats_:
            assert_array_equal(model.sample_stats_[name], original.sample_stats_[name])
        assert model.sampling_diagnostics_["passes"] is False
    assert resumed.sampling_diagnostics_["warmup_checkpoint"]["resumed"] is True
    assert "warmup" not in resumed.phase_seconds_
    assert "warmup" in stored.phase_seconds_


def test_interruption_after_checkpoint_resumes_in_fresh_process(tmp_path):
    from multimodalsrm.bayesian import _archive, workflow

    model, data = fixture()
    golden = copy.deepcopy(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        golden.fit(data)
    _archive.write(tmp_path / "inputs", dict(constructor=model.get_params(deep=False), data=data))

    def interrupt(event):
        if event["phase"] == "sampling" and event["status"] == "started":
            raise RuntimeError("simulated interruption after durable warmup")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        model.fit(data, warmup_checkpoint=tmp_path / "warmup", progress=interrupt)
    assert (tmp_path / "warmup" / "manifest.sha256").exists()
    script = """
from pathlib import Path
import sys, warnings
from numpyro.infer import MCMC
from multimodalsrm.bayesian import BayesianMultimodalSRM, _archive, workflow
def no_warmup(*args, **kwargs):
    raise AssertionError('warmup was repeated')
MCMC.warmup = no_warmup
root = Path(sys.argv[1])
inputs = _archive.read(root/'inputs')
model = BayesianMultimodalSRM(**inputs['constructor'])
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    model.fit(inputs['data'], warmup_checkpoint=root/'warmup', resume_warmup=True)
workflow.save_model(root/'resumed', model)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        env=os.environ.copy(),
        timeout=120,
    )
    resumed, _ = workflow.load_model(tmp_path / "resumed")
    assert_array_equal(resumed.parameter_draws_, golden.parameter_draws_)
    for name in golden.sample_stats_:
        assert_array_equal(resumed.sample_stats_[name], golden.sample_stats_[name])
    args = dict(times=[96.0, 99.0], max_draws=2, random_state=8)
    assert_array_equal(
        resumed.sample_latent(**args)["train"].samples,
        golden.sample_latent(**args)["train"].samples,
    )


@pytest.fixture(scope="module")
def saved_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("saved-warmup") / "state"
    model, data = fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data, warmup_checkpoint=path)
    return path, model, data


@pytest.mark.parametrize(
    "change",
    [
        "data",
        "prior",
        "seed",
        "sampler",
        "draws",
        "orientation",
        "search",
        "quadrature",
        "runtime",
        "source",
    ],
)
def test_checkpoint_rejects_changed_target_or_execution(saved_checkpoint, change, monkeypatch):
    from dataclasses import replace

    from multimodalsrm.bayesian import Prior
    from multimodalsrm.bayesian import warmup_checkpoint as storage

    path, base, data = saved_checkpoint
    model, data = copy.deepcopy(base), copy.deepcopy(data)
    if change == "data":
        from multimodalsrm import TimeSeries

        ts = data["a"]["train"]["ref"]
        data["a"]["train"]["ref"] = TimeSeries(ts.values + 0.001, ts.times)
    elif change == "prior":
        model.length_scale = Prior.lognormal(np.log(3), 0.4).bounded(1, 6)
    elif change == "seed":
        model.random_state += 1
    elif change == "sampler":
        model.sampler = replace(model.sampler, target_accept=0.9)
    elif change == "draws":
        model.sampler = replace(model.sampler, draws=5)
    elif change == "orientation":
        model.sampler = replace(model.sampler, orientation_refresh="none")
    elif change == "search":
        model.search = replace(model.search, maxiter=4)
    elif change == "quadrature":
        model.response_quadrature_order = 32
    else:
        identity = storage.runtime_identity()
        identity[change] = "changed"
        monkeypatch.setattr(storage, "runtime_identity", lambda: identity)
    with pytest.raises(ValueError, match="checkpoint.*(target|runtime|identity)"):
        model.fit(data, warmup_checkpoint=path, resume_warmup=True)


def test_checkpoint_never_overwrites_and_rejects_corruption(saved_checkpoint, tmp_path):
    import shutil

    path, base, data = saved_checkpoint
    with pytest.raises(FileExistsError):
        copy.deepcopy(base).fit(data, warmup_checkpoint=path)
    bad = tmp_path / "bad"
    shutil.copytree(path, bad)
    with (bad / "arrays.npz").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="hash|archive"):
        copy.deepcopy(base).fit(data, warmup_checkpoint=bad, resume_warmup=True)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("jax_threefry_partitionable", False),
        ("jax_default_prng_impl", "rbg"),
        ("jax_random_seed_offset", 1),
    ],
)
def test_resume_rejects_changed_prng_settings_before_sampling(
    saved_checkpoint, flag, value, monkeypatch
):
    import jax
    from numpyro.infer import MCMC

    path, base, data = saved_checkpoint
    previous = getattr(jax.config, flag)
    assert previous != value

    def no_sampling(*args, **kwargs):
        raise AssertionError("changed PRNG settings reached sampling")

    monkeypatch.setattr(MCMC, "run", no_sampling)
    try:
        jax.config.update(flag, value)
        with pytest.raises(ValueError, match="checkpoint runtime identity differs"):
            copy.deepcopy(base).fit(data, warmup_checkpoint=path, resume_warmup=True)
    finally:
        jax.config.update(flag, previous)


def test_checkpoint_requires_posterior_training_and_path(tmp_path):
    model, data = make_model()
    with pytest.raises(ValueError, match="posterior"):
        model.fit(data, warmup_checkpoint=tmp_path / "state")
    model.inference = "posterior"
    with pytest.raises(ValueError, match="checkpoint"):
        model.fit(data, resume_warmup=True)
    model.random_state = None
    with pytest.raises(ValueError, match="explicit integer random_state"):
        model.fit(data, warmup_checkpoint=tmp_path / "state")


@pytest.mark.parametrize(
    "flag,value",
    [("jax_default_prng_impl", "rbg"), ("jax_enable_custom_prng", True)],
)
def test_checkpoint_rejects_unsupported_key_representation_before_map(
    tmp_path, flag, value, monkeypatch
):
    import jax

    from multimodalsrm.bayesian import model as implementation

    model, data = make_model("posterior")
    previous = getattr(jax.config, flag)

    def no_search(*args, **kwargs):
        raise AssertionError("unsupported key representation reached MAP")

    monkeypatch.setattr(implementation, "search", no_search)
    try:
        jax.config.update(flag, value)
        with pytest.raises(ValueError, match="require legacy Threefry PRNG keys"):
            model.fit(data, warmup_checkpoint=tmp_path / "state")
    finally:
        jax.config.update(flag, previous)
