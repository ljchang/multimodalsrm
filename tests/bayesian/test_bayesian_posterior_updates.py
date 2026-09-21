"""Joint targets count evidence and priors once and preserve the reference."""

import copy
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import multivariate_normal

from multimodalsrm import TimeSeries
from multimodalsrm.bayesian import (
    BayesianMultimodalSRM,
    SamplerConfig,
    SearchConfig,
    SpectralConfig,
    _archive,
)
from multimodalsrm.bayesian.blocks import ParameterSubspace
from multimodalsrm.bayesian.orthogonal import validate_orthogonal_target

from .test_bayesian_multifactor import fixture, independent, log_prior


def prepared(features=3, algebra="grouped"):
    """Prepared reference for deterministic target tests, without an MCMC claim."""
    p, x, data = fixture(features, algebra)
    model = BayesianMultimodalSRM(
        priors=p.priors,
        features=features,
        responses=p.responses,
        max_observations=p.adapter.max_observations,
        factor_anchors=tuple(p.keys[:features]),
        linear_algebra=algebra,
        search=SearchConfig(starts=1, maxiter=3),
        sampler=SamplerConfig(
            chains=2,
            warmup=4,
            draws=4,
            max_tree_depth=2,
            mass_matrix="diagonal",
            orientation_refresh="haar",
        ),
    )
    model._config()
    model.adapter_, model.problem_ = p.adapter, p
    model._factor_anchor_keys_ = tuple(p.keys[:features])
    model.specification_ = model._specification()
    model.training_data_, model.prediction_runs_ = data, dict(p.adapter.domains_)
    model.targets_ = None
    model.parameter_names_ = list(p.names)
    model.map_parameters_ = x
    model.parameter_draws_ = np.broadcast_to(x, (2, 4, len(x))).copy()
    model.configuration_ = {"inference": "posterior"}
    model.training_fit_ = {
        "inference": "posterior",
        "map_parameters": x,
        "configuration": model.configuration_,
    }
    return model


def donors(model, run="new"):
    ts = model.training_data_["a"]["train"]["brain"]
    return {"a": {run: {"brain": ts, "aux": object()}}}


@pytest.mark.parametrize("features", [3, 5])
@pytest.mark.parametrize("algebra", ["dense", "grouped"])
def test_joint_target_matches_separate_gaussian_and_one_prior(features, algebra):
    from multimodalsrm.bayesian.posterior_updates import (
        prepare_condition,
        validate_update_target,
    )

    model = prepared(features, algebra)
    before = copy.deepcopy(model.training_data_)
    result = prepare_condition(model, donors(model), targets={"a": ["aux"]})
    p, x = result.problem_, model.map_parameters_
    expected = 0.0
    for run in ("train", "new"):
        C, _, offsets, idx, _ = independent(p, x, run)
        expected -= multivariate_normal.logpdf(p.systems[run].values, mean=offsets[idx], cov=C)
    assert_allclose(p.objective(x), expected - log_prior(p, x), atol=1e-7)
    assert _archive.same(model.training_data_, before)
    assert p.names == model.problem_.names
    assert p.parameter_priors == model.problem_.parameter_priors
    assert _archive.same(p.systems["train"], model.problem_.systems["train"])
    assert result._factor_anchor_keys_ == model._factor_anchor_keys_
    assert result.prediction_runs_ == {"new": (0.0, 16.0)}
    validate_update_target(p)
    validate_orthogonal_target(ParameterSubspace(p, x))


def test_dense_grouped_update_gradient_parity():
    from multimodalsrm.bayesian.posterior_updates import prepare_condition

    targets = {"a": ["aux"]}
    a, b = prepared(3, "dense"), prepared(3, "grouped")
    pa = prepare_condition(a, donors(a), targets=targets).problem_
    pb = prepare_condition(b, donors(b), targets=targets).problem_
    va, ga = pa.value_gradient(a.map_parameters_)
    vb, gb = pb.value_gradient(b.map_parameters_)
    assert_allclose(va, vb, atol=1e-7)
    assert_allclose(ga, gb, atol=1e-7)


def test_excluded_payload_is_never_read_and_repeated_condition_replaces_donors():
    from multimodalsrm.bayesian.posterior_updates import prepare_condition

    model = prepared()
    first = prepare_condition(model, donors(model), targets={"a": ["aux"]})
    # Install only the minimal fitted fields used by reference selection.
    first.parameter_draws_ = model.parameter_draws_
    first.configuration_ = {"inference": "posterior"}
    second = prepare_condition(first, donors(model, "later"), targets={"a": ["aux"]})
    direct = prepare_condition(model, donors(model, "later"), targets={"a": ["aux"]})
    assert set(second.problem_.systems) == {"train", "later"}
    assert _archive.same(second.posterior_update_, direct.posterior_update_)


@pytest.mark.parametrize("damage", ["run", "features", "subject"])
def test_invalid_donors_rejected_before_sampling(damage):
    from multimodalsrm.bayesian.posterior_updates import prepare_condition

    model = prepared()
    data = donors(model, "train" if damage == "run" else "new")
    if damage == "features":
        ts = data["a"]["new"]["brain"]
        data["a"]["new"]["brain"] = TimeSeries(ts.values[:, :1], ts.times)
    elif damage == "subject":
        data["unknown"] = data.pop("a")
        data["unknown"]["new"].pop("aux")
    with pytest.raises(ValueError, match="overlap|feature count|mapping"):
        prepare_condition(model, data, targets={"a": ["aux"]})


@pytest.mark.parametrize("damage", ["observations", "ledger", "prior", "packing"])
def test_haar_checks_actual_update_not_only_a_marker(damage):
    from multimodalsrm.bayesian import Prior
    from multimodalsrm.bayesian.posterior_updates import prepare_condition

    model = prepared()
    result = prepare_condition(model, donors(model), targets={"a": ["aux"]})
    p = result.problem_
    space = ParameterSubspace(p, model.map_parameters_)
    if damage == "observations":
        p.systems["new"].values[0] += 10
    elif damage == "ledger":
        p._posterior_update["source_fit_id"] = "0" * 64
    elif damage == "packing":
        p._packed["new"][0][0] = 1
    else:
        p.parameter_priors[-1] = Prior.normal(0, 99)
    with pytest.raises(ValueError, match="update|evidence|target"):
        validate_orthogonal_target(space)


def test_public_joint_condition_runs_sampler_and_queries():
    model = prepared(features=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(model.training_data_)
        result = model.condition(donors(model), targets={"a": ["aux"]})
    assert result.parameter_draws_.shape[:2] == (2, 4)
    assert result.sampling_diagnostics_["orientation_refresh"] == "haar"
    assert result.configuration_["diagnostic_scope"] == "reference_plus_current_donors"
    query = result.predict(times={"new": np.array([7.0, 9.0])}, include_noise=True)
    assert np.isfinite(query["a"]["new"]["aux"].variance).all()
    assert_array_equal(result._factor_anchor_keys_, model._factor_anchor_keys_)


@pytest.mark.parametrize("damage", ["distribution", "transform", "cached_transform"])
def test_update_authenticates_effective_density_and_coordinate_transforms(damage):
    import numpyro.distributions as dist

    from multimodalsrm.bayesian.posterior_updates import prepare_condition

    model = prepared()
    p = prepare_condition(model, donors(model), targets={"a": ["aux"]}).problem_
    space = ParameterSubspace(p, model.map_parameters_)
    if damage == "distribution":
        p.distributions[0] = dist.Normal(1.0, 1.3)
    elif damage == "transform":
        p.transforms[-1] = dist.transforms.AffineTransform(1.0, 2.0)
    else:
        positions, indices, _ = space._transform_groups[-1]
        space._transform_groups = (
            *space._transform_groups[:-1],
            (positions, indices, dist.transforms.AffineTransform(1.0, 2.0)),
        )
    with pytest.raises(ValueError, match="target|transform|density"):
        validate_orthogonal_target(space)


@pytest.mark.parametrize(
    "damage", ["observations", "ids", "draw_digest", "specification", "priors"]
)
def test_reference_identity_rejects_coherent_ledger_tampering(damage):
    from dataclasses import replace

    from multimodalsrm.bayesian.posterior_updates import (
        content_id,
        prepare_condition,
        rebuild_target,
        validate_update_target,
    )

    model = prepared()
    p = prepare_condition(model, donors(model), targets={"a": ["aux"]}).problem_
    state = copy.deepcopy(p._posterior_update)
    ref = state["reference"]
    if damage == "observations":
        data = ref["adapter"]["fitted"]["_training_data"]
        ts = data["a"]["train"]["brain"]
        data["a"]["train"]["brain"] = TimeSeries(ts.values + 1, ts.times, ts.mask)
    elif damage == "ids":
        state["source_fit_id"] = ref["source_fit_id"] = "invented"
    elif damage == "draw_digest":
        ref["source_draws_digest"] = "0" * 64
    elif damage == "specification":
        state["specification"]["length_scale"] += 1
    else:
        state["priors"] = replace(state["priors"], loading_sd=3)
    # Recomputing the outer ledger identity cannot authenticate its reference.
    p._posterior_update, p._posterior_update_id = state, content_id(state)
    with pytest.raises(ValueError, match="reference|identity|specification|prior"):
        rebuild_target(state)
    with pytest.raises(ValueError, match="reference|identity|specification|prior"):
        validate_update_target(p)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"sample_blocks": ("loading",)},
        {
            "linear_algebra": "spectral",
            "spectral": SpectralConfig(rank=12, padding=12.0),
        },
        {"run_baseline_sd": {"signal": 0.3}},
    ],
)
def test_scalar_posterior_condition_preserves_legacy_configurations(options, monkeypatch):
    from .test_bayesian_model import make_model

    model, data = make_model("posterior", **options)

    # Fit preparation and public routing are real; sampling is irrelevant to
    # this compatibility check and replaced by a deterministic field installer.
    def install(self, **kwargs):
        self.parameter_draws_ = np.asarray(self.problem_.initial)[None, None, :]
        self.configuration_ = {"inference": "posterior"}
        self.training_fit_ = {"inference": "posterior"}
        self.specification_ = self._specification()
        return self

    monkeypatch.setattr(BayesianMultimodalSRM, "_fit_problem", install)
    # Public fit normally snapshots additional fitted fields; prepare its input
    # using the same observation adapter constructor for this routing-only test.
    from multimodalsrm.bayesian.observation_adapter import (
        BayesianObservationAdapter,
    )
    from multimodalsrm.bayesian.problem import BayesianProblem

    model._config()
    adapter = BayesianObservationAdapter(
        features=1,
        latent_dt=1.0,
        responses=model.responses,
        length_scale=model.length_scale,
        standardize=False,
    )
    adapter._prepare(data)
    model.adapter_ = adapter
    model.problem_ = BayesianProblem(
        adapter,
        model.priors,
        anchor=model.anchor,
        linear_algebra=model.linear_algebra,
        spectral=model.spectral,
        run_baseline_sd=model.run_baseline_sd,
    )
    model.training_data_ = adapter._training_data
    install(model)
    result = model.condition({"a": {"new": data["a"]["train"]}}, targets={"b": ["signal"]})
    assert result.sample_blocks == model.sample_blocks
    assert result.problem_.linear_algebra == model.linear_algebra
    assert result.problem_.run_baseline_sd == model.problem_.run_baseline_sd
    if options:
        assert result.configuration_["diagnostic_scope"] == "training_plus_donors"
        assert not hasattr(result, "posterior_update_")
    else:
        assert result.configuration_["diagnostic_scope"] == "reference_plus_current_donors"
        assert result.posterior_update_["kind"] == "donor"


@pytest.mark.parametrize("damage", ["mean", "scale", "shape", "extra", "responses"])
def test_rebuild_rejects_invalid_reference_contract_even_with_new_identity(damage):
    from multimodalsrm import Gaussian, Response
    from multimodalsrm.bayesian.posterior_updates import (
        content_id,
        prepare_condition,
        rebuild_target,
    )

    model = prepared()
    state = copy.deepcopy(
        prepare_condition(model, donors(model), targets={"a": ["aux"]}).posterior_update_
    )
    ref = state["reference"]
    fitted = ref["adapter"]["fitted"]
    stats = fitted["preprocessing_"]["a"]["brain"]
    if damage == "mean":
        stats["mean"] = stats["mean"] + 1
    elif damage == "scale":
        stats["scale"] = stats["scale"] * 2
    elif damage == "shape":
        stats["constant_features"] = np.zeros(1, bool)
    elif damage == "extra":
        fitted["preprocessing_"]["a"]["extra"] = copy.deepcopy(stats)
    else:
        fitted["responses_"]["brain"] = Response(
            Gaussian(1.0, 0.0), pooling="shared", estimate=False
        )
    new_id = content_id({k: v for k, v in ref.items() if k != "source_fit_id"})
    assert new_id != state["source_fit_id"]
    ref["source_fit_id"] = state["source_fit_id"] = new_id
    with pytest.raises(ValueError, match="preprocessing|response|mapping"):
        rebuild_target(state)


@pytest.mark.parametrize("damage", ["mean", "scale", "shape", "extra"])
def test_rebuild_rejects_invalid_added_preprocessing(damage):
    from multimodalsrm.bayesian.posterior_participants import (
        prepare_calibration,
    )
    from multimodalsrm.bayesian.posterior_updates import rebuild_target

    from .test_bayesian_posterior_participants import calibration

    model = prepared()
    state = copy.deepcopy(prepare_calibration(model, calibration(model)).posterior_update_)
    added = state["added_preprocessing"]["newperson"]
    if damage in ("mean", "scale"):
        added["brain"][damage] = added["brain"][damage] + 1
    elif damage == "shape":
        added["brain"]["constant_features"] = np.zeros(1, bool)
    else:
        added["extra"] = copy.deepcopy(added["brain"])
    with pytest.raises(ValueError, match="preprocessing|mapping"):
        rebuild_target(state)
