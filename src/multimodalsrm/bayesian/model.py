"""Shared GP SRM with MAP and experimental posterior inference."""

import copy
import warnings
from collections.abc import Mapping
from dataclasses import asdict

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted

from ..data import readonly_array
from .blocks import validate_blocks
from .environment import check_environment_once
from .fitting import SamplerConfig, SearchConfig, sample, search
from .gp_hyperparameters import (
    LENGTH_SCALE,
    length_scale_settings,
    validate_length_scale_scope,
)
from .problem import BayesianProblem


class MAPFitError(RuntimeError):
    """No finite MAP candidate, with records retained across conditioning."""

    def __init__(self, restarts):
        super().__init__("no finite MAP fit; inspect restart_diagnostics")
        self.restart_diagnostics = copy.deepcopy(restarts)


class BayesianMultimodalSRM(BaseEstimator):
    """Experimental Bayesian SRM with a shared GP timescale.

    ``features>1`` supports MAP with dense/grouped/state_space algebra. Optional
    ``factor_anchors`` select a training QR reporting rotation; without them,
    the fitted optimizer coordinates are retained. Explicit anchors must name
    one observed training feature per factor, starting with ``anchor`` when
    supplied. Raw optimizer parameters remain unchanged. For MAP coordinates,
    use loadings ``W_raw @ configuration_['factor_orientation']['rotation']``.
    All factors share modality filters and GP timescale. Multifactor posterior
    supports dense/grouped Identity/Gaussian responses, plus Gamma/DoubleGamma/BachSCR
    with explicit response quadrature, with white noise and
    explicit training anchors. ``reported_parameter_draws()`` applies a separate
    positive-diagonal QR to each draw; raw draws and diagnostics are unchanged.
    Latent mixtures use the same draw-specific coordinates and full conditional
    covariance in each projection. These are reporting conventions only.
    Multifactor spectral mode and run baselines remain unsupported. Full
    dense/grouped posterior supports explicit joint donor/participant updates.
    ``sample_latent`` returns joint conditional trajectory draws.

    ``priors`` describes loadings, offsets, noise variance and free response
    parameters. A numeric ``length_scale`` fixes the latent GP correlation
    timescale in timestamp units. Pass a ``Prior`` with positive finite bounds
    to learn one shared timescale, keeping latent variance fixed at one. This
    supports dense/grouped MAP and posterior with Identity/Gaussian responses
    or quadrature Gamma/DoubleGamma/BachSCR, independent noise and no run
    baselines. It is not a response width.
    Optional ``anchor=(subject, modality, feature)`` sets a positive loading
    for one factor; otherwise the first sorted eligible training feature is
    used. The anchor may have any supported response family.
    Optional ``reference_modality`` selects a fixed lag (shape may be learned).
    Otherwise an already fixed lag is used if available; with all lags free,
    proper supplied priors define the common offset and relative lag summaries
    subtract the across-modality mean. Latent queries and lag priors/bounds
    remain in model-clock coordinates. No convention establishes physiological
    timing or identifies disconnected data components.
    The model consumes supplied units without standardization. MAP and NUTS
    share a proper physical parameter density and marginalize the latent GP.
    Sampling diagnostics do not establish calibration or complete mode coverage.

    ``linear_algebra="grouped"`` opts into an exact reduction over repeated
    modality/time functionals. ``"dense"`` remains the default reference.
    Both paths retain the original scalar-observation capacity guard.

    ``linear_algebra="state_space"`` uses exact Matérn-3/2 Kalman inference
    with stationary initialization and native observation times. It supports
    MAP with Identity, Gaussian, or integer-shape Gamma/DoubleGamma responses
    and independent observation noise. Widths/scales, undershoot ratios, and
    lags may be learned; Gamma shapes remain fixed. Gaussian responses use
    a qualified rational approximation; Gamma tails are restored. A temporal
    covariance error bound is checked over the full declared parameter domain
    against covariance_tolerance. ``state_space_gaussian="auto"`` tries compact
    rational banks before the larger Laguerre representation; either method
    can be requested explicitly. BachSCR supports fixed shapes and optional
    lag learning, with explicit error from restoring its 90-second cutoff.
    The canonical slow tail requires a substantially larger tolerance than
    the default. SampledKernel remains unsupported.
    Prediction uses an RTS smoother; no dense observation covariance is built.
    The explicit max_observations guard still applies and can be raised.

    ``linear_algebra="spectral", spectral=SpectralConfig(rank, padding)``
    explicitly approximates the GP with a padded sine basis. Its finite rank
    and boundary errors require separate numerical and posterior validation.
    Boundaries can introduce spurious common-lag likelihood dependence; this
    does not establish absolute physiological timing.
    Gaussian coefficients are marginalized exactly within that approximation.

    ``response_quadrature_order`` explicitly enables finite response-functional
    quadrature for Gamma, DoubleGamma and BachSCR (and Gaussian/Identity).
    Orders 8..1024 are per integration panel. This path supports dense/grouped
    MAP and posterior sampling, including continuous shape estimation. BachSCR
    requires fixing t0 when learning lag. The latent process
    remains continuous; response integrals and L2
    norms are approximate. ``covariance_tolerance`` does not certify this path:
    compare higher orders and independent integrals for the declared bounds.
    Approximation metadata is retained in configuration, queries and archives.

    ``run_baseline_sd={modality: sd}`` optionally adds independent zero-mean
    Gaussian offsets per participant/run/feature, in supplied observation units.
    Their fixed prior SDs are explicit; offsets are analytically marginalized.
    Dense/grouped calculations support them; spectral does not yet. Prediction
    includes baseline uncertainty even with ``include_noise=False``; the shared
    latent excludes these offsets. None/zero preserves the original model.
    """

    def __init__(
        self,
        *,
        priors,
        anchor=None,
        features=1,
        factor_anchors=None,
        reference_modality=None,
        responses=None,
        length_scale=3.0,
        inference="posterior",
        search=None,
        sampler=None,
        sample_blocks=None,
        random_state=0,
        max_observations=800,
        covariance_tolerance=1e-7,
        linear_algebra="dense",
        spectral=None,
        run_baseline_sd=None,
        noise_timescales=None,
        response_quadrature_order=None,
        state_space_gaussian="auto",
    ):
        self.priors = priors
        self.anchor = anchor
        self.features = features
        self.factor_anchors = factor_anchors
        self.reference_modality = reference_modality
        self.responses = responses
        self.length_scale = length_scale
        self.inference = inference
        self.search = search
        self.sampler = sampler
        self.sample_blocks = sample_blocks
        self.random_state = random_state
        self.max_observations = max_observations
        self.covariance_tolerance = covariance_tolerance
        self.linear_algebra = linear_algebra
        self.spectral = spectral
        self.run_baseline_sd = run_baseline_sd
        self.noise_timescales = noise_timescales
        self.response_quadrature_order = response_quadrature_order
        self.state_space_gaussian = state_space_gaussian

    def _config(self):
        from .spectral import validate_config

        _, length_scale_prior = length_scale_settings(self.length_scale)
        validate_length_scale_scope(
            length_scale_prior,
            linear_algebra=self.linear_algebra,
            response_quadrature_order=self.response_quadrature_order,
            run_baseline_sd=self.run_baseline_sd,
            noise_timescales=self.noise_timescales,
            responses=self.responses,
        )
        if self.state_space_gaussian not in ("auto", "laguerre", "rational"):
            raise ValueError("state_space_gaussian must be auto, laguerre, or rational")

        if self.linear_algebra == "state_space" and self.inference != "map":
            raise ValueError("state_space currently requires MAP inference")
        if self.response_quadrature_order is not None:
            from .response_quadrature import validate_order

            validate_order(self.response_quadrature_order)
            if self.linear_algebra not in (
                "dense",
                "grouped",
            ):
                raise ValueError("response quadrature requires dense/grouped inference")
        if self.inference == "posterior":
            from .response_scope import validate_posterior_responses

            validate_posterior_responses(self.responses, self.response_quadrature_order)

        if (
            isinstance(self.features, (bool, np.bool_))
            or not isinstance(self.features, (int, np.integer))
            or self.features < 1
        ):
            raise ValueError("features must be a positive integer")
        if self.features > 1:
            if self.inference == "posterior" and self.factor_anchors is None:
                raise ValueError("multifactor posterior requires explicit factor_anchors")
            if self.linear_algebra not in ("dense", "grouped", "state_space"):
                raise ValueError(
                    "multiple factors support dense/grouped/state_space calculations only"
                )
        elif self.factor_anchors is not None:
            raise ValueError("factor_anchors requires multiple factors; use anchor for one factor")
        validate_config(self.linear_algebra, self.spectral)
        from .temporal_noise import validate as validate_noise

        noise = validate_noise(
            self.noise_timescales,
            self.responses if self.responses is not None else (self.noise_timescales or {}),
        )
        if noise and (
            self.inference != "map" or self.linear_algebra == "spectral" or self.run_baseline_sd
        ):
            raise ValueError("noise_timescales requires dense/grouped MAP without run baselines")
        validate_blocks(self.sample_blocks)
        if self.sample_blocks is not None and self.inference != "posterior":
            raise ValueError("sample_blocks requires posterior inference")
        if self.inference not in ("map", "posterior"):
            raise ValueError("inference must be map or posterior")
        if self.search is not None and not isinstance(self.search, SearchConfig):
            raise ValueError("search must be SearchConfig")
        if self.sampler is not None and not isinstance(self.sampler, SamplerConfig):
            raise ValueError("sampler must be SamplerConfig")
        if self.random_state is not None and (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, (int, np.integer))
            or not 0 <= self.random_state < 2**32
        ):
            raise ValueError("random_state must be None or an integer in [0, 2**32)")
        self.search_config_ = self.search or SearchConfig()
        self.sampler_config_ = self.sampler or SamplerConfig()
        self.seed_ = int(
            self.random_state
            if self.random_state is not None
            else np.random.SeedSequence().generate_state(1)[0]
        )

    def _anchor_choice(self):
        """Keep requested constructor values intact for cloning/persistence."""
        if self.anchor is not None:
            return self.anchor
        if isinstance(self.factor_anchors, (tuple, list)) and self.factor_anchors:
            return self.factor_anchors[0]
        return None

    def _specification(self):
        return copy.deepcopy(
            dict(
                priors=asdict(self.priors),
                anchor=self.anchor,
                **(
                    {"features": self.features, "factor_anchors": self.factor_anchors}
                    if self.features > 1
                    else {}
                ),
                reference_modality=self.reference_modality,
                responses=self.responses,
                length_scale=self.length_scale,
                max_observations=self.max_observations,
                covariance_tolerance=self.covariance_tolerance,
                linear_algebra=self.linear_algebra,
                spectral=self.spectral,
                **(
                    {"response_quadrature_order": self.response_quadrature_order}
                    if self.response_quadrature_order is not None
                    else {}
                ),
                **(
                    {"state_space_gaussian": self.state_space_gaussian}
                    if self.state_space_gaussian != "auto"
                    else {}
                ),
                **({"noise_timescales": self.noise_timescales} if self.noise_timescales else {}),
                **(
                    {"run_baseline_sd": self.run_baseline_sd}
                    if self.run_baseline_sd is not None
                    else {}
                ),
            )
        )

    def fit(
        self,
        data,
        y=None,
        *,
        progress=None,
        search_progress=None,
        warmup_checkpoint=None,
        resume_warmup=False,
    ):
        """Fit the model; search_progress receives restart and iterate MAP checkpoints.

        The existing progress callback retains its phase timing event format.
        For full dense/grouped posterior training, warmup_checkpoint saves the
        complete sampler state after warmup. Set resume_warmup=True to reuse it
        with identical data, settings, source, and runtime. An explicit integer
        random_state is required; existing checkpoint paths are never overwritten.
        """
        for key in list(vars(self)):
            if key.endswith("_"):
                delattr(self, key)
        self._config()
        if type(resume_warmup) is not bool or (resume_warmup and warmup_checkpoint is None):
            raise ValueError("resume_warmup requires a warmup_checkpoint path and a Boolean flag")
        if warmup_checkpoint is not None and self.random_state is None:
            raise ValueError("warmup checkpoints require an explicit integer random_state")
        if warmup_checkpoint is not None and (
            self.inference != "posterior"
            or self.linear_algebra not in ("dense", "grouped")
            or self.run_baseline_sd
            or self.noise_timescales
            or self.sample_blocks is not None
            or type(self) is not BayesianMultimodalSRM
        ):
            raise ValueError(
                "warmup checkpoints require full dense/grouped posterior training "
                "with independent noise and no baselines"
            )
        check_environment_once()
        from .observation_adapter import BayesianObservationAdapter

        length_scale, length_scale_prior = length_scale_settings(self.length_scale)
        adapter = BayesianObservationAdapter(
            features=self.features,
            latent_dt=1.0,
            responses=self.responses,
            length_scale=length_scale,
            standardize=False,
            max_observations=self.max_observations,
            covariance_tolerance=self.covariance_tolerance,
        )
        adapter._prepare(data)
        self.adapter_ = adapter
        self.problem_ = BayesianProblem(
            adapter,
            self.priors,
            anchor=self._anchor_choice(),
            reference_modality=self.reference_modality,
            linear_algebra=self.linear_algebra,
            spectral=self.spectral,
            run_baseline_sd=self.run_baseline_sd,
            noise_timescales=self.noise_timescales,
            response_quadrature_order=self.response_quadrature_order,
            state_space_gaussian=self.state_space_gaussian,
            length_scale_prior=length_scale_prior,
        )
        if self.features > 1:
            from .multifactor import validate_anchors

            keys = validate_anchors(self.problem_, self.factor_anchors)
            self._factor_anchor_keys_ = None if keys is None else tuple(keys)
        self.specification_ = self._specification()
        self.training_data_ = adapter._training_data
        self.prediction_runs_ = dict(adapter.domains_)
        self.targets_ = None
        storage = None
        if warmup_checkpoint is not None:
            from .posterior_updates import adapter_state
            from .warmup_checkpoint import WarmupCheckpoint

            storage = WarmupCheckpoint(
                warmup_checkpoint,
                dict(
                    specification=self.specification_,
                    adapter=adapter_state(adapter),
                    factor_anchors=getattr(self, "_factor_anchor_keys_", None),
                    search=asdict(self.search_config_),
                    sampler=asdict(self.sampler_config_),
                    seed=self.seed_,
                    sampler_seed=(self.seed_ + 1) % 2**32,
                ),
                resume=resume_warmup,
            )
        self._fit_problem(
            progress=progress, search_progress=search_progress, warmup_checkpoint=storage
        )
        # Compact independent provenance; never retain another estimator.
        self.training_fit_ = copy.deepcopy(
            dict(
                inference=self.inference,
                map_parameters=self.map_parameters_,
                map_diagnostics=self.map_diagnostics_,
                restart_diagnostics=self.restart_diagnostics_,
                objective=self.objective_,
                configuration=self.configuration_,
            )
        )
        return self

    def _fit_problem(self, *, progress=None, search_progress=None, warmup_checkpoint=None):
        from .timing import PhaseTimings

        if hasattr(self.problem_, "_posterior_update"):
            from . import _archive
            from .posterior_updates import validate_update_target

            validate_update_target(self.problem_)
            if (
                not _archive.same(self._specification(), self.specification_)
                or not _archive.same(
                    self.specification_,
                    self.problem_._posterior_update["specification"],
                )
                or not _archive.same(self.posterior_update_, self.problem_._posterior_update)
            ):
                raise ValueError("posterior update model differs from its target contract")
        timing = PhaseTimings(progress)
        self.phase_seconds_ = timing.seconds
        with timing.phase("map"):
            checkpoints = {} if search_progress is None else {"progress": search_progress}
            best, self.restart_diagnostics_ = search(
                self.problem_, self.search_config_, self.seed_, **checkpoints
            )
        if best is None:
            raise MAPFitError(self.restart_diagnostics_)
        self.map_diagnostics_ = dict(best)
        self.map_parameters_ = readonly_array(best["parameters"])
        self.parameter_names_ = list(self.problem_.names)
        self.objective_ = best["objective"]
        self.sampling_diagnostics_ = None
        self.sample_stats_ = {}
        if self.features > 1:
            from .multifactor import orientation
            from .posterior_coordinates import metadata

            self.factor_orientation_ = (
                metadata(self._factor_anchor_keys_)
                if self.inference == "posterior"
                else orientation(self.problem_, self.map_parameters_, self._factor_anchor_keys_)
            )
        if self.inference == "posterior":
            xs, stats, loglik, diagnostic = sample(
                self.problem_,
                self.restart_diagnostics_,
                self.sampler_config_,
                (self.seed_ + 1) % 2**32,
                blocks=self.sample_blocks,
                progress=progress,
                factor_anchors=getattr(self, "_factor_anchor_keys_", None),
                **(
                    {"warmup_checkpoint": warmup_checkpoint}
                    if warmup_checkpoint is not None
                    else {}
                ),
            )
            self.parameter_draws_ = readonly_array(xs)
            self.sample_stats_ = {k: readonly_array(v, v.dtype) for k, v in stats.items()}
            self.log_likelihood_draws_ = readonly_array(loglik)
            self.sampling_diagnostics_ = diagnostic
            self.phase_seconds_.update(diagnostic["phase_seconds"])
            if not diagnostic["passes"]:
                warnings.warn(
                    "Bayesian SRM sampling diagnostic thresholds were not met; inspect sampling_diagnostics_",
                    ConvergenceWarning,
                    stacklevel=2,
                )
        else:
            self.parameter_draws_ = readonly_array(self.map_parameters_[None, None, :])
            self.log_likelihood_draws_ = readonly_array(
                [[-float(self.problem_.nll(self.map_parameters_))]]
            )
        if not best["meets_gradient_tolerance"]:
            warnings.warn(
                "Bayesian SRM MAP physical gradient threshold was not met; inspect map_diagnostics_",
                ConvergenceWarning,
                stacklevel=2,
            )
        self._set_configuration()
        return self

    def _set_configuration(self):
        """Describe the current observation problem and retained inference."""
        self.configuration_ = dict(
            features=self.features,
            latent_pooling="shared",
            length_scale=float(self.problem_.gp_length_scale(self.map_parameters_)),
            latent_variance=1.0,
            latent_marginalization="analytic",
            parameter_density="physical",
            inference=self.inference,
            data_units="supplied_without_standardization",
            priors=asdict(self.priors),
            anchor=self.anchor,
            reference_convention=copy.deepcopy(self.problem_.reference_convention),
            seed=self.seed_,
            response_metadata={m: r.metadata for m, r in self.problem_.responses.items()},
            search=asdict(self.search_config_),
            sampler=asdict(self.sampler_config_),
            temporal_covariance_error_bound=self.problem_.covariance_error_bound,
            max_observations=self.max_observations,
            linear_algebra=self.linear_algebra,
            covariance_approximation={
                run: basis.metadata() for run, basis in self.problem_.spectral_bases.items()
            }
            or None,
            linear_system_sizes={
                run: dict(
                    observations=len(system.times),
                    functionals=len(self.problem_.grouped_systems[run].times)
                    if self.linear_algebra in ("grouped", "spectral")
                    else len(system.times),
                    basis_rank=self.spectral.rank if self.spectral is not None else None,
                )
                for run, system in self.problem_.systems.items()
            },
            sample_blocks=None if self.sample_blocks is None else tuple(self.sample_blocks),
            fixed_parameters=[]
            if self.sampling_diagnostics_ is None
            else self.sampling_diagnostics_["fixed_parameters"],
            uncertainty=(
                "conditional_parameter_posterior_mixture"
                if self.sampling_diagnostics_ is not None
                and self.sampling_diagnostics_["fixed_parameters"]
                else "parameter_posterior_mixture"
            )
            if self.inference == "posterior"
            else "conditional_on_MAP",
            calibration_established=False,
        )
        if self.problem_.length_scale_prior is not None:
            self.configuration_["gp_hyperparameters"] = {
                "length_scale": {
                    "parameter": list(LENGTH_SCALE),
                    "sharing": "all_factors_and_runs",
                    "prior": asdict(self.problem_.length_scale_prior),
                    "map": self.configuration_["length_scale"],
                    "configuration_value": "MAP_summary",
                }
            }
        if self.problem_.response_quadrature is not None:
            self.configuration_["response_quadrature"] = (
                self.problem_.response_quadrature.metadata()
            )
        if self.linear_algebra == "state_space":
            self.configuration_["state_space"] = self.problem_.response_state_space.metadata(
                self.features
            )
        if self.features > 1:
            self.configuration_["factor_orientation"] = copy.deepcopy(self.factor_orientation_)
            self.configuration_["reference_convention"]["factor_orientation"] = copy.deepcopy(
                self.factor_orientation_
            )
            for run, detail in self.configuration_["linear_system_sizes"].items():
                detail["factor_functionals"] = (
                    len(self.problem_.grouped_systems[run].times) * self.features
                    if self.linear_algebra == "grouped"
                    else None
                )
        if self.problem_.noise_timescales:
            self.configuration_["temporal_noise"] = dict(
                family="OU",
                timescales=dict(self.problem_.noise_timescales),
                timescale_units="seconds",
                unspecified_modalities="white",
                variance="fitted_per_participant_modality_marginal_variance",
                independence="participant_run_modality_feature",
                include_noise="same_residual_process_at_query_times",
            )
        if self.problem_.run_baseline_sd:
            self.configuration_["run_baseline"] = dict(
                sd=dict(self.problem_.run_baseline_sd),
                units="supplied_observation_units",
                independence="participant_run_modality_feature",
                marginalization="analytic",
                prior_mean=0.0,
                scale_estimated=False,
                included_in_filtered_prediction=True,
                included_in_shared_latent=False,
            )
        return self

    def condition(self, donors, *, targets, donor_layout="runs", mode="joint", progress=None):
        """Condition independent new runs after removing target payloads.

        ``mode="joint"`` (default) refits training plus current donors.
        ``mode="frozen"`` uses the original training MAP without optimization
        or sampling and conditions each new run's common GP on current donors.
        Frozen mode requires a MAP fit; uncertainty excludes parameter
        uncertainty. Returns an independent estimator. Chained calls retain
        the training/calibration reference and replace the current donor batch.
        Target payloads are removed before validation and support selection.
        """
        from .prediction import donor_data

        if mode not in ("joint", "frozen"):
            raise ValueError("mode must be joint or frozen")
        check_is_fitted(self, "parameter_draws_")
        if (
            self.configuration_["inference"] == "posterior"
            and mode == "joint"
            and (
                self.problem_.features > 1
                or hasattr(self, "posterior_update_")
                or (
                    self.linear_algebra in ("dense", "grouped")
                    and self.sample_blocks is None
                    and not self.run_baseline_sd
                    and not self.noise_timescales
                )
            )
        ):
            from .posterior_updates import condition_posterior

            return condition_posterior(
                self,
                donors,
                targets=targets,
                donor_layout=donor_layout,
                progress=progress,
            )
        if self.problem_.features > 1 and mode != "frozen":
            raise ValueError("multiple factors require mode='frozen' donor conditioning")
        if mode == "frozen" and (
            self.inference != "map"
            or self.training_fit_["inference"] != "map"
            or self.configuration_["inference"] != "map"
        ):
            raise ValueError(
                "frozen conditioning requires fitted MAP inference; posterior is unsupported"
            )
        if self._specification() != self.specification_:
            raise ValueError(
                "model specification changed after fitting; call fit again before condition"
            )
        if not isinstance(targets, Mapping) or not targets:
            raise ValueError("targets must map subjects to nonempty modality lists")
        excluded = set()
        for s, modalities in targets.items():
            if isinstance(modalities, str) or not modalities:
                raise ValueError("targets must contain nonempty modality lists")
            for m in modalities:
                if (s, m) not in self.problem_.groups:
                    raise ValueError("target mapping is unavailable")
                excluded.add((s, m))
        selected = donor_data(donors, excluded, donor_layout)
        if not selected:
            raise ValueError("no usable donor observations remain after target exclusion")
        adapter = copy.deepcopy(self.adapter_)
        _, domains = adapter._grids(selected)
        if set(domains) & set(adapter.domains_):
            raise ValueError(
                "donor run IDs overlap training; use distinct IDs for independent runs"
            )
        donor_systems, _ = adapter._systems(selected, domains)
        training_systems, _ = adapter._systems(self.training_data_, adapter.domains_)
        result = clone(self)
        result._config()
        result.adapter_ = adapter
        result.training_data_ = adapter._training_data
        result.training_fit_ = copy.deepcopy(self.training_fit_)
        result.targets_ = {s: tuple(ms) for s, ms in targets.items()}
        result.prediction_runs_ = domains
        result.problem_ = BayesianProblem(
            adapter,
            result.priors,
            anchor=result._anchor_choice(),
            reference_modality=result.reference_modality,
            systems=donor_systems if mode == "frozen" else {**training_systems, **donor_systems},
            linear_algebra=self.linear_algebra,
            spectral=self.spectral,
            run_baseline_sd=self.run_baseline_sd,
            noise_timescales=self.noise_timescales,
            response_quadrature_order=self.response_quadrature_order,
            state_space_gaussian=self.state_space_gaussian,
            conventions_from=self.problem_,
            length_scale_prior=self.problem_.length_scale_prior,
        )
        result.specification_ = result._specification()
        if mode == "joint":
            result._fit_problem(progress=progress)
            result.configuration_.update(
                conditioning_mode="joint",
                parameter_source="training_plus_donors",
                parameter_refits=1,
                diagnostic_scope="training_plus_donors",
            )
        else:
            result._condition_frozen()
        return result

    def _condition_frozen(self):
        """Evaluate only donor likelihood; retain explicitly scoped training gates."""
        training = self.training_fit_
        self.map_parameters_ = readonly_array(training["map_parameters"])
        self.parameter_names_ = list(self.problem_.names)
        self.map_diagnostics_ = copy.deepcopy(training["map_diagnostics"])
        self.map_diagnostics_["scope"] = "original_training"
        self.restart_diagnostics_ = copy.deepcopy(training["restart_diagnostics"])
        for diagnostic in self.restart_diagnostics_:
            diagnostic["scope"] = "original_training"
        self.objective_ = training["objective"]
        self.parameter_draws_ = readonly_array(self.map_parameters_[None, None, :])
        self.sampling_diagnostics_ = None
        self.sample_stats_ = {}
        self.phase_seconds_ = {}
        if self.features > 1:
            self.factor_orientation_ = copy.deepcopy(
                training["configuration"]["factor_orientation"]
            )
        self.log_likelihood_draws_ = readonly_array(
            [[-float(self.problem_.nll(self.map_parameters_))]]
        )
        self._set_configuration()
        self.configuration_.update(
            conditioning_mode="frozen",
            parameter_source="original_training_MAP",
            parameter_refits=0,
            uncertainty="conditional_on_training_MAP",
            diagnostic_scope="original_training",
            objective_scope="original_training",
            log_likelihood_scope="current_donors_at_training_MAP",
        )
        # These describe saved training diagnostics, not an executed refit.
        self.configuration_["search"] = copy.deepcopy(training["configuration"]["search"])
        self.configuration_["seed"] = training["configuration"]["seed"]

    def relative_lag_draws(self):
        """Relative modality lags, in timestamp units (normally seconds).

        Return read-only arrays with the original (chain, draw) axes, including
        fixed modalities. MAP gives shape (1, 1). A fixed reference lag is
        subtracted when available. Without one, subtract
        the across-modality mean separately for each draw, preserving all
        pairwise differences. This centers summaries only: latent query times
        and lag priors remain in model coordinates. It does not establish
        absolute timing identification or calibrated uncertainty.
        """
        check_is_fitted(self, "parameter_draws_")
        reference_lag = self.problem_.reference_convention["reference_lag_seconds"]
        output = {}
        for m, response in self.problem_.responses.items():
            index = self.problem_.indices.get(("filter", m, "lag"))
            lags = (
                np.full(
                    self.parameter_draws_.shape[:2],
                    getattr(response.initial_kernel(), "lag", 0.0),
                )
                if index is None
                else self.parameter_draws_[..., index]
            )
            output[m] = lags
        if reference_lag is None:
            reference_lag = np.mean(np.stack(list(output.values())), axis=0)
        return {m: readonly_array(lags - reference_lag) for m, lags in output.items()}

    def predict(self, *, times, include_noise=False, max_draws=200):
        """Query excluded targets after ``condition``; default 200 mixture draws.

        Output nesting is subject/run/modality. ``max_draws=None`` uses all
        retained draws; otherwise selection is balanced across chains.
        """
        from .prediction import queries, result, selected_draws

        check_is_fitted(self, "parameter_draws_")
        if self.targets_ is None:
            raise ValueError("call condition(donors, targets=...) before target prediction")
        if not isinstance(include_noise, (bool, np.bool_)):
            raise ValueError("include_noise must be Boolean")
        draw_set = selected_draws(self, max_draws)
        output = {}
        for run, query in queries(self, times).items():
            for s, modalities in self.targets_.items():
                for m in modalities:
                    keys = [k for k in self.problem_.keys if k[:2] == (s, m)]
                    output.setdefault(s, {}).setdefault(run, {})[m] = result(
                        self, run, query, draw_set, keys, include_noise=include_noise
                    )
        return output

    def calibrate(self, data, *, search=None, random_state=None, progress=None):
        """Fit one new participant against the frozen training MAP group.

        Calibration run IDs/times must refer to the same training stimulus.
        Returns a separate ParticipantCalibration for independent new-run
        transforms; only its loadings, offsets and noise are optimized.
        Dense/grouped MAP without run baselines is supported. Units remain
        explicit; fit new-participant preprocessing on calibration data only.
        """
        from .calibration import calibrate

        return calibrate(
            self,
            data,
            search_config=search,
            random_state=random_state,
            progress=progress,
        )

    def transform(self, data, *, times, modalities=None, data_layout="runs", readout="gp"):
        """Infer each participant's held-out latent using only their observations.

        Returns ``{subject: {run: GaussianMixtureSeries}}`` in the original
        training MAP coordinates. Parameters are fixed, including filters and
        preprocessing; no optimization or sampling occurs. Known participants
        and new run IDs are required. ``times`` is an explicit query vector or
        run-to-vector mapping. Observations retain their native clocks/masks.

        ``modalities`` optionally selects a nonempty list before reading any
        excluded payload. Every supplied participant/run must retain input.
        Uncertainty is conditional on the training MAP. Independent estimates
        concern the same model latent; they do not introduce private factors.

        ``readout="gp"`` uses the fitted temporal prior. ``"instantaneous"``
        uses its unit-variance marginal prior and same-timestamp observations
        only: selected modalities must have Identity responses, white residual
        noise and no run baselines; spectral fits are unsupported. Only exact
        observed timestamps with at least one available feature are valid.
        Its uncertainty belongs to this instantaneous marginal model, not the
        full GP posterior. Neither readout refits or changes the trained model.
        """
        from .transform import independent_transform

        return independent_transform(
            self,
            data,
            times=times,
            modalities=modalities,
            data_layout=data_layout,
            readout=readout,
        )

    def sample_latent(
        self,
        *,
        times,
        max_draws=200,
        draws_per_parameter=1,
        random_state=None,
        max_joint_size=2048,
    ):
        """Draw joint shared latent paths from an existing parameter posterior.

        Returns one ``TrajectorySamples`` per run, with sample axes parameter
        draw / conditional draw / time / factor. All times and factors in a
        path are sampled jointly in that parameter draw's fitted reporting
        coordinates. No MAP search, MCMC or preprocessing is performed.

        Supports dense/grouped Identity/Gaussian and quadrature Gamma/DoubleGamma/BachSCR
        posteriors with independent observation noise. ``max_draws`` balances retained draws
        across chains, as in ``infer_latent``. ``random_state`` seeds NumPy's
        local generator; the model's sampling seed and state are unchanged.
        Increasing ``draws_per_parameter`` does not increase parameter ESS.

        ``max_joint_size`` bounds query times times factors per run, since
        joint covariance storage and factorization scale quadratically and
        cubically in that size. Invalid times remain NaN, as in marginal
        queries. Separate calls/windows do not preserve cross-call correlation.
        """
        from .trajectories import sample_latent

        check_is_fitted(self, "parameter_draws_")
        return sample_latent(
            self,
            times=times,
            max_draws=max_draws,
            draws_per_parameter=draws_per_parameter,
            random_state=random_state,
            max_joint_size=max_joint_size,
        )

    def infer_latent(self, *, times, max_draws=200):
        """Common latent marginal mixtures in fitted reporting coordinates.

        Multifactor posterior uses each draw's training-anchor QR rotation;
        MAP uses its fixed training rotation. Cross-factor covariance contributes
        to each projected variance; joint trajectory covariance is not returned.
        """
        from .prediction import queries, result, selected_draws

        check_is_fitted(self, "parameter_draws_")
        draw_set, indices = selected_draws(self, max_draws, return_indices=True)
        return {
            r: result(
                self,
                r,
                q,
                draw_set,
                [None] * self.problem_.features,
                include_noise=False,
                draw_indices=indices,
            )
            for r, q in queries(self, times).items()
        }

    def reported_parameter_draws(self):
        """Read-only chain/draw/parameter copy with only loadings rotated.

        Posterior rotations use the immutable fitted anchor keys. MAP retains
        its established training rotation; one factor retains raw coordinates.
        Fixed-parameter and sampler metadata always refer to raw coordinates.
        """
        from .posterior_coordinates import rotations
        from .prediction import selected_draws

        check_is_fitted(self, "parameter_draws_")
        draws, indices = selected_draws(self, None, return_indices=True)
        output = draws.copy()
        k = self.problem_.features
        if k > 1:
            Q = (
                rotations(self.problem_, draws, self._factor_anchor_keys_, indices)
                if self.configuration_["inference"] == "posterior"
                else np.asarray(self.configuration_["factor_orientation"]["rotation"])
            )
            count = len(self.problem_.keys) * k
            output[:, :count] = (draws[:, :count].reshape(-1, count // k, k) @ Q).reshape(-1, count)
        return readonly_array(output.reshape(self.parameter_draws_.shape))

    def calibrate_posterior(
        self, data, *, search=None, sampler=None, random_state=None, progress=None
    ):
        """Jointly sample group and one new participant's mapping parameters.

        Calibration uses existing reference stimulus run IDs within their original
        domains. The returned posterior establishes an augmented reference while
        this estimator remains unchanged. Original reporting anchors are retained.
        This operation fits and samples; it does not establish calibration coverage.
        """
        from .posterior_participants import calibrate_posterior

        return calibrate_posterior(
            self,
            data,
            search=search,
            sampler=sampler,
            random_state=random_state,
            progress=progress,
        )

    def condition_participants(self, data, *, sampler=None, random_state=None, progress=None):
        """Fit independent participant updates on distinct new stimulus run IDs.

        Return a participant-to-posterior mapping. Each target combines the saved
        training/calibration reference with only that participant's current batch;
        previous new-run batches are replaced. All parameter uncertainty is
        sampled jointly, and no other participant's new observations enter that
        target. Existing MAP ``transform`` remains a fixed-parameter operation.
        """
        from .posterior_participants import condition_participants

        return condition_participants(
            self,
            data,
            sampler=sampler,
            random_state=random_state,
            progress=progress,
        )
