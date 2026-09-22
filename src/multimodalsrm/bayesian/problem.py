"""One shared marginalized probability model, with physical parameter names."""

import copy

import numpy as np

from ..kernels import Gaussian, Identity
from ._backend import accuracy_bound, runtime, temporal
from .gp_hyperparameters import LENGTH_SCALE, validate_length_scale_scope
from .priors import BayesianPriors, Prior


def _group_priors(priors):
    """Batch equal immutable priors without changing coordinate order."""
    groups = {}
    for i, prior in enumerate(priors):
        groups.setdefault(prior, []).append(i)
    return [(np.asarray(indices, dtype=int), prior) for prior, indices in groups.items()]


class BayesianProblem:
    """Prepared native observations with a shared GP timescale.

    Names are tuples: (loading/offset, subject, modality, feature),
    (noise, subject, modality), (filter, modality, parameter), or an optional
    (gp, length_scale) shared by all factors and runs.
    All vectors, including posterior draws, use physical parameter units.
    """

    # Historical pickled fixed-GP problems did not carry this optional field.
    length_scale_prior = None

    def __init__(
        self,
        adapter,
        priors,
        *,
        anchor=None,
        reference_modality=None,
        systems=None,
        linear_algebra="dense",
        spectral=None,
        run_baseline_sd=None,
        noise_timescales=None,
        response_quadrature_order=None,
        state_space_gaussian="auto",
        conventions_from=None,
        length_scale_prior=None,
    ):
        from .spectral import validate_config

        self._full_training_target = systems is None and conventions_from is None
        validate_config(linear_algebra, spectral)
        self.linear_algebra = linear_algebra
        self.spectral = spectral
        if state_space_gaussian not in ("auto", "laguerre", "rational"):
            raise ValueError("state_space_gaussian must be auto, laguerre, or rational")
        self.state_space_gaussian = state_space_gaussian
        self.features = adapter.features
        if self.features > 1 and linear_algebra == "spectral":
            raise ValueError("multiple factors support dense/grouped calculations only")
        if not isinstance(priors, BayesianPriors):
            raise ValueError("priors must be BayesianPriors")
        self.adapter, self.priors = adapter, priors
        self.length_scale = adapter.length_scale
        validate_length_scale_scope(
            length_scale_prior,
            linear_algebra=linear_algebra,
            response_quadrature_order=response_quadrature_order,
            run_baseline_sd=run_baseline_sd,
            noise_timescales=noise_timescales,
            responses=adapter.responses_,
        )
        self.length_scale_prior = length_scale_prior
        self.responses = adapter.responses_
        from .temporal_noise import validate as validate_noise

        self.noise_timescales = validate_noise(noise_timescales, self.responses)
        if self.noise_timescales and (linear_algebra == "spectral" or run_baseline_sd):
            raise ValueError("noise_timescales supports dense/grouped without run baselines")
        from .baselines import validate

        self.run_baseline_sd = validate(run_baseline_sd, self.responses)
        if self.features > 1 and self.run_baseline_sd:
            raise ValueError("run baselines currently support one factor only")
        if self.run_baseline_sd and linear_algebra == "spectral":
            raise ValueError("run baselines currently support dense/grouped calculations only")
        if (
            linear_algebra != "state_space"
            and response_quadrature_order is None
            and any(
                type(r.initial_kernel()) not in (Identity, Gaussian)
                for r in self.responses.values()
            )
        ):
            raise ValueError("structured responses require explicit response_quadrature_order")
        if response_quadrature_order is not None:
            from .response_quadrature import validate_order

            validate_order(response_quadrature_order)
            if linear_algebra == "spectral":
                raise ValueError("response quadrature supports dense/grouped only")
        self.response_quadrature_order = response_quadrature_order
        self.learned_response_lags = False
        self.parameterized_responses = False
        self.dynamic_state_space = False
        if linear_algebra == "state_space":
            from .state_space import validate

            self.response_state_space = validate(self)
            self.learned_response_lags = bool(self.response_state_space.learned_lags)
            self.parameterized_responses = bool(
                getattr(self.response_state_space, "variable_parameters", ())
            )
            self.dynamic_state_space = self.learned_response_lags or self.parameterized_responses
            if self.learned_response_lags and not self.parameterized_responses:
                from .state_space_delays import transition_function

                self.delay_transition = transition_function(self.response_state_space)
        if any(
            r.pooling != "shared" or r.prior is not None or r.lag_prior is not None
            for r in self.responses.values()
        ):
            raise ValueError(
                "use shared responses and explicit Bayesian priors, without legacy response penalties"
            )
        self.groups = [(s, m) for s, mods in adapter.preprocessing_.items() for m in mods]
        self.keys = [
            (s, m, f)
            for s, m in self.groups
            for f in range(len(adapter.preprocessing_[s][m]["mean"]))
        ]
        from .reference import reference_convention, resolve_anchor

        training_systems = adapter._systems(adapter._training_data, adapter.domains_)[0]
        if conventions_from is None:
            self.reference_convention = reference_convention(
                adapter,
                self.keys,
                anchor,
                reference_modality,
                training_systems,
                linear_algebra=linear_algebra,
            )
        else:
            # Calibration augments the participant set, but cannot reorient the
            # frozen group or select a different clock from the new observations.
            anchor = conventions_from.anchor
            self.reference_convention = copy.deepcopy(conventions_from.reference_convention)
        anchor = resolve_anchor(self.keys, anchor, training_systems)
        if length_scale_prior is not None:
            self.reference_convention.pop("fixed_length_scale_seconds", None)
            self.reference_convention["length_scale_parameter"] = list(LENGTH_SCALE)
        if self.features > 1:
            self.reference_convention.update(
                positive_loading_anchor=None,
                loading_coordinates="unconstrained_isotropic_factor_loadings",
                factor_orientation="training_only_QR_reporting_convention",
            )
        self.anchor = anchor
        expected = {(m, p) for m, r in self.responses.items() for p in r.free_parameters}
        supplied = {(m, p) for m, ps in priors.filters.items() for p in ps}
        if supplied != expected:
            raise ValueError("filter priors must cover exactly every free response parameter")
        self.covariance_error_bound = (
            self.response_state_space.tail_bound
            if linear_algebra == "state_space"
            else (
                None
                if response_quadrature_order is not None
                else accuracy_bound(
                    self.responses,
                    self.length_scale if length_scale_prior is None else length_scale_prior.lower,
                    adapter.covariance_tolerance,
                )
            )
        )
        self.systems = training_systems if systems is None else dict(systems)
        self.modalities = list(self.responses)
        self.names, self.parameter_priors = [], []

        def add(name, prior):
            self.names.append(name)
            self.parameter_priors.append(prior)

        prior = Prior.normal(0.0, priors.loading_sd)
        for key in self.keys:
            if self.features == 1:
                add(
                    ("loading", *key),
                    prior.bounded(0.0, np.inf) if key == anchor else prior,
                )
            else:
                for factor in range(self.features):
                    add(("loading", *key, factor), prior)
        offset_prior = Prior.normal(0.0, priors.offset_sd)
        for key in self.keys:
            add(("offset", *key), offset_prior)
        for group in self.groups:
            add(("noise", *group), priors.noise)
        for m, response in self.responses.items():
            for parameter in response.free_parameters:
                add(
                    ("filter", m, parameter),
                    priors.filters[m][parameter].bounded(*response.parameter_bounds()[parameter]),
                )
        if length_scale_prior is not None:
            add(LENGTH_SCALE, length_scale_prior)
        self.indices = {name: i for i, name in enumerate(self.names)}
        self.response_quadrature = None
        if response_quadrature_order is not None:
            from .response_quadrature import ResponseQuadrature

            self.response_quadrature = ResponseQuadrature(
                self.responses,
                self.indices,
                self.length_scale,
                response_quadrature_order,
                minimum_length_scale=(
                    None if length_scale_prior is None else length_scale_prior.lower
                ),
            )
        self._prepare_parameter_priors()
        jax, _, _, _ = runtime()
        self._packed = {}
        self.grouped_systems = {}
        self.spectral_bases = {}
        self.baseline_designs = {}
        self.noise_systems = {}
        self.state_space_systems = {}
        self.grouped_state_space = (
            linear_algebra == "state_space"
            and not self.dynamic_state_space
            and (priors.noise.family == "lognormal" or priors.noise.lower > 0)
        )
        key_indices = {key: i for i, key in enumerate(self.keys)}
        group_indices = {key: i for i, key in enumerate(self.groups)}
        modality_indices = {key: i for i, key in enumerate(self.modalities)}
        for run, system in self.systems.items():
            if len(system.times) > adapter.max_observations:
                raise ValueError("run exceeds max_observations")
            self._packed[run] = (
                np.array([key_indices[k] for k in system.keys]),
                np.array([group_indices[k[:2]] for k in system.keys]),
                np.array([modality_indices[k[1]] for k in system.keys]),
            )
            if linear_algebra == "state_space" and not self.dynamic_state_space:
                from .state_space import StateSpaceSystem

                times, modalities = system.times, self._packed[run][2]
                if self.grouped_state_space:
                    from .grouped import GroupedSystem

                    nodes = GroupedSystem.prepare(times, modalities, covariance_lookup=False)
                    self.grouped_systems[run] = nodes
                    times, modalities = nodes.times, nodes.modalities
                self.state_space_systems[run] = StateSpaceSystem.prepare(
                    times - self.response_state_space.lags[modalities],
                    self.response_state_space,
                    self.features,
                )
            if self.noise_timescales:
                from .temporal_noise import NoiseSystem

                self.noise_systems[run] = NoiseSystem.prepare(
                    system.times, system.keys, self.keys, self.noise_timescales
                )
            if self.run_baseline_sd:
                from .baselines import design

                self.baseline_designs[run] = design(system, self.run_baseline_sd)
            if linear_algebra in ("grouped", "spectral"):
                from .grouped import GroupedSystem

                self.grouped_systems[run] = GroupedSystem.prepare(
                    system.times,
                    self._packed[run][2],
                    covariance_lookup=linear_algebra == "grouped",
                )
            if linear_algebra == "spectral":
                from .spectral import SpectralBasis

                self.spectral_bases[run] = SpectralBasis.prepare(
                    system.times, self.length_scale, spectral
                )
        self._vg = jax.jit(jax.value_and_grad(self.objective))

    def _prepare_parameter_priors(self):
        """Keep scalar metadata while sharing equal prior/transform operations."""
        _, _, _, dist = runtime()
        self._prior_groups = _group_priors(self.parameter_priors)
        self.initial = np.empty(len(self.parameter_priors))
        self.bounds = [(p.lower, p.upper) for p in self.parameter_priors]
        self.distributions = [None] * len(self.parameter_priors)
        self.transforms = [None] * len(self.parameter_priors)
        for indices, p in self._prior_groups:
            self.initial[indices] = p.ppf(0.5)
            distribution = p.distribution()
            # Transformed distributions can lose a truncated base's bounds.
            # Always construct coordinates from the declared physical support.
            if np.isfinite(p.lower) and np.isfinite(p.upper):
                support = dist.constraints.interval(p.lower, p.upper)
            elif np.isfinite(p.lower):
                support = dist.constraints.greater_than(p.lower)
            elif np.isfinite(p.upper):
                support = dist.constraints.less_than(p.upper)
            else:
                support = dist.constraints.real
            transform = dist.transforms.biject_to(support)
            for i in indices:
                self.distributions[i] = distribution
                self.transforms[i] = transform

    def gp_length_scale(self, x):
        """Physical timescale for this draw, or the original fixed value."""
        index = self.indices.get(LENGTH_SCALE)
        return self.length_scale if index is None else x[index]

    def arrays(self, x):
        _, jnp, _, _ = runtime()
        x = jnp.asarray(x)
        n, g = len(self.keys), len(self.groups)
        widths, lags = [], []
        for m, response in self.responses.items():
            kernel = response.initial_kernel()
            values = []
            for parameter in ("width", "lag"):
                idx = self.indices.get(("filter", m, parameter))
                values.append(x[idx] if idx is not None else getattr(kernel, parameter, 0.0))
            widths.append(values[0])
            lags.append(values[1])
        nw = n * self.features
        return (
            x[:nw] if self.features == 1 else x[:nw].reshape(n, self.features),
            x[nw : nw + n],
            x[nw + n : nw + n + g],
            jnp.array(widths),
            jnp.array(lags),
        )

    def covariance(self, x, run, *, include_noise=True):
        _, jnp, _, _ = runtime()
        w, _, noise, widths, lags = self.arrays(x)
        ki, gi, mi = self._packed[run]
        times = self.systems[run].times
        if self.linear_algebra == "spectral":
            F = self.spectral_bases[run].features(times, widths[mi], lags[mi])
            T = F @ F.T
        else:
            T = self.temporal_covariance(x, times, mi, times, mi)
        C = T * w[ki, None] * w[ki][None, :] if self.features == 1 else T * (w[ki] @ w[ki].T)
        if self.run_baseline_sd:
            U = jnp.asarray(self.baseline_designs[run][0])
            C = C + U @ U.T
        if include_noise:
            C = C + (
                noise[gi, None] * self.noise_systems[run].correlation()
                if self.noise_timescales
                else jnp.diag(noise[gi])
            )
        return C

    def temporal_covariance(self, x, times_a, modalities_a, times_b, modalities_b):
        """Shared temporal block; modality -1 denotes the unfiltered latent."""
        if self.linear_algebra == "state_space":
            if self.parameterized_responses:
                from .state_space_parameters import parameter_covariance

                return parameter_covariance(self, x, times_a, modalities_a, times_b, modalities_b)
            if self.learned_response_lags:
                from .state_space_delays import covariance

                return covariance(self, x, times_a, modalities_a, times_b, modalities_b)
            _, jnp, _, _ = runtime()
            return jnp.asarray(
                self.response_state_space.covariance(times_a, modalities_a, times_b, modalities_b)
            )
        if self.response_quadrature is not None:
            return self.response_quadrature.covariance(
                x,
                times_a,
                modalities_a,
                times_b,
                modalities_b,
                length_scale=self.gp_length_scale(x),
            )
        _, jnp, _, _ = runtime()
        _, _, _, widths, lags = self.arrays(x)
        widths = jnp.concatenate((widths, jnp.zeros(1)))
        lags = jnp.concatenate((lags, jnp.zeros(1)))
        ma, mb = np.asarray(modalities_a), np.asarray(modalities_b)
        return temporal(
            times_a,
            times_b,
            widths[ma],
            widths[mb],
            lags[ma],
            lags[mb],
            self.gp_length_scale(x),
        )

    def nll(self, x):
        if self.linear_algebra == "state_space":
            from .state_space import nll

            return sum(nll(self, x, run) for run in self.systems)
        if self.linear_algebra == "spectral":
            from .spectral import nll

            return sum(nll(self, x, run) for run in self.systems)
        if self.linear_algebra == "grouped":
            if self.run_baseline_sd:
                from .baselines import grouped_factor

                return sum(grouped_factor(self, x, run)[0] for run in self.systems)
            from .grouped import nll

            return sum(nll(self, x, run) for run in self.systems)
        _, jnp, jsp, _ = runtime()
        _, offsets, _, _, _ = self.arrays(x)
        value = 0.0
        for run, system in self.systems.items():
            L = jnp.linalg.cholesky(self.covariance(x, run))
            y = jnp.asarray(system.values) - offsets[self._packed[run][0]]
            whitened = jsp.linalg.solve_triangular(L, y, lower=True)
            value += (
                0.5 * jnp.sum(whitened**2)
                + jnp.log(jnp.diag(L)).sum()
                + 0.5 * len(y) * np.log(2 * np.pi)
            )
        return value

    def log_prior(self, x):
        _, jnp, _, _ = runtime()
        x = jnp.asarray(x)
        valid = jnp.all(jnp.isfinite(x))
        density = 0.0
        for indices, p in self._prior_groups:
            values = x[indices]
            valid = valid & jnp.all((values >= p.lower) & (values <= p.upper))
            if p.family == "lognormal":
                valid = valid & jnp.all(values > 0)
            density += jnp.sum(self.distributions[indices[0]].log_prob(values))
        return jnp.where(valid, density, -jnp.inf)

    def objective(self, x):
        return self.nll(x) - self.log_prior(x)

    def value_gradient(self, x):
        value, gradient = self._vg(x)
        return float(value), np.asarray(gradient)

    def from_unconstrained(self, z):
        _, jnp, _, _ = runtime()
        z = jnp.asarray(z)
        z = z.astype(jnp.result_type(z, float))
        x = jnp.empty_like(z)
        jac = 0.0
        for indices, _ in self._prior_groups:
            transform = self.transforms[indices[0]]
            values = transform(z[indices])
            x = x.at[indices].set(values)
            jac += jnp.sum(transform.log_abs_det_jacobian(z[indices], values))
        return x, jac

    def to_unconstrained(self, x):
        x = np.asarray(x, float)
        if x.shape != self.initial.shape or not np.isfinite(x).all():
            raise ValueError("physical parameters must be a finite vector matching parameter names")
        z = np.empty_like(x)
        for indices, p in self._prior_groups:
            values = x[indices]
            if np.any((values <= p.lower) | (values >= p.upper)):
                raise ValueError("physical parameters must lie strictly inside prior support")
            z[indices] = self.transforms[indices[0]].inv(values)
        if not np.isfinite(z).all():
            raise ValueError("physical parameters must lie strictly inside prior support")
        return z

    def potential(self, z):
        x, jac = self.from_unconstrained(z)
        return self.objective(x) - jac
