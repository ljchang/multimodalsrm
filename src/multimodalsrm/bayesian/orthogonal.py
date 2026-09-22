"""Restricted common O(K) refresh composed after retained NUTS transitions.

The concrete dense model has covariance T * (W W.T); the grouped equivalent
uses K_time x I_factor (see multifactor.factor_statistics). All factors have
one identical temporal law. With iid spherical loading priors, W -> W Q is
therefore a unit-Jacobian symmetry. This module deliberately does not infer
that symmetry for subclasses, conditional targets, or unsupported responses.
Admitted finite-response quadrature preserves the common temporal law.
"""

import numpy as np

from ._backend import runtime
from .blocks import ParameterSubspace
from .priors import Prior
from .problem import BayesianProblem
from .response_scope import validate_posterior_response_target


def validate_orthogonal_target(space):
    """Return ordered loading indices, or reject targets outside this proof.

    All parameter coordinates must be active. Dense/grouped full-training or
    validated joint-update targets support Identity/Gaussian responses and
    explicit Gamma/DoubleGamma/BatemanSCR quadrature with the same temporal law per factor.
    """
    if type(space) is not ParameterSubspace or type(space.problem) is not BayesianProblem:
        raise ValueError("Haar refresh requires concrete ParameterSubspace/BayesianProblem")
    problem = space.problem
    if (
        hasattr(problem, "_posterior_update")
        or getattr(problem, "_full_training_target", False) is not True
    ):
        from .posterior_updates import same_pytree, validate_update_target

        validate_update_target(problem)
        expected_space = ParameterSubspace(problem, space.reference)
        if not same_pytree(space._transform_groups, expected_space._transform_groups):
            raise ValueError("posterior update cached coordinate transforms differ from target")
    if problem.features < 2:
        raise ValueError("Haar refresh requires at least two factors")
    if problem.linear_algebra not in ("dense", "grouped"):
        raise ValueError("Haar refresh supports dense/grouped targets only")
    if tuple(space.active_indices) != tuple(range(len(problem.names))) or space.fixed_indices:
        raise ValueError("Haar refresh requires all coordinates active; no fixed parameters")
    validate_posterior_response_target(problem)
    if problem.run_baseline_sd or problem.noise_timescales:
        raise ValueError("Haar refresh does not support baseline or temporal noise extensions")
    # Compare the actual observations rather than treating class identity alone
    # as evidence of a training target: BayesianProblem also accepts systems=.
    training = problem.adapter._systems(problem.adapter._training_data, problem.adapter.domains_)[0]
    if set(problem.systems) != set(training) or any(
        not np.array_equal(getattr(problem.systems[run], field), getattr(training[run], field))
        for run in training
        for field in ("times", "keys", "values")
    ):
        raise ValueError("Haar refresh requires the full training observation systems")
    expected = [
        ("loading", *key, factor) for key in problem.keys for factor in range(problem.features)
    ]
    nw = len(expected)
    if (
        not nw
        or problem.names[:nw] != expected
        or any(name[0] == "loading" for name in problem.names[nw:])
        or any(problem.indices.get(name) != i for i, name in enumerate(problem.names))
    ):
        raise ValueError("Haar refresh requires complete ordered loading coordinates")
    loading_priors = problem.parameter_priors[:nw]
    if len(loading_priors) != nw or any(
        type(p) is not Prior
        or p.family != "normal"
        or p.loc != 0.0
        or p.lower != -np.inf
        or p.upper != np.inf
        or p != loading_priors[0]
        for p in loading_priors
    ):
        raise ValueError("Haar refresh requires iid zero-mean unbounded normal loading priors")
    _, _, _, dist = runtime()
    identity = dist.transforms.IdentityTransform
    if any(type(t) is not identity for t in problem.transforms[:nw]):
        raise ValueError("Haar refresh requires identity loading transforms")
    # ParameterSubspace caches its transform groups: check what it will actually
    # execute, not just the problem's scalar metadata.
    covered = []
    for positions, indices, transform in space._transform_groups:
        if not np.array_equal(positions, indices):
            raise ValueError("Haar refresh requires all coordinates in their original order")
        selected = np.asarray(indices)[np.asarray(indices) < nw]
        if len(selected):
            if type(transform) is not identity:
                raise ValueError("Haar refresh requires identity loading transform groups")
            covered.extend(selected.tolist())
    if sorted(covered) != list(range(nw)):
        raise ValueError("Haar refresh requires complete loading transform groups")
    return np.arange(nw, dtype=int)


def _haar_matrix(key, dtype, features=2):
    """Draw Haar O(K), including both determinant signs.

    Preserve the original O(2) random stream. For K>2, sign-correcting the
    QR factorization of an iid standard Gaussian matrix gives Haar O(K)
    (Mezzadri, 2007, https://arxiv.org/abs/math-ph/0609050). Do not force a
    positive determinant: that would leave the reflection modes disconnected.
    """
    jax, jnp, _, _ = runtime()
    if features > 2:
        q, r = jnp.linalg.qr(jax.random.normal(key, (features, features), dtype=dtype))
        signs = jnp.where(jnp.diag(r) < 0, -1.0, 1.0)
        return q * signs[None, :]
    angle_key, reflection_key = jax.random.split(key)
    angle = jax.random.uniform(angle_key, (), dtype=dtype, minval=0, maxval=2 * jnp.pi)
    sign = jnp.where(jax.random.bernoulli(reflection_key), 1.0, -1.0)
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[c, -sign * s], [s, sign * c]], dtype=dtype)


def orthogonal_kernel(base_kernel, space):
    """Compose exact NUTS with a post-warmup Haar refresh; keep HMCState.

    NUTS redraws momentum from its fixed adapted metric on every transition
    (r=None). No persistent momentum is transformed: a dense adapted metric
    generally is not invariant under this loading symmetry. The refreshed
    state retains the NUTS transition statistics and kinetic-energy value,
    correcting energy only for roundoff in the recomputed potential.
    """
    from numpyro.infer import NUTS
    from numpyro.infer.hmc_util import euclidean_kinetic_energy
    from numpyro.infer.mcmc import MCMCKernel

    indices = validate_orthogonal_target(space)
    if type(base_kernel) is not NUTS:
        raise ValueError("Haar refresh requires an ordinary NUTS kernel")
    if base_kernel._potential_fn != space.potential or base_kernel.model is not None:
        raise ValueError("NUTS potential must be the supplied space.potential")
    if base_kernel._kinetic_fn is not euclidean_kinetic_energy:
        raise ValueError("Haar refresh requires the standard NUTS kinetic energy")
    jax, jnp, _, _ = runtime()
    nw = len(indices)
    features = space.problem.features
    value_gradient = jax.value_and_grad(space.potential)

    class OrthogonalNUTS(MCMCKernel):
        @property
        def _sample_fn(self):
            # MCMC uses this private NumPyro hook to decide whether init is
            # needed on continuation. Missing it re-vectorizes a batched base.
            return base_kernel._sample_fn

        @property
        def sample_field(self):
            return base_kernel.sample_field

        @property
        def default_fields(self):
            return base_kernel.default_fields

        def postprocess_fn(self, model_args, model_kwargs):
            return base_kernel.postprocess_fn(model_args, model_kwargs)

        def get_diagnostics_str(self, state):
            return base_kernel.get_diagnostics_str(state)

        def init(
            self,
            rng_key,
            num_warmup,
            init_params=None,
            model_args=(),
            model_kwargs=None,
        ):
            self._num_warmup = num_warmup
            state = base_kernel.init(
                rng_key,
                num_warmup,
                init_params,
                model_args,
                {} if model_kwargs is None else model_kwargs,
            )
            if state.r is not None:
                raise ValueError("Haar refresh does not support persistent momentum")
            return state

        def _refresh(self, state):
            next_key, orientation_key = jax.random.split(state.rng_key)
            q = _haar_matrix(orientation_key, state.z.dtype, features)
            z = state.z.at[:nw].set((state.z[:nw].reshape(-1, features) @ q).ravel())
            potential, gradient = value_gradient(z)
            return state._replace(
                z=z,
                z_grad=gradient,
                potential_energy=potential,
                energy=state.energy + (potential - state.potential_energy),
                rng_key=next_key,
            )

        def _after_step(self, state, incoming_iteration):
            return jax.lax.cond(
                incoming_iteration >= self._num_warmup,
                self._refresh,
                lambda s: s,
                state,
            )

        def sample(self, state, model_args, model_kwargs):
            if state.r is not None:
                raise ValueError("Haar refresh does not support persistent momentum")
            result = base_kernel.sample(state, model_args, model_kwargs)
            # NumPyro's iteration counter is not reset when warmup finishes.
            # A batched base already executes NUTS under vmap; only refresh here.
            if jnp.ndim(state.i):
                return jax.vmap(self._after_step)(result, state.i)
            return self._after_step(result, state.i)

    return OrthogonalNUTS()
