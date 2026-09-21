"""Fixed-topology rational response banks with differentiable physical parameters."""

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import gammainc, gammaln
from scipy.stats import gamma as gamma_distribution

from ..kernels import DoubleGamma, Gamma, Gaussian, Identity
from ._backend import runtime
from .state_space_gaussian import GAUSSIAN_ORDERS, gaussian_template
from .state_space_responses import ResponseStateSpace


def _parts(kernel, p):
    if type(kernel) is Gamma:
        return [(int(kernel.shape), p["scale"], 1.0)]
    return [
        (int(kernel.peak_shape), p["peak_scale"], 1.0),
        (int(kernel.undershoot_shape), p["undershoot_scale"], -p["undershoot_ratio"]),
    ]


def finite_gamma_energy(kernel, parameters):
    """Exact incomplete-gamma product integrals on the existing finite support."""
    _, jnp, jsp, _ = runtime()
    components = _parts(kernel, parameters)
    end = jnp.max(jnp.stack([gamma_distribution.ppf(1 - 1e-8, n) * s for n, s, _ in components]))
    energy = 0.0
    for n, s, a in components:
        for m, t, b in components:
            power, rate = n + m - 1, 1 / s + 1 / t
            log_mass = (
                gammaln(power)
                - gammaln(n)
                - gammaln(m)
                - n * jnp.log(s)
                - m * jnp.log(t)
                - power * jnp.log(rate)
            )
            energy += a * b * jnp.exp(log_mass) * jsp.special.gammainc(power, rate * end)
    return jnp.sqrt(energy)


def _parameter_box(response):
    kernel = response.initial_kernel()
    allowed = {
        Identity: set(),
        Gaussian: {"width", "lag"},
        Gamma: {"scale", "lag"},
        DoubleGamma: {"peak_scale", "undershoot_scale", "undershoot_ratio", "lag"},
    }
    if type(kernel) not in allowed or set(response.free_parameters) - allowed[type(kernel)]:
        raise ValueError(
            "state_space requires fixed integer gamma shapes; supports learned scales, ratio and lag"
        )
    box = {
        p: response.parameter_bounds()[p] if p in response.free_parameters else (v, v)
        for p, v in kernel.parameters.items()
    }
    for p, (lo, hi) in box.items():
        if not np.isfinite([lo, hi]).all() or (p in kernel.positive_parameters and lo <= 0):
            raise ValueError(
                "state_space response bounds must be finite and positive for scales/ratios"
            )
    for p in kernel.parameters:
        if "shape" in p and (
            not float(kernel.parameters[p]).is_integer() or not 1 <= kernel.parameters[p] <= 32
        ):
            raise ValueError("state_space gamma shapes must be fixed integers from 1 to 32")
    if type(kernel) is DoubleGamma:
        if box["undershoot_ratio"][1] >= 1:
            raise ValueError("state_space undershoot ratio must remain strictly below one")
        if (
            kernel.undershoot_shape * box["undershoot_scale"][0]
            <= kernel.peak_shape * box["peak_scale"][1]
        ):
            raise ValueError("DoubleGamma bounds must preserve peak/undershoot ordering throughout")
    return kernel, box


def _gamma_bounds(kernel, box):
    """Conservative L1 mass/tail bounds using a lower finite L2 norm over the box."""
    if type(kernel) is Identity:
        return 1.0, 0.0

    def component(shape, scale):
        q = gamma_distribution.ppf(1 - 1e-8, shape)
        norm2 = (
            np.exp(gammaln(2 * shape - 1) - 2 * gammaln(shape) - (2 * shape - 1) * np.log(2))
            / scale
        )
        return (
            norm2 * gammainc(2 * shape - 1, 2 * q),
            q,
            gamma_distribution.sf(q, shape),
        )

    if type(kernel) is Gamma:
        norm2, _, tail = component(kernel.shape, box["scale"][1])
        return 1 / np.sqrt(norm2), tail / np.sqrt(norm2)
    n, m = kernel.peak_shape, kernel.undershoot_shape
    slo, shi = box["peak_scale"]
    tlo, thi = box["undershoot_scale"]
    rlo, rhi = box["undershoot_ratio"]
    peak, qp, ep = component(n, shi)
    under, qu, eu = component(m, thi)

    def overlap(s, t):
        return np.exp(
            gammaln(n + m - 1)
            - gammaln(n)
            - gammaln(m)
            + (m - 1) * np.log(s)
            + (n - 1) * np.log(t)
            - (n + m - 1) * np.log(s + t)
        )

    # A homogeneous function of degree -1 has no interior maximum. Check
    # stationary points along the two lower-scale rectangle edges.
    cross = max(
        overlap(slo, np.clip((n - 1) * slo / m, tlo, thi)),
        overlap(np.clip((m - 1) * tlo / n, slo, shi), tlo),
    )
    end = max(qp * shi, qu * thi)
    mass_lower = 1 - ep - rhi
    norm2 = max(
        peak + rlo * rlo * under - 2 * rhi * cross,
        mass_lower**2 / end if mass_lower > 0 else 0,
    )
    if norm2 <= 0:
        raise ValueError(
            "state_space could not bound DoubleGamma normalization over parameter bounds"
        )
    return (1 + rhi) / np.sqrt(norm2), (ep + rhi * eu) / np.sqrt(norm2)


def parameter_transitions(model, generator, driving, deltas, features):
    """Stable, differentiable transitions for a changing rational realization.

    On a step with ||F dt||_1 <= 1/2, integrate the driving outer product with
    a positive 12-node rule and an order-18 exponential-vector series. Their
    truncation is below float64 rounding here; no I-A A.T cancellation occurs.
    A statically bounded doubling scan permits reverse-mode differentiation.
    """
    jax, jnp, jsp, _ = runtime()
    d = model.dimension
    cutoff = (400 + 2 * d) / model.minimum_rate
    norm_bound = getattr(model, "generator_norm_bound", None)
    if norm_bound is None:
        norm_bound = (2 * d - 1) * model.maximum_rate
    maximum_doublings = int(np.ceil(np.log2(max(2 * norm_bound * cutoff, 1))))
    nodes, weights = np.polynomial.legendre.leggauss(12)
    nodes, weights = (nodes + 1) / 2, weights / 2
    powers = jnp.asarray(nodes[None, :] ** np.arange(19)[:, None])

    def one(delta):
        interval = jnp.minimum(delta, cutoff)
        size = jnp.linalg.norm(generator, ord=1) * interval
        count = jnp.ceil(jnp.log2(jnp.maximum(2 * size, 1))).astype(jnp.int32)
        step = interval / 2.0**count
        X = generator * step
        A = jsp.linalg.expm(X, max_squarings=0)

        def term(previous, k):
            value = X @ previous / k
            return value, value

        _, vectors = jax.lax.scan(term, driving, jnp.arange(1, 19))
        coefficients = jnp.concatenate((driving[None, :], vectors))
        sampled = coefficients.T @ powers * jnp.sqrt(jnp.asarray(weights))
        Q = step * (sampled @ sampled.T)

        def double(i, pair):
            def advance(pair):
                A, Q = pair
                return A @ A, Q + A @ Q @ A.T

            return jax.lax.cond(i < count, advance, lambda pair: pair, pair)

        A, Q = jax.lax.fori_loop(0, maximum_doublings, double, (A, Q))
        return jnp.where(delta > cutoff, jnp.zeros_like(A), A), jnp.where(
            delta > cutoff, jnp.eye(d), (Q + Q.T) * 0.5
        )

    A, Q = jax.vmap(one)(deltas)
    return jnp.kron(jnp.eye(features), A), jnp.kron(jnp.eye(features), Q)


def parameter_event_system(model, state, times, features):
    from .state_space import StateSpaceSystem

    _, jnp, _, _ = runtime()
    order = jnp.argsort(times, stable=True)
    ordered = times[order]
    A, Q = parameter_transitions(
        model, state[0], state[1], jnp.diff(ordered, prepend=ordered[0]), features
    )
    return StateSpaceSystem(order, A, Q, ordered)


def parameter_covariance(problem, x, times_a, modalities_a, times_b, modalities_b):
    """Dense diagnostic only, using current outputs, poles, and internal delays."""
    jax, jnp, jsp, _ = runtime()
    model = problem.response_state_space
    F, _, outputs, lags = model.realize(x, problem.indices)
    ma, mb = np.asarray(modalities_a), np.asarray(modalities_b)
    ta, tb = jnp.asarray(times_a) - lags[ma], jnp.asarray(times_b) - lags[mb]
    delta = (ta[:, None] - tb[None, :]).ravel()
    ca, cb = outputs[np.repeat(ma, len(mb))], outputs[np.tile(mb, len(ma))]
    cutoff = (400 + 2 * model.dimension) / model.minimum_rate

    def one(args):
        dt, a, b = args
        A = jsp.linalg.expm(F * jnp.minimum(jnp.abs(dt), cutoff), max_squarings=64)
        return jnp.where(jnp.abs(dt) > cutoff, 0.0, jnp.where(dt >= 0, a @ A @ b, b @ A @ a))

    return jax.lax.map(one, (delta, ca, cb), batch_size=16).reshape(len(ma), len(mb))


@dataclass(frozen=True)
class ParametricResponseStateSpace(ResponseStateSpace):
    specs: tuple = ()
    banks: tuple = ()
    variable_parameters: tuple = ()
    parameter_boxes: tuple = ()
    maximum_rate: float = 1.0
    gaussian_templates: tuple = ()

    @classmethod
    def prepare(cls, responses, length_scale, tolerance):
        specs, banks, bank_keys, boxes, details, variables = [], [], {}, [], {}, []
        initial = [_parameter_box(r) for r in responses.values()]
        has_gaussian = any(type(k) is Gaussian for k, _ in initial)
        # Select once from the entire declared domain, never at optimizer
        # iterates. Thus response dimensions and poles retain a fixed topology.
        for order in GAUSSIAN_ORDERS if has_gaussian else (None,):
            templates = [
                gaussian_template(order) if type(k) is Gaussian else None for k, _ in initial
            ]
            for (name, _), (kernel, box), template in zip(responses.items(), initial, templates):
                if template is None:
                    mass, error = _gamma_bounds(kernel, box)
                else:
                    mass, error = np.sqrt(box["width"][1]) * np.array(
                        [template.mass_bound, template.l1_error_bound]
                    )
                details[name] = dict(
                    family=type(kernel).__name__,
                    absolute_mass_bound=float(mass),
                    response_l1_error_bound=float(error),
                )
                if template is not None:
                    details[name]["gaussian_template"] = template.metadata()
            masses = np.array([d["absolute_mass_bound"] for d in details.values()] + [1.0])
            errors = np.array([d["response_l1_error_bound"] for d in details.values()] + [0.0])
            bound = float(np.max(errors[:, None] * masses + masses[:, None] * errors))
            if bound <= tolerance:
                break
        else:
            raise ValueError(
                f"state_space response error bound {bound:.3g} exceeds covariance_tolerance={tolerance:.3g} over parameter bounds; narrow bounds, use dense/grouped or explicitly increase tolerance"
            )
        for i, (name, response) in enumerate(responses.items()):
            kernel, box = initial[i]
            boxes.append(box)
            variables.extend((name, p) for p in response.free_parameters)
            template = templates[i]
            components = (
                []
                if type(kernel) is Identity
                else [("width", template.order)]
                if template
                else [("scale", int(kernel.shape))]
                if type(kernel) is Gamma
                else [
                    ("peak_scale", int(kernel.peak_shape)),
                    ("undershoot_scale", int(kernel.undershoot_shape)),
                ]
            )
            positions = []
            for parameter, order in components:
                scale = kernel.parameters[parameter]
                multiplier = template.rate if template else 1.0
                key = (
                    (name, parameter)
                    if parameter in response.free_parameters
                    else (None, multiplier / scale)
                )
                if key not in bank_keys:
                    bank_keys[key] = len(banks)
                    banks.append((i, parameter, multiplier, order))
                index = bank_keys[key]
                old = banks[index]
                banks[index] = (*old[:3], max(order, old[3]))
                positions.append(index)
            specs.append((name, kernel, tuple(positions)))
        dimension = 2 + sum(b[3] for b in banks)
        if dimension > 128:
            raise ValueError("state_space response representation exceeds 128 states per factor")
        rate = np.sqrt(3) / length_scale
        minimum_rate = min([rate] + [a / boxes[i][p][1] for i, p, a, _ in banks])
        maximum_rate = max([rate] + [a / boxes[i][p][0] for i, p, a, _ in banks])
        model = cls(
            generator=np.zeros((dimension, dimension)),
            driving=np.zeros(dimension),
            outputs=np.zeros((len(specs) + 1, dimension)),
            lags=np.zeros(len(specs) + 1),
            length_scale=length_scale,
            minimum_rate=minimum_rate,
            maximum_rate=maximum_rate,
            response_details=details,
            tail_bound=bound,
            learned_lags=tuple(m for m, p in variables if p == "lag"),
            specs=tuple(specs),
            banks=tuple(banks),
            variable_parameters=tuple(variables),
            parameter_boxes=tuple(boxes),
            gaussian_templates=tuple(templates),
        )
        F, b, C, lags = map(np.asarray, model.realize(np.empty(0), {}))
        if not all(np.isfinite(v).all() for v in (F, b, C, lags)) or abs(C[-1] @ C[-1] - 1) > 1e-8:
            raise ValueError("state_space response realization failed numerical verification")
        return replace(model, generator=F, driving=b, outputs=C, lags=lags)

    def realize(self, x, indices):
        jax, jnp, jsp, _ = runtime()
        values = [
            {
                p: x[indices["filter", name, p]]
                if ("filter", name, p) in indices
                else jnp.asarray(v)
                for p, v in kernel.parameters.items()
            }
            for name, kernel, _ in self.specs
        ]
        d = self.dimension
        rate = np.sqrt(3) / self.length_scale
        latent_row = jnp.array([1.0, -1.0]) / np.sqrt(2)
        F = jnp.zeros((d, d)).at[:2, :2].set(jnp.array([[-rate, 0], [-2 * rate, -rate]]))
        B = jnp.zeros(d).at[:2].set(np.sqrt(2 * rate))
        poles, slices, start = [jnp.array([rate, rate])], [], 2
        for i, p, multiplier, order in self.banks:
            a = multiplier / values[i][p]
            block = slice(start, start + order)
            F = F.at[block, block].set(
                -a * jnp.eye(order) - 2 * a * jnp.tril(jnp.ones((order, order)), -1)
            )
            F = F.at[block, :2].set(jnp.sqrt(2 * a) * jnp.broadcast_to(latent_row, (order, 2)))
            poles.append(jnp.full(order, a))
            slices.append(block)
            start += order
        C = jnp.zeros((len(self.specs) + 1, d)).at[-1, :2].set(latent_row)
        lags = []
        for i, (_, kernel, positions) in enumerate(self.specs):
            p = values[i]
            template = self.gaussian_templates[i]
            lags.append(
                p.get("lag", jnp.asarray(0.0)) - (template.shift * p["width"] if template else 0.0)
            )
            if type(kernel) is Identity:
                C = C.at[i, :2].set(latent_row)
                continue
            if template:
                start = slices[positions[0]].start
                C = C.at[i, start : start + template.order].set(jnp.asarray(template.coefficients))
                continue
            energy = finite_gamma_energy(kernel, p)
            for (n, scale, amplitude), position in zip(_parts(kernel, p), positions):
                coefficients = jnp.array([(-1.0) ** j * math.comb(n - 1, j) for j in range(n)])
                start = slices[position].start
                C = C.at[i, start : start + n].add(
                    amplitude * jnp.sqrt(1 / (2 * scale)) * coefficients / (2.0 ** (n - 1) * energy)
                )
        poles = jnp.concatenate(poles)
        b = jnp.sqrt(2 * poles)
        G = -jnp.diag(poles) - jnp.tril(jnp.outer(b, b), -1)

        def column(j, T):
            rhs = -B * b[j] - T @ G[j, :]
            result = jsp.linalg.solve_triangular(F - poles[j] * jnp.eye(d), rhs, lower=True)
            return T.at[:, j].set(result)

        transform = jax.lax.fori_loop(0, d, column, jnp.zeros((d, d)))
        return G, b, C @ transform, jnp.stack(lags + [jnp.asarray(0.0)])

    def metadata(self, features=1):
        result = super().metadata(features)
        result.update(
            response_support="Identity_integer_Gamma_DoubleGamma_fixed_shapes",
            response_approximation="restore_gamma_tails",
            temporal_covariance_error_bound=self.tail_bound,
            error_bound_scope="entire_declared_response_parameter_box",
            delay_handling="exact_parameter_dependent_shift_of_event_times",
            learned_response_parameters={
                name: [p for m, p in self.variable_parameters if m == name]
                for name, _, _ in self.specs
                if any(m == name for m, _ in self.variable_parameters)
            },
            realization="fixed_topology_differentiable_orthonormal_rational_basis",
        )
        if any(self.gaussian_templates):
            result.pop("temporal_covariance_tail_bound")
            result.update(
                response_support="Identity_Gaussian_integer_Gamma_DoubleGamma_fixed_shapes",
                response_approximation="qualified_Gaussian_Laguerre_and_restored_gamma_tails",
                error_bound_qualification="analytic_inequalities_with_numerically_refined_Gaussian_residual_integrals",
                gaussian_internal_delay="physical_lag_minus_six_widths",
            )
        return result
