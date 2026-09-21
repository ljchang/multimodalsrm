"""Compact Gaussian response banks with differentiated physical width and lag."""

import math
from dataclasses import dataclass, replace

import numpy as np

from ..kernels import BachSCR, Gamma, Gaussian, Identity
from ._backend import runtime
from .state_space_bach import bach_template
from .state_space_gaussian_rational import _paired_indices, rational_template
from .state_space_parameters import (
    ParametricResponseStateSpace,
    _gamma_bounds,
    _parameter_box,
    _parts,
    finite_gamma_energy,
)
from .state_space_rational import (
    conjugate_pole_block,
    rational_realization,
    real_pole_block,
    safe_decay_cutoff,
)


@dataclass(frozen=True)
class CompactResponseStateSpace(ParametricResponseStateSpace):
    compact_banks: tuple = ()
    generator_norm_bound: float = 0.0
    bach_templates: tuple = ()

    @classmethod
    def prepare(cls, responses, length_scale, tolerance, *, gaussian_method="auto"):
        initial = []
        for response in responses.values():
            kernel = response.initial_kernel()
            if type(kernel) is not BachSCR:
                initial.append(_parameter_box(response))
                continue
            if set(response.free_parameters) - {"lag"}:
                raise ValueError(
                    "state_space BachSCR requires fixed t0, sigma, lambda1 and lambda2; only additional lag may be learned"
                )
            box = {
                p: response.parameter_bounds()[p] if p in response.free_parameters else (v, v)
                for p, v in kernel.parameters.items()
            }
            if any(not np.isfinite(bounds).all() for bounds in box.values()):
                raise ValueError("state_space BachSCR lag bounds must be finite")
            initial.append((kernel, box))
        has_bach = any(type(k) is BachSCR for k, _ in initial)
        if (
            has_bach
            and gaussian_method == "laguerre"
            and any(type(k) is Gaussian for k, _ in initial)
        ):
            raise ValueError(
                "mixed BachSCR/Gaussian requires auto or rational Gaussian representation"
            )
        for order in (20, 24):
            templates = [
                rational_template(order)
                if type(k) is Gaussian
                else bach_template(k, order)
                if type(k) is BachSCR
                else None
                for k, _ in initial
            ]
            details = {}
            for (name, _), (kernel, box), template in zip(responses.items(), initial, templates):
                if template is None:
                    mass, error = _gamma_bounds(kernel, box)
                elif type(kernel) is Gaussian:
                    mass, error = np.sqrt(box["width"][1]) * np.array(
                        [template.mass_bound, template.l1_error_bound]
                    )
                else:
                    mass, error = template.mass_bound, template.l1_error_bound
                details[name] = dict(
                    family=type(kernel).__name__,
                    absolute_mass_bound=float(mass),
                    response_l1_error_bound=float(error),
                )
                if template is not None:
                    details[name][
                        "bach_template" if type(kernel) is BachSCR else "gaussian_template"
                    ] = template.metadata()
            masses = np.array([d["absolute_mass_bound"] for d in details.values()] + [1.0])
            errors = np.array([d["response_l1_error_bound"] for d in details.values()] + [0.0])
            bound = float(np.max(np.outer(errors, masses) + np.outer(masses, errors)))
            if bound <= tolerance:
                break
        else:
            if gaussian_method == "auto" and not has_bach:
                return ParametricResponseStateSpace.prepare(responses, length_scale, tolerance)
            raise ValueError(
                f"compact response bound {bound:.3g} exceeds covariance_tolerance={tolerance:.3g} over parameter bounds"
                + (
                    "; BachSCR restores the 90-second tail: use dense/grouped or explicitly request a larger tolerance"
                    if has_bach
                    else ""
                )
            )

        specs, banks, keys, variables = [], [], {}, []
        boxes = [box for _, box in initial]
        for i, ((name, response), (kernel, _), template) in enumerate(
            zip(responses.items(), initial, templates)
        ):
            variables.extend((name, p) for p in response.free_parameters)
            components = (
                []
                if type(kernel) is Identity
                else [("bach", "sigma", template.order + 2)]
                if type(kernel) is BachSCR
                else [("gaussian", "width", template.order)]
                if template
                else [("gamma", "scale", int(kernel.shape))]
                if type(kernel) is Gamma
                else [
                    ("gamma", "peak_scale", int(kernel.peak_shape)),
                    ("gamma", "undershoot_scale", int(kernel.undershoot_shape)),
                ]
            )
            positions = []
            for kind, parameter, size in components:
                key = (
                    (kind, name, parameter)
                    if parameter in response.free_parameters
                    else (
                        kind,
                        kernel.parameters[parameter],
                        template.order if template else None,
                    )
                )
                if kind == "bach":
                    key = (
                        kind,
                        tuple((p, v) for p, v in kernel.parameters.items() if p != "lag"),
                        template.order,
                    )
                if key not in keys:
                    keys[key] = len(banks)
                    banks.append((i, kind, parameter, size, template))
                position = keys[key]
                old = banks[position]
                banks[position] = (*old[:3], max(size, old[3]), old[4])
                positions.append(position)
            specs.append((name, kernel, tuple(positions)))
        dimension = 2 + sum(b[3] for b in banks)
        if dimension > 128:
            raise ValueError("state_space response representation exceeds 128 states per factor")
        rate = np.sqrt(3) / length_scale
        decay, norms2, drives2 = [rate], [6 * rate**2], [4 * rate]
        for i, kind, parameter, size, template in banks:
            lo, hi = boxes[i][parameter]
            if kind == "gamma":
                decay.append(1 / hi)
                norms2.extend([1 / lo**2] * size)
                drives2.extend([2 / lo] * size)
            else:
                if kind == "bach":
                    # Bach templates already contain physical Gaussian poles.
                    lo = hi = 1.0
                for pair in _paired_indices(template.poles):
                    pole = template.poles[pair[0]]
                    a, radius = -pole.real / lo, abs(pole) / lo
                    decay.append(-pole.real / hi)
                    norms2.append(a * a if len(pair) == 1 else 2 * radius**2 + 4 * a * a)
                    drives2.append((2 if len(pair) == 1 else 4) * a)
                if kind == "bach":
                    for parameter in ("lambda1", "lambda2"):
                        a = boxes[i][parameter][0]
                        decay.append(a)
                        norms2.append(a * a)
                        drives2.append(2 * a)
        # Each off-diagonal cascade block is -b_i b_j.T. Bound its Frobenius
        # norm over the entire parameter box, including conjugate-pole blocks.
        M = np.sqrt(sum(norms2) + (sum(drives2) ** 2 - sum(q * q for q in drives2)) / 2)
        cutoff = safe_decay_cutoff(dimension, min(decay), M)
        model = cls(
            generator=np.zeros((dimension, dimension)),
            driving=np.zeros(dimension),
            outputs=np.zeros((len(specs) + 1, dimension)),
            lags=np.zeros(len(specs) + 1),
            length_scale=length_scale,
            minimum_rate=(400 + 2 * dimension) / cutoff,
            maximum_rate=max(np.sqrt(norms2)),
            generator_norm_bound=np.sqrt(dimension) * M,
            response_details=details,
            tail_bound=bound,
            learned_lags=tuple(m for m, p in variables if p == "lag"),
            specs=tuple(specs),
            compact_banks=tuple(banks),
            variable_parameters=tuple(variables),
            parameter_boxes=tuple(boxes),
            gaussian_templates=tuple(
                t if type(k) is Gaussian else None for (k, _), t in zip(initial, templates)
            ),
            bach_templates=tuple(
                t if type(k) is BachSCR else None for (k, _), t in zip(initial, templates)
            ),
        )
        F, b, C, lags = map(np.asarray, model.realize(np.empty(0), {}))
        if (
            not all(np.isfinite(v).all() for v in (F, b, C, lags))
            or np.max(abs(F + F.T + np.outer(b, b))) > 1e-10
            or np.linalg.eigvals(F).real.max() >= 0
        ):
            raise ValueError("compact response realization failed numerical verification")
        return replace(model, generator=F, driving=b, outputs=C, lags=lags)

    def realize(self, x, indices):
        _, jnp, _, _ = runtime()
        values = [
            {
                p: x[indices["filter", name, p]]
                if ("filter", name, p) in indices
                else jnp.asarray(v)
                for p, v in kernel.parameters.items()
            }
            for name, kernel, _ in self.specs
        ]
        d, rate = self.dimension, np.sqrt(3) / self.length_scale
        latent = jnp.array([1.0, -1.0]) / np.sqrt(2)
        A0 = jnp.array([[-rate, 0.0], [-2 * rate, -rate]])
        b0 = jnp.full(2, np.sqrt(2 * rate))
        F = jnp.zeros((d, d)).at[:2, :2].set(A0)
        B = jnp.zeros(d).at[:2].set(b0)
        blocks, readouts, slices, start = [(A0, b0)], [], [], 2
        for i, kind, parameter, size, template in self.compact_banks:
            scale = values[i][parameter]
            bank_start = start
            if kind == "gamma":
                a = 1 / scale
                section = slice(start, start + size)
                F = F.at[section, section].set(
                    -a * jnp.eye(size) - 2 * a * jnp.tril(jnp.ones((size, size)), -1)
                )
                F = F.at[section, :2].set(jnp.sqrt(2 * a) * jnp.broadcast_to(latent, (size, 2)))
                blocks.extend([real_pole_block(a)] * size)
                readout = None
                start += size
            else:
                readout = jnp.zeros(size)
                for pair in _paired_indices(template.poles):
                    pole = template.poles[pair[0]] / (scale if kind == "gaussian" else 1.0)
                    residue = template.residues[pair[0]] / (
                        jnp.sqrt(scale) if kind == "gaussian" else 1.0
                    )
                    if len(pair) == 1:
                        A = jnp.array([[pole.real]])
                        drive, output = jnp.ones(1), jnp.array([residue.real])
                        blocks.append(real_pole_block(-pole.real))
                    else:
                        A = jnp.array([[pole.real, -pole.imag], [pole.imag, pole.real]])
                        drive, output = (
                            jnp.array([1.0, 0.0]),
                            2 * jnp.array([residue.real, -residue.imag]),
                        )
                        blocks.append(conjugate_pole_block(-pole.real, pole.imag))
                    end = start + len(pair)
                    F = F.at[start:end, start:end].set(A)
                    F = F.at[start:end, :2].set(drive[:, None] * latent)
                    readout = readout.at[start - bank_start : end - bank_start].set(output)
                    start = end
                if kind == "bach":
                    gaussian_end = start
                    for parameter in ("lambda1", "lambda2"):
                        a = values[i][parameter]
                        F = F.at[start, start].set(-a)
                        F = F.at[start, bank_start:gaussian_end].set(readout[: template.order])
                        blocks.append(real_pole_block(a))
                        start += 1
                    readout = jnp.zeros(size).at[-2:].set(1 / template.finite_normalizer)
            slices.append(slice(bank_start, start))
            readouts.append(readout)
        C = jnp.zeros((len(self.specs) + 1, d)).at[-1, :2].set(latent)
        lags = []
        for i, (_, kernel, positions) in enumerate(self.specs):
            p, template = values[i], self.gaussian_templates[i]
            lags.append(
                p.get("lag", jnp.asarray(0.0)) - (template.shift * p["width"] if template else 0.0)
            )
            if type(kernel) is Identity:
                C = C.at[i, :2].set(latent)
            elif template:
                C = C.at[i, slices[positions[0]]].set(readouts[positions[0]])
            elif type(kernel) is BachSCR:
                C = C.at[i, slices[positions[0]]].set(readouts[positions[0]])
            else:
                energy = finite_gamma_energy(kernel, p)
                for (n, scale, amplitude), position in zip(_parts(kernel, p), positions):
                    coefficient = jnp.array([(-1.0) ** j * math.comb(n - 1, j) for j in range(n)])
                    begin = slices[position].start
                    C = C.at[i, begin : begin + n].add(
                        amplitude
                        * jnp.sqrt(1 / (2 * scale))
                        * coefficient
                        / (2.0 ** (n - 1) * energy)
                    )
        G, b, output = rational_realization(F, B, C, blocks)
        # These known readouts are constant and exact, including their derivatives.
        exact_latent = jnp.zeros(d).at[:2].set(latent)
        output = output.at[-1].set(exact_latent)
        for i, (_, kernel, _) in enumerate(self.specs):
            if type(kernel) is Identity:
                output = output.at[i].set(exact_latent)
        return G, b, output, jnp.stack(lags + [jnp.asarray(0.0)])

    def metadata(self, features=1):
        result = super().metadata(features)
        result.update(
            response_approximation="qualified_Gaussian_vector_fit_and_restored_gamma_tails",
            realization="fixed_topology_differentiable_real_block_orthonormal_rational_basis",
            transition_decay_bound="uniform_complex_Schur_polynomial_exponential_bound",
            transition_generator_norm_bound=float(self.generator_norm_bound),
        )
        if any(self.bach_templates):
            result.pop("temporal_covariance_tail_bound", None)
            result.update(
                response_support="Identity_Gaussian_integer_Gamma_DoubleGamma_fixed_shape_BachSCR",
                response_approximation="qualified_causal_Gaussian_rational_responses_and_explicit_Gamma_BachSCR_tail_bounds",
                error_bound_qualification="analytic_inequalities_with_numerically_refined_response_residual_integrals",
                bach_supported_learning="additional_lag_only_fixed_shape",
                bach_finite_support_seconds=90.0,
            )
        return result
