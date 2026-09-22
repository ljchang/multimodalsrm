"""Compact Gaussian response banks with differentiated physical width and lag."""

import math
from dataclasses import dataclass, replace

import numpy as np

from .. import _bateman
from ..kernels import BatemanSCR, Gamma, Gaussian, Identity
from ._backend import runtime
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

    @classmethod
    def prepare(cls, responses, length_scale, tolerance, *, gaussian_method="auto"):
        initial = [_parameter_box(response) for response in responses.values()]
        has_bateman = any(type(k) is BatemanSCR for k, _ in initial)
        if (
            has_bateman
            and gaussian_method == "laguerre"
            and any(type(k) is Gaussian for k, _ in initial)
        ):
            raise ValueError("mixed SCR/Gaussian requires auto or rational Gaussian representation")
        for order in (20, 24):
            templates = [
                rational_template(order) if type(k) is Gaussian else None for k, _ in initial
            ]
            details = {}
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
            bound = float(np.max(np.outer(errors, masses) + np.outer(masses, errors)))
            if bound <= tolerance:
                break
        else:
            if gaussian_method == "auto" and not has_bateman:
                return ParametricResponseStateSpace.prepare(responses, length_scale, tolerance)
            raise ValueError(
                f"compact response bound {bound:.3g} exceeds covariance_tolerance={tolerance:.3g} over parameter bounds"
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
                else [("bateman", "rise", 2)]
                if type(kernel) is BatemanSCR
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
                if kind == "bateman":
                    key = (
                        (kind, name)
                        if set(response.free_parameters) - {"lag"}
                        else (kind, size, kernel.rise, kernel.decay)
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
            if kind == "bateman":
                for p in ("rise", "decay"):
                    low, high = boxes[i][p]
                    decay.append(1 / high)
                    norms2.append(1 / low**2)
                    drives2.append(2 / low)
            elif kind == "gamma":
                decay.append(1 / hi)
                norms2.extend([1 / lo**2] * size)
                drives2.extend([2 / lo] * size)
            else:
                for pair in _paired_indices(template.poles):
                    pole = template.poles[pair[0]]
                    a, radius = -pole.real / lo, abs(pole) / lo
                    decay.append(-pole.real / hi)
                    norms2.append(a * a if len(pair) == 1 else 2 * radius**2 + 4 * a * a)
                    drives2.append((2 if len(pair) == 1 else 4) * a)
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
            if kind == "bateman":
                rates = [1 / values[i]["rise"], 1 / values[i]["decay"]]
                for j, a in enumerate(rates):
                    F = F.at[start, start].set(-a)
                    if j == 0:
                        F = F.at[start, :2].set(a * latent)
                    else:
                        F = F.at[start, start - 1].set(a)
                    blocks.append(real_pole_block(a))
                    start += 1
                readout = (
                    jnp.zeros(size)
                    .at[-1]
                    .set(1 / _bateman.energy(values[i]["rise"], values[i]["decay"], jnp))
                )
            elif kind == "gamma":
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
                    pole = template.poles[pair[0]] / scale
                    residue = template.residues[pair[0]] / jnp.sqrt(scale)
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
            elif template or type(kernel) is BatemanSCR:
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
        if any(type(k) is BatemanSCR for _, k, _ in self.specs):
            result.update(
                response_support=result["response_support"] + "_BatemanSCR",
                scr_realization="exact_Bateman_cascade_with_explicit_90_second_tail_bound",
                scr_supported_learning="rise_decay_lag",
            )
        return result
