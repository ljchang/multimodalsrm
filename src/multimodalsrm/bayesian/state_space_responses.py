"""Rational response realizations, with explicit restored-tail error.

Integer gamma densities are finite combinations of orthonormal Laguerre
impulses. Sharing their banks avoids redundant states. An orthonormal rational
basis gives identity stationary covariance even for nearly coincident poles.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_sylvester
from scipy.stats import gamma as gamma_distribution

from ..kernels import BachSCR, DoubleGamma, Gamma, Gaussian, Identity


def _components(kernel):
    if type(kernel) is Identity:
        return []
    if type(kernel) is Gamma:
        return [(kernel.shape, kernel.scale, 1.0)]
    if type(kernel) is DoubleGamma:
        return [
            (kernel.peak_shape, kernel.peak_scale, 1.0),
            (
                kernel.undershoot_shape,
                kernel.undershoot_scale,
                -kernel.undershoot_ratio,
            ),
        ]
    raise ValueError("state_space supports fixed Identity/Gamma/DoubleGamma responses")


@dataclass(frozen=True)
class ResponseStateSpace:
    generator: np.ndarray
    driving: np.ndarray
    outputs: np.ndarray
    lags: np.ndarray
    length_scale: float
    minimum_rate: float
    response_details: dict
    tail_bound: float
    learned_lags: tuple = ()

    @property
    def dimension(self):
        return len(self.generator)

    @property
    def identity_only(self):
        return self.dimension == 2

    @classmethod
    def prepare(cls, responses, length_scale, tolerance, *, gaussian_method="auto"):
        if gaussian_method not in ("auto", "laguerre", "rational"):
            raise ValueError("state_space_gaussian must be auto, laguerre or rational")
        if any(type(r.initial_kernel()) is BachSCR for r in responses.values()) or (
            gaussian_method != "laguerre"
            and any(type(r.initial_kernel()) is Gaussian for r in responses.values())
        ):
            from .state_space_compact import CompactResponseStateSpace

            return CompactResponseStateSpace.prepare(
                responses, length_scale, tolerance, gaussian_method=gaussian_method
            )
        if any(
            type(r.initial_kernel()) is Gaussian or set(r.free_parameters) - {"lag"}
            for r in responses.values()
        ):
            from .state_space_parameters import ParametricResponseStateSpace

            return ParametricResponseStateSpace.prepare(responses, length_scale, tolerance)
        banks, parts, details = {}, [], {}
        for name, response in responses.items():
            if set(response.free_parameters) - {"lag"}:
                raise ValueError(
                    "state_space requires fixed response parameters except lag; use dense/grouped for learned shapes/scales/ratios"
                )
            kernel = response.initial_kernel()
            components = _components(kernel)
            for shape, scale, _ in components:
                if not float(shape).is_integer() or not 1 <= shape <= 32:
                    raise ValueError(
                        "state_space gamma shapes must be integers from 1 to 32; use dense/grouped"
                    )
                banks[scale] = max(banks.get(scale, 0), int(shape))
            parts.append((kernel, components))
            mass, tail = 1.0, 0.0
            if components:
                end = kernel.support[1] - kernel.lag
                mass = sum(abs(c) for _, _, c in components) / kernel._energy
                tail = (
                    sum(abs(c) * gamma_distribution.sf(end, n, scale=s) for n, s, c in components)
                    / kernel._energy
                )
            details[name] = dict(
                family=type(kernel).__name__,
                absolute_mass_bound=float(mass),
                restored_tail_mass_bound=float(tail),
            )
        masses = np.array([d["absolute_mass_bound"] for d in details.values()] + [1.0])
        tails = np.array([d["restored_tail_mass_bound"] for d in details.values()] + [0.0])
        bound = float(np.max(tails[:, None] * masses + masses[:, None] * tails))
        if bound > tolerance:
            raise ValueError(
                f"state_space restored-tail bound {bound:.3g} exceeds covariance_tolerance={tolerance:.3g}; use dense/grouped or explicitly increase tolerance"
            )
        dimension = 2 + sum(banks.values())
        if dimension > 128:
            raise ValueError(
                "state_space response representation exceeds 128 states per factor; use dense/grouped"
            )
        rate = np.sqrt(3.0) / length_scale
        F = np.zeros((dimension, dimension))
        F[:2, :2] = [[0, rate], [-rate, -2 * rate]]
        B = np.zeros(dimension)
        B[1] = np.sqrt(4 * rate)
        slices, start = {}, 2
        for scale, order in banks.items():
            block = slice(start, start + order)
            a = 1 / scale
            F[block, block] = -a * np.eye(order) - 2 * a * np.tril(np.ones((order, order)), -1)
            F[block, 0] = np.sqrt(2 * a)
            slices[scale] = block
            start += order
        C, lags = np.zeros((len(parts) + 1, dimension)), np.zeros(len(parts) + 1)
        C[-1, 0] = 1.0  # Modality -1 denotes the unfiltered latent.
        for i, (kernel, components) in enumerate(parts):
            if not components:
                C[i, 0] = 1.0
                continue
            lags[i] = kernel.lag
            for shape, scale, amplitude in components:
                n = int(shape)
                block = slices[scale]
                coefficients = np.array([(-1.0) ** j * math.comb(n - 1, j) for j in range(n)])
                C[i, block.start : block.start + n] += (
                    amplitude
                    * np.sqrt(1 / (2 * scale))
                    * coefficients
                    / 2.0 ** (n - 1)
                    / kernel._energy
                )
        if banks:
            # Cascade all-pass sections with these poles. G+G.T=-b b.T,
            # hence its controllability Gramian is exactly identity. Computing
            # the cross Gramian maps the original outputs without inverting an
            # ill-conditioned covariance of almost redundant filter states.
            poles = np.array([rate, rate] + [1 / s for s, n in banks.items() for _ in range(n)])
            b = np.sqrt(2 * poles)
            G = -np.diag(poles) - np.tril(np.outer(b, b), -1)
            transform = solve_sylvester(F, G.T, -np.outer(B, b))
            residual = np.linalg.norm(F @ transform - transform @ G)
            drive_error = np.linalg.norm(B - transform @ b)
            if (
                not np.isfinite(transform).all()
                or residual > 1e-8 * max(1.0, np.linalg.norm(F) * np.linalg.norm(transform))
                or drive_error > 1e-8 * max(1.0, np.linalg.norm(B))
            ):
                raise ValueError(
                    "state_space response realization failed numerical verification; use dense/grouped"
                )
            C, F, B = C @ transform, G, b
            if not np.isfinite(C).all() or abs(C[-1] @ C[-1] - 1.0) > 1e-8:
                raise ValueError("state_space latent normalization failed; use dense/grouped")
        return cls(
            F,
            B,
            C,
            lags,
            length_scale,
            min([rate] + [1 / s for s in banks]),
            details,
            bound,
            tuple(m for m, r in responses.items() if "lag" in r.free_parameters),
        )

    def metadata(self, features=1):
        result = dict(
            family="matern32",
            state_dimension=self.dimension * features,
            states_per_factor=self.dimension,
            initialization="independent_joint_stationary_per_run",
            response_support="fixed_Identity_integer_Gamma_DoubleGamma",
            approximation=not self.identity_only,
            response_approximation="none" if self.identity_only else "restore_gamma_tails",
            normalization="existing_finite_support_continuous_l2",
            temporal_covariance_tail_bound=self.tail_bound,
            bound_excludes="floating_point_error_and_posterior_error",
            responses=self.response_details,
            delay_handling="exact_fixed_shift_of_event_times",
            likelihood="scalar_Kalman_innovations",
            prediction="RTS_smoothing",
        )
        # Archives validate the full descriptor. Keep fixed-response metadata
        # byte-for-byte compatible with the original state-space milestones.
        if self.learned_lags:
            result.update(
                response_support="Identity_integer_Gamma_DoubleGamma_fixed_shapes_scales_ratios",
                delay_handling="exact_learned_shift_of_event_times",
                learned_response_parameters={m: ["lag"] for m in self.learned_lags},
            )
        return result

    def _transition(self, delta):
        d = self.dimension
        if delta == 0:
            return np.eye(d), np.zeros((d, d))
        if delta > (400 + 2 * d) / self.minimum_rate:
            return np.zeros((d, d)), np.eye(d)
        # Small-step Van Loan integration avoids subtracting almost equal
        # stationary covariances and avoids unstable large-interval exponentials.
        size = np.linalg.norm(self.generator, ord=1) * delta
        doublings = max(0, int(np.ceil(np.log2(max(size / 0.5, 1)))))
        step = delta / 2.0**doublings
        block = np.zeros((2 * d, 2 * d))
        block[:d, :d], block[d:, d:] = self.generator, -self.generator.T
        block[:d, d:] = np.outer(self.driving, self.driving)
        E = expm(block * step)
        A = E[:d, :d]
        Q = E[:d, d:] @ A.T
        for _ in range(doublings):
            Q = Q + A @ Q @ A.T
            A = A @ A
        Q = (Q + Q.T) * 0.5
        values, vectors = np.linalg.eigh(Q)
        if not np.isfinite(Q).all() or values[0] < -1e-10:
            raise FloatingPointError("invalid state_space process covariance")
        if values[0] < 0:
            Q = (vectors * np.maximum(values, 0)) @ vectors.T
        return A, Q

    def transitions(self, deltas, features):
        if self.identity_only:
            from .state_space import transitions

            return transitions(deltas, self.length_scale, features)
        unique, inverse = np.unique(deltas, return_inverse=True)
        pairs = [self._transition(float(t)) for t in unique]
        a, q = (np.stack([p[i] for p in pairs])[inverse] for i in (0, 1))
        return np.kron(np.eye(features), a), np.kron(np.eye(features), q)

    def covariance(self, times_a, modalities_a, times_b, modalities_b):
        """Explicit dense diagnostic only; fitting and smoothing never call it."""
        ma, mb = np.asarray(modalities_a), np.asarray(modalities_b)
        ta = np.asarray(times_a) - self.lags[ma]
        tb = np.asarray(times_b) - self.lags[mb]
        result = np.empty((len(ta), len(tb)))
        for m in np.unique(ma):
            ia = np.flatnonzero(ma == m)
            for n in np.unique(mb):
                ib = np.flatnonzero(mb == n)
                delta = ta[ia, None] - tb[ib][None, :]
                unique, inverse = np.unique(delta, return_inverse=True)
                values = np.empty(len(unique))
                # Bound temporary state matrices even in an explicitly dense
                # diagnostic. Never form an observations² by states² tensor.
                for start in range(0, len(unique), 64):
                    section = slice(start, start + 64)
                    A, _ = self.transitions(np.abs(unique[section]), 1)
                    positive = np.einsum("i,tij,j->t", self.outputs[m], A, self.outputs[n])
                    negative = np.einsum("i,tji,j->t", self.outputs[m], A, self.outputs[n])
                    values[section] = np.where(unique[section] >= 0, positive, negative)
                result[np.ix_(ia, ib)] = values[inverse].reshape(delta.shape)
        return result
