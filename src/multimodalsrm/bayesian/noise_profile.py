"""Exact, bounded single-variance density differences for numerical polishing.

This optional repair never changes the fitted density. Unavailable profiles
leave the general polisher in charge. Grouped mode factors only the functional
system and retains at most a 1024-row selected precision block per run.
"""

from dataclasses import dataclass, field
from math import fsum

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, lu_factor, lu_solve

MAX_SELECTED_ROWS = 1024
RHS_CHUNK = 32
# Optional multi-factor repair only: at most 128 MiB per expanded float64
# matrix. This does not replace the caller's whole-process memory limit.
MAX_FACTOR_ROWS = 4096


@dataclass
class VarianceProfile:
    available: bool = False
    reason: str = ""
    gradient: float = np.nan
    curvature: float = np.nan
    variance: float = np.nan
    prior: object = None
    terms: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def _change_parts(self, delta):
        v, prior = self.variance, self.prior
        updated = v + delta
        if (
            not self.available
            or not np.isfinite([delta, updated]).all()
            or updated <= 0
            or not prior.lower <= updated <= prior.upper
        ):
            return None
        parts = []
        for lam, eta2 in self.terms:
            denominator = 1 + delta * lam
            if np.any(denominator <= 0) or not np.isfinite(denominator).all():
                return None
            parts.extend(0.5 * np.log1p(delta * lam))
            parts.extend(-0.5 * delta * eta2 / denominator)
        if prior.family == "lognormal":
            a = np.log(v) - prior.loc
            dlog = np.log1p(delta / v)
            parts.extend((dlog, dlog * a / prior.scale**2, 0.5 * dlog**2 / prior.scale**2))
        elif prior.family == "normal":
            parts.extend(
                (
                    delta * (v - prior.loc) / prior.scale**2,
                    delta**2 / (2 * prior.scale**2),
                )
            )
        return parts if np.isfinite(parts).all() else None

    def change(self, delta):
        """Exact negative log posterior difference; infinity outside support."""
        parts = self._change_parts(delta)
        return np.inf if parts is None else fsum(parts)

    def cancellation_bound(self, delta):
        """Conservative arithmetic cancellation screen, not a solver error bound.

        Scale by the *unsigned constituent terms*, before their cancellation.
        This is never an allowance for a positive density change. Fresh physical
        gradients separately guard the numerical solve used to build a profile.
        """
        parts = self._change_parts(delta)
        return np.inf if parts is None else 64 * np.finfo(float).eps * fsum(abs(p) for p in parts)

    def is_descent(self, delta):
        return self.change(delta) < -self.cancellation_bound(delta)


def _grouped_solver(problem, x, run, diagnostics):
    from .grouped import operands

    K, w, residual, variance, nodes = (np.asarray(a) for a in operands(problem, x, run))
    if not all(np.isfinite(a).all() for a in (K, w, residual, variance)):
        raise FloatingPointError("nonfinite grouped operands")
    if np.any(variance <= 0):
        raise FloatingPointError("nonpositive observation variance")
    if getattr(problem, "features", 1) > 1:
        return _multifactor_solver(K, w, residual, variance, nodes, diagnostics)

    def segment(a):
        result = np.zeros((len(K), a.shape[1]))
        np.add.at(result, nodes, a)
        return result

    precision = np.bincount(nodes, weights=w * w / variance, minlength=len(K))
    factor = lu_factor(np.eye(len(K)) + K * precision[None, :])

    def first(rhs):
        score = segment(w[:, None] * rhs / variance[:, None])
        q = lu_solve(factor, score, trans=1)
        return (rhs - w[:, None] * (K @ q)[nodes]) / variance[:, None]

    def product(a):
        return variance[:, None] * a + w[:, None] * (K @ segment(w[:, None] * a))[nodes]

    def solve0(rhs):
        answer = first(rhs)
        for _ in range(2):
            answer += first(rhs - product(answer))
        return answer

    if run in problem.baseline_designs and problem.baseline_designs[run][0].shape[1]:
        U = problem.baseline_designs[run][0]
        V = np.column_stack(
            [solve0(U[:, begin : begin + RHS_CHUNK]) for begin in range(0, U.shape[1], RHS_CHUNK)]
        )
        small = cho_factor(np.eye(U.shape[1]) + U.T @ V, lower=True)

        def base(rhs):
            z = solve0(rhs)
            return z - V @ cho_solve(small, U.T @ z)

        def solve(rhs):
            answer = base(rhs)
            for _ in range(2):
                answer += base(rhs - product(answer) - U @ (U.T @ answer))
            return answer
    else:
        solve = solve0
    diagnostics["functional_rows"] = len(K)
    return solve, residual


def _multifactor_solver(K, w, residual, variance, nodes, diagnostics):
    """Woodbury solve without an observation map or inverse latent covariance.

    Node-major factor blocks form B = I + (K tensor I) D. D accumulates
    individual loading outer products, retaining zero/cancelling loadings.
    Applying K separately to each factor avoids constructing its Kronecker
    expansion or an explicit block-diagonal D.
    """
    m, factors = len(K), w.shape[1]
    size = m * factors
    diagnostics.update(temporal_nodes=m, factor_system_rows=size, functional_rows=m)
    if size > MAX_FACTOR_ROWS:
        raise ValueError(f"factor-system capacity exceeds {MAX_FACTOR_ROWS}")
    blocks = np.zeros((m, factors, factors))
    for begin in range(0, len(w), RHS_CHUNK):
        selected = slice(begin, begin + RHS_CHUNK)
        for k in range(factors):
            np.add.at(
                blocks[:, k, :],
                nodes[selected],
                w[selected, k, None] * w[selected] / variance[selected, None],
            )
    B = np.einsum("ij,jab->iajb", K, blocks).reshape(size, size)
    B.flat[:: size + 1] += 1.0
    factor = lu_factor(B, overwrite_a=True)

    def scatter(rhs):
        score = np.zeros((m, factors, rhs.shape[1]))
        for k in range(factors):
            np.add.at(score[:, k, :], nodes, w[:, k, None] * rhs)
        return score.reshape(size, -1)

    def gather(score):
        latent = (K @ score.reshape(m, -1)).reshape(m, factors, -1)
        result = np.zeros((len(w), latent.shape[2]))
        for k in range(factors):
            result += w[:, k, None] * latent[:, k, :][nodes]
        return result

    def first(rhs):
        q = lu_solve(factor, scatter(rhs / variance[:, None]), trans=1)
        return (rhs - gather(q)) / variance[:, None]

    def product(a):
        return variance[:, None] * a + gather(scatter(a))

    def solve(rhs):
        answer = first(rhs)
        for _ in range(2):
            answer += first(rhs - product(answer))
        error = np.max(np.abs(rhs - product(answer))) / max(1.0, np.max(np.abs(rhs)))
        if not np.isfinite(error):
            raise FloatingPointError("nonfinite grouped solve residual")
        diagnostics["maximum_relative_solve_residual"] = max(
            diagnostics.get("maximum_relative_solve_residual", 0.0), float(error)
        )
        return answer

    return solve, residual


def _dense_solver(problem, x, run, diagnostics):
    C = np.asarray(problem.covariance(x, run))
    factor = cho_factor(C, lower=True)
    _, offsets, _, _, _ = problem.arrays(x)
    residual = problem.systems[run].values - np.asarray(offsets)[problem._packed[run][0]]

    def solve(rhs):
        answer = cho_solve(factor, rhs)
        for _ in range(2):
            answer += cho_solve(factor, rhs - C @ answer)
        return answer

    return solve, residual


def variance_profile(problem, x, coordinate):
    """Build a physical variance profile, or return ``available=False`` + reason.

    Spectral mode and runs with more than MAX_SELECTED_ROWS matching observations
    are explicitly unsupported by this optional repair. Other runs contribute
    zero. Identity RHS columns are constructed in bounded chunks, never as an
    observation identity or observation covariance in grouped mode.
    """
    profile = VarianceProfile()
    try:
        if getattr(problem, "noise_timescales", None):
            raise ValueError("variance profile is unavailable for correlated noise")
        mode = getattr(problem, "linear_algebra", None)
        if mode not in ("dense", "grouped"):
            raise ValueError("variance profile supports dense/grouped modes only")
        name = problem.names[coordinate]
        if name[0] != "noise":
            raise ValueError("coordinate is not a noise variance")
        v, prior = float(x[coordinate]), problem.parameter_priors[coordinate]
        if not np.isfinite(v) or v <= 0 or not prior.lower <= v <= prior.upper:
            raise ValueError("variance is outside physical prior support")
        selected = {
            run: np.flatnonzero([key[:2] == name[1:] for key in system.keys])
            for run, system in problem.systems.items()
        }
        if any(len(rows) > MAX_SELECTED_ROWS for rows in selected.values()):
            raise ValueError(f"selected-row capacity exceeds {MAX_SELECTED_ROWS}")
        profile.diagnostics = dict(method=f"{mode}_selected_precision_eigen", runs=[])
        gradient_parts, curvature_parts = [], []
        for run, rows in selected.items():
            if not len(rows):
                continue
            detail = dict(run=run, selected_rows=len(rows), rhs_batches=0, maximum_rhs_columns=0)
            builder = _grouped_solver if mode == "grouped" else _dense_solver
            solve, residual = builder(problem, x, run, detail)
            P = np.empty((len(rows), len(rows)))
            for begin in range(0, len(rows), RHS_CHUNK):
                indices = rows[begin : begin + RHS_CHUNK]
                rhs = np.zeros((len(residual), len(indices)))
                rhs[indices, np.arange(len(indices))] = 1.0
                P[:, begin : begin + len(indices)] = solve(rhs)[rows]
                detail["rhs_batches"] += 1
                detail["maximum_rhs_columns"] = max(detail["maximum_rhs_columns"], len(indices))
            alpha = solve(residual[:, None])[rows, 0]
            detail["rhs_batches"] += 1
            detail["relative_precision_asymmetry"] = float(
                np.max(np.abs(P - P.T)) / max(1.0, np.max(np.abs(P)))
            )
            lam, Q = eigh((P + P.T) * 0.5, check_finite=True)
            eta2 = (Q.T @ alpha) ** 2
            if np.any(lam <= 0) or not np.isfinite(eta2).all():
                raise FloatingPointError("selected precision is not finite positive definite")
            detail["minimum_precision_eigenvalue"] = float(lam[0])
            profile.diagnostics["runs"].append(detail)
            profile.terms.append((lam, eta2))
            gradient_parts.extend(0.5 * lam)
            gradient_parts.extend(-0.5 * eta2)
            curvature_parts.extend(-0.5 * lam**2)
            curvature_parts.extend(eta2 * lam)
        if prior.family == "lognormal":
            a = np.log(v) - prior.loc
            gradient_parts.append((1 + a / prior.scale**2) / v)
            curvature_parts.append((1 / prior.scale**2 - 1 - a / prior.scale**2) / v**2)
        elif prior.family == "normal":
            gradient_parts.append((v - prior.loc) / prior.scale**2)
            curvature_parts.append(1 / prior.scale**2)
        profile.gradient, profile.curvature = (
            fsum(gradient_parts),
            fsum(curvature_parts),
        )
        if not np.isfinite([profile.gradient, profile.curvature]).all():
            raise FloatingPointError("nonfinite profile derivatives")
        profile.variance, profile.prior, profile.available = v, prior, True
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        profile.reason = f"{type(exc).__name__}: {exc}"
    return profile
