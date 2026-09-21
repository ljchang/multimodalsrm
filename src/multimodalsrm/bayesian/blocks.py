"""Conditional parameter coordinates with an explicit fixed physical reference."""

import numpy as np

from ..data import readonly_array
from ._backend import runtime


def validate_blocks(blocks):
    if blocks is None:
        return
    if (
        not isinstance(blocks, (tuple, list))
        or not blocks
        or any(not isinstance(b, str) for b in blocks)
        or len(set(blocks)) != len(blocks)
        or any(b not in ("filter", "loading", "offset", "noise", "gp") for b in blocks)
    ):
        raise ValueError("blocks must be a nonempty unique sequence of parameter blocks")


class ParameterSubspace:
    """Sample selected blocks conditional on all remaining physical values.

    The reference is copied and immutable. Only active coordinates are
    transformed, so fixed boundary values remain exact and contribute no
    transform Jacobian. The full physical density supplies the conditional
    posterior up to a constant.
    """

    def __init__(self, problem, reference, *, blocks=None):
        validate_blocks(blocks)
        x = np.asarray(reference, float)
        if x.shape != problem.initial.shape or not np.isfinite(x).all():
            raise ValueError("reference must be a finite full physical parameter vector")
        if not np.isfinite(float(problem.objective(x))):
            raise ValueError("reference must have finite physical posterior density")
        if blocks is not None and any(not any(n[0] == b for n in problem.names) for b in blocks):
            raise ValueError("requested blocks must contain learned parameters")
        self.problem = problem
        self.reference = readonly_array(x)
        self.active_indices = tuple(
            i for i, n in enumerate(problem.names) if blocks is None or n[0] in blocks
        )
        active_set = set(self.active_indices)
        self.fixed_indices = tuple(i for i in range(len(problem.names)) if i not in active_set)
        groups = {}
        for active_position, physical_index in enumerate(self.active_indices):
            prior = problem.parameter_priors[physical_index]
            positions, indices = groups.setdefault(prior, ([], []))
            positions.append(active_position)
            indices.append(physical_index)
        self._transform_groups = tuple(
            (
                np.asarray(positions, dtype=int),
                np.asarray(indices, dtype=int),
                problem.transforms[indices[0]],
            )
            for positions, indices in groups.values()
        )
        self.active_parameters = [list(problem.names[i]) for i in self.active_indices]
        self.fixed_parameters = [
            dict(name=list(problem.names[i]), value=float(x[i])) for i in self.fixed_indices
        ]

    def from_unconstrained(self, z):
        _, jnp, _, _ = runtime()
        if np.shape(z) != (len(self.active_indices),):
            raise ValueError("unconstrained vector must match active parameters")
        z = jnp.asarray(z)
        x = jnp.asarray(self.reference)
        jac = jnp.asarray(0.0)
        for positions, indices, transform in self._transform_groups:
            unconstrained = z[positions]
            values = transform(unconstrained)
            x = x.at[indices].set(values)
            jac += jnp.sum(transform.log_abs_det_jacobian(unconstrained, values))
        return x, jac

    def to_unconstrained(self, x):
        x = np.asarray(x, float)
        if x.shape != self.reference.shape or not np.isfinite(x).all():
            raise ValueError("physical parameters must be a finite full vector")
        if any(
            not self.problem.parameter_priors[i].lower
            < x[i]
            < self.problem.parameter_priors[i].upper
            for i in self.active_indices
        ):
            raise ValueError("active parameters must lie strictly inside prior support")
        z = np.empty(len(self.active_indices), dtype=float)
        for positions, indices, transform in self._transform_groups:
            z[positions] = np.asarray(transform.inv(x[indices]))
        if not np.isfinite(z).all():
            raise ValueError("active parameters must lie strictly inside prior support")
        return z

    def potential(self, z):
        x, jac = self.from_unconstrained(z)
        return self.problem.objective(x) - jac
